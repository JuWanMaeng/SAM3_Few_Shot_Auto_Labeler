# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
from collections import OrderedDict

import torch

from sam3.model.sam3_tracker_base import concat_points, NO_OBJ_SCORE, Sam3TrackerBase
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.utils.sam2_utils import load_video_frames
from tqdm.auto import tqdm


class Sam3TrackerPredictor(Sam3TrackerBase):
    """
    The demo class that extends the `Sam3TrackerBase` to handle user interactions
    and manage inference states, with support for multi-object tracking.
    """

    def __init__(
        self,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        # if fill_hole_area > 0, we fill small holes in the final masks up to this area (after resizing them to the original video resolution)
        fill_hole_area=0,
        # if always_start_from_first_ann_frame is True, we always start tracking from the frame where we receive the first annotation (clicks or mask)
        # and ignore the `start_frame_idx` passed to `propagate_in_video`
        always_start_from_first_ann_frame=False,
        # the maximum number of points to be used in the prompt encoder, which reduce the domain gap between training (that only has 8 points)
        # - if it's set to a positive integer, we only take the `max_point_num_in_prompt_enc//2` points and
        #   the last `(max_point_num_in_prompt_enc - max_point_num_in_prompt_enc//2)` points in the prompt encoder
        # - if it's set to 0 or negative, this option is turned off and we use all points in the prompt encoder
        max_point_num_in_prompt_enc=16,
        non_overlap_masks_for_output=True,
        # checkpoint_file=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output

        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.bf16_context.__enter__()  # keep using for the entire model process

        self.iter_use_prev_mask_pred = True
        self.add_all_frames_to_correct_as_cond = True

    @torch.inference_mode()
    def init_state(
        self,
        video_height=None,
        video_width=None,
        num_frames=None,
        video_path=None,
        cached_features=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """
        추론 상태(inference state)를 초기화합니다.
        비디오 프레임, 입력 프롬프트, 모델 출력 결과 등을 저장할 딕셔너리를 생성하고 설정합니다.
        """
        inference_state = {}
        
        # 비디오 프레임(이미지)을 CPU 메모리로 오프로드할지 여부 설정
        # 이 옵션을 켜면 약간의 오버헤드만으로 GPU 메모리를 크게 절약할 수 있습니다.
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        
        # 추론 상태(state) 자체를 CPU 메모리로 오프로드할지 여부 설정
        # 이 옵션을 켜면 GPU 메모리는 절약되지만, 추적 FPS(속도)가 저하됩니다.
        # (예: 768x768 모델 테스트 시, 객체 1개 추적 27->24 FPS, 객체 2개 추적 24->21 FPS로 하락)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        
        # 현재 연산에 사용할 디바이스 설정 (주로 CUDA)
        inference_state["device"] = self.device
        
        # 데이터를 저장할 디바이스 설정 (옵션에 따라 CPU 또는 CUDA)
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")

        # 비디오 경로가 주어진 경우, 비디오 프레임을 로드합니다.
        if video_path is not None:
            images, video_height, video_width = load_video_frames(
                video_path=video_path,
                image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=async_loading_frames,
                compute_device=inference_state["storage_device"],
            )
            inference_state["images"] = images
            inference_state["num_frames"] = len(images)
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
        else:
            # 비디오 경로가 없는 경우, 메타데이터(크기, 프레임 수)만 설정
            # (최종 출력 점수 등을 리사이징할 때 원본 해상도 정보로 사용됨)
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
            inference_state["num_frames"] = num_frames
            
        # 각 프레임별 입력(포인트, 마스크)을 저장할 딕셔너리 초기화
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        
        # 빠른 상호작용을 위해 최근 방문한 소수의 프레임에 대한 시각적 특징(feature)을 캐싱
        inference_state["cached_features"] = (
            {} if cached_features is None else cached_features
        )
        
        # 프레임 간 변하지 않는 상수 값들을 저장 (메모리 절약을 위해 한 카피만 유지)
        inference_state["constants"] = {}
        
        # 클라이언트 측 객체 ID와 모델 내부 객체 인덱스 간의 매핑 정보
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        
        # 각 프레임에서의 모델 추적 결과와 상태를 저장할 저장소
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},      # 조건부(프롬프트가 있는) 프레임의 출력 결과 {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # 비조건부(추론된) 프레임의 출력 결과 {frame_idx: <out>}
        }
        
        # 첫 번째 어노테이션(입력)을 받은 프레임의 인덱스
        inference_state["first_ann_frame_idx"] = None
        
        # 각 객체별 추적 결과를 볼 수 있는 뷰(Slice), "output_dict"와 메모리를 공유함
        inference_state["output_dict_per_obj"] = {}
        
        # 사용자가 프레임과 상호작용(클릭/마스크 추가)할 때 새로운 출력을 임시 저장할 공간
        # (전파(propagation)가 시작되기 전에 "output_dict"로 병합됨)
        inference_state["temp_output_dict_per_obj"] = {}
        
        # 클릭이나 마스크 입력으로부터 통합된(consolidated) 출력을 이미 가지고 있는 프레임 인덱스들
        # (추적 중에는 이 통합된 출력을 바로 사용함)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),      # 조건부 프레임 인덱스 집합
            "non_cond_frame_outputs": set(),  # 비조건부 프레임 인덱스 집합
        }
        
        # 각 추적 프레임에 대한 메타데이터 (예: 어느 방향으로 추적되었는지 등)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        
        # 비디오 내의 모든 기존 포인트 입력을 초기화
        self.clear_all_points_in_video(inference_state)
        
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        rel_coordinates=True,
        use_prev_mem_frame=False,
        normalize_coords=True,
        box=None,
    ):
        """
        [기능] 특정 프레임에 새로운 점(Points)이나 박스(Box) 프롬프트를 추가합니다.
        사용자가 클릭하거나 박스를 그릴 때 호출되는 함수입니다.
        """
        # 1. 클라이언트의 obj_id(예: 1)를 모델 내부의 인덱스(예: 0)로 변환, x번 id의 대한 값들 초기화
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        
        # 해당 객체의 프롬프트 입력 저장소 가져오기
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        # 2. 입력값 유효성 검사 (Validation)
        if (points is not None) != (labels is not None):
            raise ValueError("points와 labels는 반드시 함께 제공되어야 합니다.")
        if points is None and box is None:
            raise ValueError("points 또는 box 중 하나는 반드시 입력되어야 합니다.")

        # 3. 텐서(Tensor) 변환 및 초기화
        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
            
        # 배치(Batch) 차원 추가 (SAM 모델 입력 규격 맞춤)
        if points.dim() == 2:
            points = points.unsqueeze(0)  
        if labels.dim() == 1:
            labels = labels.unsqueeze(0) 

        # 4. 좌표계 변환 (상대 좌표 0~1 -> 절대 좌표 픽셀)
        if rel_coordinates:
            if points is not None:
                points = points * self.image_size
            if box is not None:
                box = box * self.image_size

        # =========================================================================
        # [중요] 박스 처리 로직 
        # 박스(Box)가 들어오면 내부적으로 점(Points)으로 변환합니다.
        # SAM 2/3 학습 방식에 맞춰 좌상단(Label 2)과 우하단(Label 3) 점으로 처리합니다.
        # =========================================================================
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "박스를 추가할 때는 기존 포인트를 지워야 합니다 (clear_old_points=True)."
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            
            # [버그 원인 지점] 여기서 box를 (1, 2, 2)로 reshape 하므로
            # 박스가 여러 개(N개) 들어오면 에러가 났던 것입니다.
            box_coords = box.reshape(1, 2, 2)
            
            # 박스 시작점(2)과 끝점(3) 라벨 생성
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            
            # 기존 포인트와 박스 포인트를 합침
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        # 5. 프롬프트 저장
        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        # 기존 포인트 뒤에 새 포인트를 이어 붙임
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None) # 포인트가 들어오면 마스크 입력은 삭제 (충돌 방지)

        # 6. 추론 모드 결정 (초기 입력인지, 수정인지)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        
        # 역방향 추적 여부 결정
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
            
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 7. 입력 포인트 개수 제한 (학습 환경과의 차이 줄이기 위함)
        # 너무 많은 점을 찍으면 앞쪽 절반, 뒤쪽 절반만 남기고 중간은 버림
        num_points = point_inputs["point_coords"].size(1)
        if num_points > self.max_point_num_in_prompt_enc > 0:
            num_first = self.max_point_num_in_prompt_enc // 2
            num_last = self.max_point_num_in_prompt_enc - num_first
            point_inputs["point_coords"] = torch.cat(
                [
                    point_inputs["point_coords"][:, :num_first],
                    point_inputs["point_coords"][:, -num_last:],
                ],
                dim=1,
            )
            point_inputs["point_labels"] = torch.cat(
                [
                    point_inputs["point_labels"][:, :num_first],
                    point_inputs["point_labels"][:, -num_last:],
                ],
                dim=1,
            )
            logging.warning(
                f"너무 많은 포인트({num_points})가 입력되었습니다. 처음 {num_first}개와 마지막 {num_last}개만 사용됩니다."
            )

        # 8. 이전 마스크 결과 재활용 (Iterative refinement)
        # 사용자가 수정을 위해 클릭을 추가한 경우, 이전 추론 결과(Logits)를 힌트로 사용
        prev_sam_mask_logits = None
        if self.iter_use_prev_mask_pred:
            prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
                if prev_out is None:
                    prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

            if prev_out is not None and prev_out["pred_masks"] is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
                prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        # 9. 단일 프레임 추론 실행 (SAM Core)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict, 
            frame_idx=frame_idx,
            batch_size=1, 
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False, # 클릭 단계에서는 메모리 인코딩을 하지 않음 (속도 최적화)
            prev_sam_mask_logits=prev_sam_mask_logits,
            use_prev_mem_frame=use_prev_mem_frame,
        )
        
        # 결과를 임시 저장소에 저장
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # 10. 결과 병합 및 정리 (Consolidation)
        # 다른 객체들과의 충돌을 해결하고 최종 마스크 생성
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        
        # 원본 해상도 마스크 추출
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None 
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        add_mask_to_memory=False,
    ):
        """
        [기능] 특정 프레임에 마스크(Mask) 자체를 프롬프트로 추가합니다.
        사용자가 붓(Brush) 도구로 직접 칠해서 입력을 줄 때 사용됩니다.
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        # 1. 마스크 차원 확인
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None] # (Batch, Channel, H, W) 차원 추가
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # 2. 마스크 리사이징 (모델 입력용 256x256)
        if mask_H != self.input_mask_size or mask_W != self.input_mask_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.input_mask_size, self.input_mask_size),
                align_corners=False,
                mode="bilinear",
                antialias=True, 
            )
        else:
            mask_inputs = mask_inputs_orig

        # 3. 마스크 리사이징 (원본 비디오 해상도용)
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        if mask_H != video_H or mask_W != video_W:
            mask_inputs_video_res = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(video_H, video_W),
                align_corners=False,
                mode="bilinear",
                antialias=True, 
            )
        else:
            mask_inputs_video_res = mask_inputs_orig
            
        # 이진화 (0.5 기준 Thresholding)
        mask_inputs_video_res = mask_inputs_video_res > 0.5

        # 입력 저장
        mask_inputs_per_frame[frame_idx] = mask_inputs_video_res
        point_inputs_per_frame.pop(frame_idx, None) # 마스크가 들어오면 포인트 입력 삭제

        # 4. 추론 방향 및 조건 설정
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
            
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 5. 추론 실행
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict, 
            frame_idx=frame_idx,
            batch_size=1, 
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs, # 포인트 대신 마스크 전달
            reverse=reverse,
            run_mem_encoder=False,
        )
        
        # 6. [중요] 사용자가 그린 마스크를 그대로 출력으로 사용
        # SAM이 예측한 것보다 사용자가 직접 칠한 것을 우선시합니다 (편집 경험 향상).
        # NO_OBJ_SCORE는 배경(-), -NO_OBJ_SCORE는 전경(+)을 의미
        current_out["pred_masks"] = None
        current_out["pred_masks_video_res"] = torch.where(
            mask_inputs_video_res, -NO_OBJ_SCORE, NO_OBJ_SCORE
        )
        
        # 결과 임시 저장
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        
        # 7. 다른 객체에서 겹치는 부분 제거 (Non-overlapping)
        # "내가 칠한 부분은 내 땅이니까 다른 객체들은 여기서 비켜"
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        for obj_idx2, obj_temp_output_dict2 in temp_output_dict_per_obj.items():
            if obj_idx2 == obj_idx:
                continue
            current_out2 = obj_temp_output_dict2[storage_key].get(frame_idx, None)
            if current_out2 is not None and "pred_masks_video_res" in current_out2:
                # 겹치는 부분을 배경(NO_OBJ_SCORE)으로 덮어씀
                current_out2["pred_masks_video_res"] = torch.where(
                    mask_inputs_video_res,
                    NO_OBJ_SCORE,
                    current_out2["pred_masks_video_res"],
                )

        # 8. 최종 병합 및 반환
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None 
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """[Deprecated] add_new_points_or_box를 대신 사용하세요."""
        return self.add_new_points_or_box(*args, **kwargs)

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks_for_output:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks, self.fill_hole_area
            )
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.low_res_mask_size
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value=10.0,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        if self.use_memory_selection:
            consolidated_out["iou_score"] = torch.full(
                size=(batch_size, 1),
                fill_value=0.0,
                dtype=torch.float32,
                device=inference_state["device"],
            )
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            # (use "pred_masks_video_res" if it's available)
            obj_mask = out.get("pred_masks_video_res", out["pred_masks"])
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                is_downsampling = "pred_masks_video_res" in out
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                    antialias=is_downsampling,  # use antialias for downsampling
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]
            if self.use_memory_selection:
                consolidated_out["iou_score"][obj_idx : obj_idx + 1] = out["iou_score"]
        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # A dummy (empty) mask with a single object
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=inference_state["device"],
        )

        # Retrieve correct image features
        (
            image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=None,
            mask_inputs=mask_inputs,
            gt_masks=None,
            frames_to_add_correction_pt=[],
            output_dict={
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state, run_mem_encoder=True):
        """
        추적(tracking)을 시작하기 전에 inference_state를 준비하고 임시 출력들을 통합(consolidate)합니다.
        """
        # 추적이 시작되었음을 표시합니다. 세션이 리셋되기 전까지는 새로운 객체 추가가 허용되지 않습니다.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # 각 객체별로 흩어져 있는 임시 출력("temp_output_dict_per_obj")을 통합하여
        # 메인 출력 저장소인 "output_dict"에 추가합니다.
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        
        # "consolidated_frame_inds"는 통합된 임시 출력이 추가된 프레임들의 인덱스를 저장합니다.
        # (이번 호출 또는 이전의 `propagate_in_video_preflight` 호출에서 추가된 것들)
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        
        # 조건부 출력(is_cond=True, 사용자 입력 기반)과 비조건부 출력(is_cond=False, 모델 추론 기반)을 분리하여 처리
        for is_cond in [False, True]:
            # 저장소 키 설정 ("cond_frame_outputs" 또는 "non_cond_frame_outputs")
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            
            # 임시 출력을 가지고 있는 모든 프레임 인덱스를 찾습니다.
            # (이들은 `add_new_points`나 `add_new_mask`를 통해 방금 클릭이나 마스크 입력을 받은 프레임들입니다.)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            
            # 통합된 프레임 인덱스 목록 업데이트
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            
            # 해당 프레임에 있는 모든 객체의 임시 출력을 하나로 통합(consolidate)합니다.
            # [중요] 여기서 run_mem_encoder=True이면 메모리 인코딩(기억 저장)이 수행됩니다.
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=run_mem_encoder,
                )
                # 통합된 결과를 메인 "output_dict"에 병합하고, 객체별 슬라이스(뷰)도 생성합니다.
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                
                # 다중 객체 상황이거나 배치 크기가 1 이하일 때, 입력 프레임 주변의 비조건부 메모리를 정리할지 결정
                # (새로운 입력이 들어왔으므로 주변의 오래된 추론 결과는 지워주는 것이 좋을 수 있음)
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # 입력 주변 프레임의 비조건부 메모리(이전 추론 결과)를 삭제합니다.
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # 처리가 끝났으므로 `temp_output_dict_per_obj`의 임시 출력을 비웁니다.
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # 예외 처리(Edge case): 만약 어떤 프레임에 "cond_frame_outputs"(사용자 입력 결과)가 추가되었다면,
        # 동일한 프레임에 있던 기존의 "non_cond_frame_outputs"(이전 추론 결과)는 삭제해야 합니다.
        # (사용자가 직접 수정한 내용이 모델이 추측한 내용보다 우선하기 때문)
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        
        # 객체별 딕셔너리에서도 동일하게 삭제
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        
        # 인덱스 집합에서도 삭제
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # "consolidated_frame_inds"에 있는 프레임들이 실제로 포인트나 마스크 입력이 있는 프레임들과
        # 정확히 일치하는지 확인합니다. (올바른 데모 워크플로우라면 일치해야 함)
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        
        assert all_consolidated_frame_inds == input_frames_inds
        
        # 첫 번째 상호작용(어노테이션)이 발생한 프레임 인덱스를 기록합니다. (추적 시작 지점으로 사용됨)
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = min(
                input_frames_inds, default=None
            )
            
        # 만약 `first_ann_frame_idx`가 현재 조건부 프레임 목록에 없다면 (예: 해당 프레임의 입력을 지운 경우),
        # 남아있는 조건부 프레임 중 가장 빠른 것을 선택합니다.
        if (
            inference_state["first_ann_frame_idx"]
            not in output_dict["cond_frame_outputs"]
        ):
            inference_state["first_ann_frame_idx"] = min(
                output_dict["cond_frame_outputs"], default=None
            )

    def _get_processing_order(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        num_frames = inference_state["num_frames"]
        # set start index, end index, and processing order
        if self.always_start_from_first_ann_frame:
            # in this case, we always start tracking from the frame where we receive
            # the initial annotation and ignore the provided start_frame_idx
            start_frame_idx = inference_state["first_ann_frame_idx"]
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                # this is the edge case where we start from frame 0 and track in reverse order;
                # in this case, we track a single frame (frame 0)
                processing_order = [0]
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        tqdm_disable=False,
        obj_ids=None,
        run_mem_encoder=True,
        propagate_preflight=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        if propagate_preflight:
            self.propagate_in_video_preflight(inference_state)
        # NOTE: This is a copy from the parent class, except that we return object scores as well.
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        if obj_ids is not None:
            raise NotImplementedError(
                "Per-object tracking yet for batched inference if not implemented."
            )
        obj_ids = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        processing_order = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        )

        for frame_idx in tqdm(
            processing_order, desc="propagate in video", disable=tqdm_disable
        ):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                obj_scores = current_out["object_score_logits"]
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            low_res_masks, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if self.use_memory_selection:
                obj_out["iou_score"] = current_out["iou_score"][obj_slice]
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def clear_all_points_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check and see if there are still any inputs left on this frame
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If this frame has no remaining inputs for any objects, we further clear its
        # conditioning frame status
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # The frame is not a conditioning frame anymore since it's not receiving inputs,
                # so we "downgrade" its output (if exists) to a non-conditioning frame output.
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # Similarly, do it for the sliced output on each object.
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # If all the conditioning frames have been removed, we also clear the tracking outputs
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None  # not needed by the demo
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    @torch.inference_mode()
    def clear_all_points_in_video(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            if self.backbone is None:
                raise RuntimeError(
                    f"Image features for frame {frame_idx} are not cached. "
                    "Please run inference on this frame first."
                )
            else:
                # Cache miss -- we will run inference on a single image
                image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)
                backbone_out = self.forward_image(image)
                # Cache the most recent frame's feature (for repeated interactions with
                # a frame; we can use an LRU cache for more frames in the future).
                inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        if "tracker_backbone_out" in backbone_out:
            backbone_out = backbone_out["tracker_backbone_out"]  # get backbone output

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            feat = feat.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["backbone_fpn"][i] = feat
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            use_prev_mem_frame=use_prev_mem_frame,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        if self.use_memory_selection:
            compact_current_out["iou_score"] = current_out["iou_score"]
            compact_current_out["eff_iou_score"] = current_out["eff_iou_score"]
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        image, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            image=image,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.clear_all_points_in_video(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        # (note that "consolidated_frame_inds" doesn't need to be updated in this step as
        # it's already handled in Step 0)
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # Step 3: For packed tensor storage, we index the remaining ids and rebuild the per-object slices.
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [
                    x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]
                ]
                # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][
                    remain_old_obj_inds
                ]
                if self.use_memory_selection:
                    out["iou_score"] = out["iou_score"][remain_old_obj_inds]
                    out["eff_iou_score"] = self.cal_mem_score(
                        out["object_score_logits"], out["iou_score"]
                    )  # recalculate the memory frame score
                # also update the per-object slices
                self._add_output_per_object(
                    inference_state, frame_idx, out, storage_key
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        # Step 4: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_frame_outputs.pop(t, None)

    def _suppress_shrinked_masks(
        self, pred_masks, new_pred_masks, shrink_threshold=0.3
    ):
        area_before = (pred_masks > 0).sum(dim=(-1, -2))
        area_after = (new_pred_masks > 0).sum(dim=(-1, -2))
        area_before = torch.clamp(area_before, min=1.0)
        area_ratio = area_after / area_before
        keep = area_ratio >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        pred_masks_after = torch.where(
            keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0)
        )
        return pred_masks_after

    def _suppress_object_pw_area_shrinkage(self, pred_masks):
        """
        This function suppresses masks that shrink in area after applying pixelwise non-overlapping constriants.
        Note that the final output can still be overlapping.
        """
        # Apply pixel-wise non-overlapping constraint based on mask scores
        pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
            pred_masks
        )
        # Fully suppress masks with high shrinkage (probably noisy) based on the pixel wise non-overlapping constraints
        # NOTE: The output of this function can be a no op if none of the masks shrinked by a large factor.
        pred_masks = self._suppress_shrinked_masks(
            pred_masks, pixel_level_non_overlapping_masks
        )
        return pred_masks

    def _apply_object_wise_non_overlapping_constraints(
        self, pred_masks, obj_scores, background_value=-10.0
    ):
        """
        Applies non-overlapping constraints object wise (i.e. only one object can claim the overlapping region)
        """
        # Replace pixel scores with object scores
        pred_masks_single_score = torch.where(
            pred_masks > 0, obj_scores[..., None, None], background_value
        )
        # Apply pixel-wise non-overlapping constraint based on mask scores
        pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        # Replace object scores with pixel scores. Note, that now only one object can claim the overlapping region
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks
