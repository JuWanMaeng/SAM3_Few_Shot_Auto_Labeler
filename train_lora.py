import os, glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import v2
from sam3.model.data_misc import FindStage
from tqdm import tqdm

# SAM3 관련
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# LoRA 관련
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    "checkpoint_path": 'models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt', # 경로 수정 필요
    "lora_r": 8,
    "lora_alpha": 16,
    "target_modules": ["qkv", "proj"],
    "lr": 1e-4,
    "batch_size": 2,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. 전처리
# ==========================================
def get_transforms(resolution):
    return v2.Compose([
        # 1. PIL 이미지를 텐서로 변환 (0~255 유지)
        v2.ToImage(), 
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((resolution, resolution), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# ==========================================
# 3. Dataset
# ==========================================
class SemiconductorDataset(Dataset):
    def __init__(self, folder_path, resolution=1008):
        self.folder_path = folder_path  
        self.mask_path = folder_path.replace('images', 'masks')
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
        self.resolution = resolution
        self.transform = get_transforms(resolution)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        
        # 1. 이미지 로드
        image_path = os.path.join(self.folder_path, img_filename)
        image_pil = Image.open(image_path).convert("RGB")
        
        # Transform 적용 (PIL -> Tensor -> Resize -> Normalize 한방에 처리)
        image_tensor = self.transform(image_pil)

        # 2. 마스크 로드
        mask_path = os.path.join(self.mask_path, img_filename)
        
        mask_pil = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_pil)
        mask_np = (mask_np > 128).astype(np.uint8) # 이진화

        # 마스크 리사이즈 (Nearest Neighbor 필수)
        mask_resized = cv2.resize(mask_np, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)

        text_prompt = 'particle'

        return image_tensor, mask_tensor, text_prompt


# ==========================================
# 4. Main
# ==========================================
def main():
    # 1. 모델 로드
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    
    print("Loading SAM3 Model...")
    original_model = build_sam3_image_model(
        bpe_path=bpe_path, 
        checkpoint_path=CONFIG['checkpoint_path']
    )
    
    # 2. LoRA 적용
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['target_modules'],
        lora_dropout=0.05,
        bias="none",
        task_type=None
    )
    
    model = get_peft_model(original_model, peft_config)
    model.to(CONFIG['device'])
    model.print_trainable_parameters()

    # 3. 데이터셋
    # (주의: Dataset에서 mask_tensor의 shape는 [1, H, W] 여야 합니다)
    dataset = SemiconductorDataset(folder_path='data/images') 
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    model.train()
    print(f"Start Training (Batch Size: {CONFIG['batch_size']})...")
    
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", ncols=30)
        for batch_idx, (images, masks, text_prompt) in enumerate(progress_bar):
            # images shape: [B, 3, 1008, 1008]
            # masks shape:  [B, 1, 1008, 1008]
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])
            
            # [수정 1] Text Prompt 리스트 변환
            # DataLoader는 텍스트를 ('particle', 'particle') 튜플로 줍니다. 리스트로 변환 필요.
            prompt_list = list(text_prompt) 

            # [수정 2] FindStage 동적 생성 (Batch 대응)
            # 배치 사이즈만큼 인덱스를 생성해야 합니다. 예: [0, 1]
            current_batch_size = images.shape[0]
            batch_ids = torch.arange(current_batch_size, device=CONFIG['device'], dtype=torch.long)
            
            find_stage = FindStage(
                img_ids=batch_ids,   # [0, 1, ... B-1]
                text_ids=batch_ids,  # [0, 1, ... B-1] (이미지와 텍스트 1:1 매칭 시)
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            optimizer.zero_grad()

            # --- Forward Pass ---
            train_state = {}
            
            # (1) Image Backbone
            train_state['backbone_out'] = model.backbone.forward_image(images)
            
            # (2) Text Backbone ([수정] 리스트 전체 전달)
            text_out = model.backbone.forward_text(prompt_list)
            train_state['backbone_out'].update(text_out)
            
            if "geometric_prompt" not in train_state:
                train_state["geometric_prompt"] = model._get_dummy_prompt(current_batch_size) # 배치 크기 전달

            # (3) Grounding
            output = model.forward_grounding(
                    backbone_out=train_state["backbone_out"],
                    find_input=find_stage,
                    geometric_prompt=train_state["geometric_prompt"],
                    find_target=None,
            )

            # --- 결과 처리 및 Loss 계산 ---
            
            # pred_logits: [Batch, Queries]
            pred_logits = output["pred_logits"]
            
            # pred_masks_low: [Batch, Queries, H_low, W_low]
            pred_masks_low = output["pred_masks"]

            # 배치별로 가장 높은 점수의 마스크 인덱스 추출
            batch_best_indices = torch.argmax(pred_logits, dim=1) 
            
            # 마스크 선택 (Advanced Indexing)
            # 결과값 Shape 예상: [Batch, 2, 288, 288] (쿼리당 2개의 마스크가 나오는 경우)
            pred_masks_raw = pred_masks_low[batch_ids, batch_best_indices]
            
            # [핵심 수정] 마스크 차원 정리
            # 만약 4차원([B, K, H, W])으로 나왔다면, K개 중 첫번째 마스크만 선택합니다.
            if pred_masks_raw.dim() == 4:
                # [Batch, 2, H, W] -> [Batch, H, W] (첫 번째 마스크 선택)
                pred_masks_raw = pred_masks_raw[:, 0, :, :]
            
            # 차원 복구: [Batch, H, W] -> [Batch, 1, H, W]
            pred_masks_selected = pred_masks_raw.unsqueeze(1) 

            # 5. 정답 마스크 크기에 맞춰 업샘플링 (이제 입력이 4차원이므로 정상 작동함)
            pred_masks_high = torch.nn.functional.interpolate(
                pred_masks_selected,
                size=(masks.shape[-2], masks.shape[-1]), 
                mode="bilinear",
                align_corners=False
            )

            # Loss 계산
            loss = criterion(pred_masks_high, masks)

            # 역전파
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

    # 저장
    model.save_pretrained("sam3_lora_finetuned")
    print("Saved LoRA weights.")

if __name__ == "__main__":
    main()