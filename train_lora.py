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
        mask_filename = img_filename.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_path, mask_filename)
        
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
    find_stage = FindStage(
            img_ids=torch.tensor([0], device=CONFIG['device'], dtype=torch.long),
            text_ids=torch.tensor([0], device=CONFIG['device'], dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
    

    # 1. 모델 로드
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    
    print("Loading SAM3 Model...")
    original_model = build_sam3_image_model(
        bpe_path=bpe_path, 
        checkpoint_path=CONFIG['checkpoint_path']
    )
    
    # Processor 생성 코드 삭제! 불필요!

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

    # 3. 데이터셋 (이제 폴더 경로만 주면 됩니다)
    dataset = SemiconductorDataset(folder_path='data/images')
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)

    # 4. Optimizer & Loss (이전과 동일)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCEWithLogitsLoss() # 간단하게 이것부터 시작

    # 5. Training Loop
    model.train()
    print("Start Training...")
    
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        for images, masks, text_prompt in dataloader:
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])
            prompt = text_prompt[0]


            optimizer.zero_grad()

            # --- Forward Pass (수동 제어) ---
            train_state = {}
            train_state['backbone_out'] = model.backbone.forward_image(images)
            text_out = model.backbone.forward_text([prompt])
            train_state['backbone_out'].update(text_out)
            if "geometric_prompt" not in train_state:
                train_state["geometric_prompt"] = model._get_dummy_prompt()

            output = model.forward_grounding(
                    backbone_out=train_state["backbone_out"],
                    find_input=find_stage,
                    geometric_prompt=train_state["geometric_prompt"],
                    find_target=None,
            )

            # 1. 예측 마스크 가져오기 (Low Resolution Logits)
            # shape: [Batch, Queries, H_low, W_low]
            pred_masks_low = output["pred_masks"] 
            # shape: [Batch, 200] -> 예: [1, 200]
            pred_logits = output["pred_logits"] 
            
            # 2. 가장 점수가 높은(확신하는) 인덱스 찾기
            # 각 배치마다 점수가 가장 높은 인덱스를 뽑습니다.
            # batch_best_indices shape: [Batch] -> 예: [42] (42번째 후보가 1등)
            batch_best_indices = torch.argmax(pred_logits, dim=1)

            # 3. 1등 마스크만 골라내기
            # pred_masks_low shape: [Batch, 200, 288, 288]
            # 여기서 위에서 찾은 인덱스에 해당하는 마스크만 뽑습니다.
            batch_indices = torch.arange(pred_logits.shape[0], device=pred_logits.device)
            pred_masks_selected = pred_masks_low[batch_indices, batch_best_indices] # shape: [Batch, 288, 288]
            
            # 4. 정답 마스크 크기에 맞춰 업샘플링 
            pred_masks_high = torch.nn.functional.interpolate(
                pred_masks_selected,
                size=(masks.shape[-2], masks.shape[-1]), 
                mode="bilinear",
                align_corners=False
            )

            # 5. Loss 계산
            loss = criterion(pred_masks_high, masks)

            # 6. 역전파
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Epoch 종료 후 로그 출력
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

    # 학습 완료 후 저장
    model.save_pretrained("sam3_lora_finetuned")
    print("Saved LoRA weights.")

if __name__ == "__main__":
    main()