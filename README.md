# 🌌 SAM3_Few_Shot_Auto_Labeler

이 프로젝트는 **SAM 3 (Segment Anything Model 3)**의 비디오 전파(Video Propagation) 알고리즘을 배치 이미지 처리에 응용한 **고성능 자동 라벨링 및 결함 검출 도구**입니다. 사용자가 최소한의 참조 이미지에 가이드를 주면, 무한대의 타겟 이미지에서 동일한 객체를 추적하고 마스킹합니다.

![SAM 3 architecture](assets/pipeline.png?raw=true) 

## User interface
![User interface](assets/image.png?raw=true) 

## 📋 핵심 기술 스택

* **Core Model:** SAM 3 (Segment Anything Model 3)
* **Compute:** PyTorch 2.9.0+cu128, CUDA 12.8
* **UI/UX:** Matplotlib Interactive Backend, `SmartSelector` GUI
* **Processing:** Batch Inference with Memory Management

## 💻 설치 및 환경 설정 (Setup)

본 프로젝트는 특정 버전의 CUDA 및 PyTorch 환경에서 테스트되었습니다. 아래 환경을 권장합니다.

### 1. 주요 요구 패키지

* `torch` (>= 2.9.0)
* `torchvision` (>= 0.24.0)
* `opencv-python`
* `matplotlib`
* `sam3` (Editable local install: `/workspace/sam3`)

### 2. **레포지토리 클론 및 설치**

```bash
git clone https://github.com/JuWanMaeng/SAM3_Few_Shot_Auto_Labelling_Model.git
cd SAM3_Few_Shot_Auto_Labelling_Model
pip install -e .
pip install -r requirements.txt
```

### 3. 가중치(Weights) 준비

SAM 3 체크포인트 파일을 다음 경로에 배치하십시오.
`models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.ptt`

## 🛠 주요 기능 및 조작법

### 1. 하이브리드 인터랙티브 라벨링 (`gui.py`)

참조 이미지에서 정밀한 마스크를 생성하기 위해 3가지 모드를 제공합니다.

| 모드 | 조작법 | 용도 |
| --- | --- | --- |
| **Box** | 좌클릭 드래그 | 객체의 대략적인 범위 지정 |
| **Point** | 좌클릭(+), 우클릭(-) | 특정 위치 포함/제외 힌트 |
| **Brush** | `B` 키 전환 후 드래그 | 세밀한 엣지 보정 (휠로 브러시 크기 조절) |

### 2. 추론 (`main.py`)

* **Reference Encoding:** 사용자가 라벨링한 정보를 SAM 3의 Feature 공간으로 임베딩합니다.
* **Propagation:** 비디오 프레임 간의 연속성을 추적하는 기술을 활용하여 타겟 이미지 간의 객체 일관성을 유지합니다.
* **Visualization:** 추론 결과를 Mask, BBox, Overlay 이미지로 자동 변환하여 저장합니다.

## 🚀 실행 가이드

1. **`config.py` 수정:** 결함 타입(`defect`)과 데이터 경로를 설정합니다.
2. **프로그램 시작:** `python main.py` 실행.
3. **가이드 입력:** 팝업되는 GUI 창에서 참조 이미지에 결함 영역을 표시합니다.
4. **결과 확인:** `output/` 폴더 내의 시각화 데이터를 분석합니다.

---

## 📂 출력물 구조 (Output)

```text
output/
├── id_masks/    # 객체 ID별 (1~16) 개별 이진 마스크
├── masks/       # 모든 객체가 통합된 컬러 마스크
├── overlays/    # 원본 + 마스크 + BBox (검토용)
└── bboxes/      # [ID, xmin, ymin, xmax, ymax] 좌표 정보

```

