# Korea Times 스타일 위반 주입 & 모델 학습 파이프라인

정상 기사에 스타일 위반을 자동으로 주입하고, Detection/Correction 모델을 학습하는 통합 파이프라인

---

## 📋 목차

1. [환경 설정](#1-환경-설정-최초-1회)
2. [API 키 설정](#2-api-키-설정)
3. [체크포인트 다운로드](#3-체크포인트-다운로드)
4. [데이터 증강 (위반 주입)](#4-데이터-증강-위반-주입)
5. [학습 데이터 변환](#5-학습-데이터-변환)
6. [모델 학습](#6-모델-학습-gpu-필요)
7. [모델 평가](#7-모델-평가-gpu-필요)
8. [추론 (실전 사용)](#8-추론-실전-사용-gpu-필요)

---

## 1. 환경 설정 (최초 1회)

### 자동 설치 (권장)

```bash
bash setup_environment.sh
```

스크립트가 자동으로 다음 작업을 수행합니다:
- Conda 설치 확인
- `korea_times` 환경 생성 (Python 3.10)
- 필수 패키지 설치 (OpenAI, PyTorch, Transformers 등)
- GPU 감지 및 설정 안내

### 수동 설치

```bash
# 1. Conda 환경 생성
conda create -n korea_times python=3.10 -y

# 2. 환경 활성화
conda activate korea_times

# 3. 패키지 설치
pip install -r requirements.txt
```

### GPU 환경 (학습/추론 시 필요)

```bash
# Linux + CUDA 환경에서만 실행
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## 2. API 키 설정

OpenAI GPT API 키가 필요합니다.

### 방법 1: 환경 변수

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 방법 2: Python 코드 내

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

---

## 3. 체크포인트 다운로드

학습된 체크포인트 및 코드 전체는 다음 경로에서 다운로드할 수 있습니다:

**다운로드 링크**: https://drive.google.com/file/d/1u4sxxaVTviEZwdhF83NrLArPgdqq8_PN/view?usp=drive_link

다운로드 후 압축을 해제하여 `test_Inference/` 폴더에 배치하세요.

포함된 체크포인트:
- `checkpoint_2c_466/`: Detection + Correction 통합 (333MB)
- `detection_checkpoint_3300/`: Detection 전용 (167MB)
- `correction_checkpoint_3200/`: Correction 전용 (167MB)

---

## 4. 데이터 증강 (위반 주입)

정상 기사에 스타일 위반을 자동으로 주입합니다.

### 4.1. 기사 준비

기사는 다음 형식이어야 합니다:

```
[TITLE]
Samsung announces new smartphone
[/TITLE]

[BODY]
Samsung Electronics Chairman Lee Jae-yong unveiled the latest Galaxy model on Monday. The device costs 1,200,000 won and features advanced AI capabilities.
[/BODY]

[CAPTION]
Samsung Electronics Chairman Lee Jae-yong presents the new Galaxy phone at the launch event in Seoul on Monday. Yonhap.
[/CAPTION]
```

### 4.2. 증강 실행

파이프라인 코드는 별도로 제공됩니다. 주요 스크립트:
- `bulk_augmentation.py`: 벌크 증강
- `augment_multiple.py`: 다중 난이도 증강
- `violation_injector.py`: 위반 주입 엔진

난이도:
- `basic`: 1-2개 위반 주입
- `intermediate`: 3-5개 위반 주입
- `advanced`: 6-10개 위반 주입

출력: `bulk_augmentation_results_YYYYMMDD_HHMMSS.json`

소요 시간: 기사당 약 1-2분

---

## 5. 학습 데이터 변환

증강 결과를 Detection/Correction 학습 형식으로 변환하고, Train/Val로 자동 분할합니다.

```bash
python3 convert_augmentation_to_training.py \
    --input bulk_augmentation_results_*.json \
    --train-ratio 0.8 \
    --seed 42
```

출력:
- `detection_dataset/detection_train.jsonl` (학습용)
- `detection_dataset/detection_val.jsonl` (검증용)
- `correction_dataset/correction_train.jsonl` (학습용)
- `correction_dataset/correction_val.jsonl` (검증용)

---

## 6. 모델 학습 (GPU 필요)

Detection과 Correction 모델을 각각 학습합니다.

### 6.1. Detection 모델 학습

```bash
python3 train_detection_lora.py \
    --train-data detection_dataset/detection_train.jsonl \
    --val-data detection_dataset/detection_val.jsonl \
    --output-dir detection_lora_v1 \
    --max-steps 5000 \
    --batch-size 4
```

### 6.2. Correction 모델 학습

```bash
python3 train_correction_lora.py \
    --train-data correction_dataset/correction_train.jsonl \
    --val-data correction_dataset/correction_val.jsonl \
    --output-dir correction_lora_v1 \
    --max-steps 5000 \
    --batch-size 4
```

출력: LoRA 어댑터 (약 166MB 각)

소요 시간: 100 기사 기준 약 1시간 (각 모델)

베이스 모델: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (자동 다운로드)

---

## 7. 모델 평가 (GPU 필요)

학습된 모델의 성능을 평가합니다.

### 7.1. Detection 모델 평가

```bash
python3 evaluate_v2_lora.py \
    --model-path detection_lora_v1 \
    --test-file detection_dataset/detection_val.jsonl \
    --task detection
```

출력 지표:
- Rule-level F1, Precision, Recall
- Component-level 정확도

### 7.2. Correction 모델 평가

```bash
python3 evaluate_v2_lora.py \
    --model-path correction_lora_v1 \
    --test-file correction_dataset/correction_val.jsonl \
    --task correction
```

출력 지표:
- Exact Match
- Text Similarity

---

## 8. 추론 (실전 사용, GPU 필요)

학습된 모델 또는 기존 체크포인트로 실제 기사를 교정합니다.

### 8.1. 인터랙티브 모드

```bash
cd test_Inference

python3 inference_2c.py \
    --checkpoint checkpoint_2c_466 \
    --interactive
```

기사를 입력하면 위반 감지 → 교정까지 자동 수행합니다.

### 8.2. 파일 입력 모드

```bash
cd test_Inference

python3 inference_2c.py \
    --checkpoint checkpoint_2c_466 \
    --input article.txt \
    --output result.json
```

---

## 📁 파일 구조

```
koreatimes_training_pipeline/
├── README.md                              # 이 파일
├── .gitignore                             # Git 제외 파일
├── setup_environment.sh                   # 환경 설정 스크립트
├── requirements.txt                       # 필수 패키지 목록
│
├── violation_injector.py                  # 위반 주입 엔진 (3-Expert)
├── bulk_augmentation.py                   # 벌크 증강
├── augment_multiple.py                    # 다중 기사 증강
│
├── convert_augmentation_to_training.py    # 데이터 변환 + 분할
│
├── train_detection_lora.py                # Detection 학습
├── train_correction_lora.py               # Correction 학습
├── evaluate_v2_lora.py                    # 모델 평가
│
└── test_Inference/                        # 추론용
    ├── inference_2c.py                    # 2-component 추론
    ├── inference_simple.py                # 간단한 추론
    ├── checkpoint_2c_466/                 # 통합 체크포인트 (다운로드 필요)
    ├── detection_checkpoint_3300/         # Detection 체크포인트 (다운로드 필요)
    └── correction_checkpoint_3200/        # Correction 체크포인트 (다운로드 필요)
```

---

## 🔧 주요 설정

### 위반 주입 설정

- 난이도: `basic` (1-2개), `intermediate` (3-5개), `advanced` (6-10개)
- 모델: GPT-4o 이상 권장
- Expert 구조: Formatting, Quotation & Naming, Grammar

### 학습 설정

- 베이스 모델: Qwen2.5-7B (7B 파라미터)
- 학습 방식: LoRA (Low-Rank Adaptation)
- LoRA Rank: Detection=16, Correction=16
- 4-bit Quantization: 메모리 효율적 학습

### 규칙 체계

- Title: H01-H11 (11개 규칙)
- Body: A01-A42 (42개 규칙)
- Caption: C01-C33 (33개 규칙)
- 총 86개 규칙 (상위 프로젝트의 style_guides.json 참조)

---

## ⚠️ 주의사항

1. **API 키**: OpenAI API 키 필수 (유료)
2. **GPU**: 모델 학습/추론은 GPU 필수 (CUDA 지원)
3. **메모리**: 최소 16GB RAM 권장
4. **디스크**: 체크포인트 포함 약 1.5GB 필요
5. **형식**: 기사는 반드시 `[TITLE]`, `[BODY]`, `[CAPTION]` 태그 사용
6. **체크포인트**: 위 Google Drive 링크에서 다운로드 필요

---

## 🐛 문제 해결

### Q: "ModuleNotFoundError: No module named 'unsloth'"
A: GPU 환경에서만 설치 가능. CPU 환경에서는 데이터 증강만 가능.

### Q: "CUDA out of memory"
A: `--batch-size`를 줄이거나 `--max-seq-length`를 줄이세요.

### Q: 증강 실패율이 높음
A: `--train-ratio`를 조정하거나, `basic` 난이도만 사용하세요.

### Q: API Rate Limit 에러
A: `bulk_augmentation.py`의 `rate_limit_per_minute` 값을 줄이세요.

---

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. `setup_environment.sh` 실행 로그
2. Conda 환경 활성화 여부: `conda activate korea_times`
3. API 키 설정 여부: `echo $OPENAI_API_KEY`
4. GPU 사용 가능 여부: `nvidia-smi`

---

**참고**: 학습 파이프라인의 전체 소스코드는 별도로 제공됩니다. 
이 README는 사용법 및 구조를 설명하기 위한 문서입니다.

버전: v1.0  
최종 업데이트: 2025-11-13
