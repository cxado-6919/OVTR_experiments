# Modern / Blackwell 설정

이 저장소는 RTX 5090, RTX PRO 6000 Blackwell 같은 NVIDIA Blackwell GPU에 적합한 최신 Linux stack에서 동작하도록 포팅되어 있습니다.

`main`과 비교한 전체 `QAT-Implement` 브랜치 요약은 [QAT_IMPLEMENT_BRANCH_CHANGES.md](QAT_IMPLEMENT_BRANCH_CHANGES.md)를 참고하십시오.

## 대상 Stack

- Ubuntu 22.04 또는 24.04
- Python 3.11
- PyTorch 2.7.1
- torchvision 0.22.1
- torchaudio 2.7.1
- CUDA-enabled wheel channel: `cu128`
- CUDA 12.8을 지원하는 toolkit / driver stack으로 local build한 custom CUDA extension

PyTorch 2.7은 Blackwell 지원과 CUDA 12.8 wheel을 명시적으로 발표한 첫 PyTorch release입니다. 기존 `torch 1.10.1 + cu111` stack은 Blackwell hardware에서 현실적으로 사용하기 어렵기 때문에, 이 포팅은 `torch==2.7.1`을 대상으로 합니다.

출처:

- [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch previous versions install matrix](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/index.html)

## 환경 생성

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
```

선택 helper:

```bash
bash scripts/setup_blackwell.sh
```

## Custom CUDA Ops Build

중복되어 있는 두 op tree는 이제 package-local in-place extension입니다. 두 extension을 모두 build해야 합니다.

```bash
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

cd ovtr/models/ops
python setup.py build_ext --inplace

cd ../../../ovtr_det_bs2_pretrain/models/ops
python setup.py build_ext --inplace
```

참고:

- `TORCH_CUDA_ARCH_LIST`를 직접 설정한 경우 해당 값을 그대로 사용합니다.
- 직접 설정하지 않으면 setup script가 visible GPU에서 architecture list를 추론하려고 시도하고, 실패하면 Blackwell(`12.0+PTX`)을 포함한 최신 list로 fallback합니다.
- Extension은 더 이상 단일 global `MultiScaleDeformableAttention` module로 install되지 않습니다. 각 tree가 자기 local `_C` extension을 in-place로 build합니다.

## 선택 CLIP Dependency

Core training / eval은 이제 precomputed embedding file만 있으면 동작합니다. CLIP embedding을 다시 생성하는 작업은 선택 사항입니다.

CLIP text 또는 image embedding을 다시 생성하려면 다음 중 하나를 install하십시오.

```bash
python -m pip install open-clip-torch
```

또는:

```bash
python -m pip install git+https://github.com/openai/CLIP.git
```

## Smoke Test

가장 작은 repo-local smoke test는 전체 dataset 없이 실행할 수 있습니다.

```bash
cd ovtr
python tools/smoke_test.py

cd ../ovtr_det_bs2_pretrain
python tools/smoke_test.py
```

이 test는 다음 항목을 확인합니다.

- Python import path 상태
- config parsing
- `MultiScaleDeformableAttention` import
- attention module의 최소 forward pass
- CUDA를 사용할 수 있을 때 선택적 CUDA-path attention 실행

## QAT 메모리 참고 사항

Full-partition QAT는 원본 OVTR fine-tuning path보다 GPU memory를 상당히 더 요구할 수 있습니다. 특히 `exp_a1_to_b`는 선택된 floating-point model weight와 learned quantization parameter를 함께 학습하므로 원래 QAT semantics를 유지합니다. 따라서 주요 memory pressure는 quantization parameter 자체보다 activation 저장량과 optimizer state에서 발생합니다.

`--quant_mode qat`를 사용할 때, experimental batched QAT를 요청하지 않았다면 `main.py`가 transformer checkpointing과 frame-wise checkpointing을 자동으로 활성화합니다. Transformer checkpointing flag는 이제 encoder layer뿐 아니라 decoder layer에도 적용됩니다.

- encoder layer는 기존 `use_transformer_ckpt` path를 통해 checkpoint됩니다.
- decoder layer body는 training 중 `torch.utils.checkpoint(..., use_reentrant=False)`로 checkpoint됩니다.
- reference point update, bbox head, class logit, aux-output collection은 checkpoint boundary 밖에 유지됩니다.
- eval과 calibration은 direct forward path를 유지하므로 observer/calibration side effect가 checkpoint replay로 다시 계산되지 않습니다.

권장 low-memory QAT baseline:

```bash
cd ovtr

MODEL_VARIANT=lite \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
BATCH_SIZE=1 \
QAT_ALLOW_BATCH=0 \
QUANT_PIPELINE=legacy \
CALIB_SAMPLES=32 \
./tools/ovtr_quant_full_model.sh \
  --no_aux_loss \
  --max_len 100 \
  --quant_mse_bins 0 \
  --quant_mse_candidates 1
```

참고:

- Memory를 줄이려면 `QAT_ALLOW_BATCH=0`을 유지하십시오. Batched QAT는 QAT-only checkpoint forcing path를 비활성화합니다.
- `--max_len`을 낮추면 각 frame에서 사용하는 sampled text/image class embedding 수가 줄어듭니다. 하지만 backbone feature memory나 object query 수는 줄어들지 않습니다.
- `--no_aux_loss`는 decoder auxiliary-output memory를 줄입니다. OOM triage에는 유용한 경우가 많지만, training loss configuration을 변경합니다.
- Decoder checkpointing 이후에도 QAT가 계속 OOM이면 다음 non-semantic memory reduction 후보는 더 강한 frame-wise checkpointing 또는 sharded optimizer state입니다.

## 예시 Entry Point

Training:

```bash
cd ovtr
bash tools/ovtr_multi_frame_lite_train.sh
```

Evaluation:

```bash
cd ovtr
bash tools/ovtr_ovmot_eval_lite_val.sh
```

Detection pretraining:

```bash
cd ovtr_det_bs2_pretrain
bash tools/ovtr_detection_pretrain.sh
```

## Troubleshooting

- CUDA extension build에서 CUDA를 사용할 수 없다고 나오면 다음을 확인하십시오.
  - PyTorch environment가 CPU-only가 아니라 CUDA wheel build인지 확인하십시오.
  - 최신 CUDA toolkit의 `nvcc`가 install되어 있고 path에서 보이는지 확인하십시오.
  - NVIDIA driver가 CUDA 12.8 / Blackwell에 충분히 최신인지 확인하십시오.
- Extension build가 architecture 문제로 실패하면:
  - Blackwell에서는 `TORCH_CUDA_ARCH_LIST=12.0+PTX`를 명시적으로 설정하십시오.
- Extension build를 건너뛰면:
  - 저장소는 pure-PyTorch deformable attention fallback을 사용할 수 있지만, 실행 속도는 더 느립니다.
