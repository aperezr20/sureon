# Setup

Use Python 3.11 and an NVIDIA GPU. Install `ffmpeg`/`ffprobe` for video downloads. Commands below use CUDA 12.8; choose another PyTorch build if your system requires it.

## Transformers

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl
python -m pip install -r requirements-transformers.txt
python -m pip check
```


## vLLM (optional)

Create a separate environment:

```bash
python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm==0.11.0 transformers==4.57.6
python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl
python -m pip install -r requirements-vllm.txt
python -m pip check
```

FlashAttention is downloaded to match Python 3.11 / PyTorch 2.8 / CUDA 12.8 stack. Other combinations require a [matching FlashAttention wheel](https://github.com/mjun0812/flash-attention-prebuild-wheels). 
