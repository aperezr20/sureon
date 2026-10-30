# Evaluation options

The [README quickstart](../README.md#quickstart) evaluates the 299 accuracy questions with and without video. 


## Sequence summarization judging

The 41 test sequence summarization samples are graded by Claude Opus 4.8. Set `judge.enabled: true` in [config.yaml](../config.yaml), then run without `--accuracy-only`:

```bash
export ANTHROPIC_API_KEY="YOUR_KEY"
python scripts/preblur.py --split test
python scripts/evaluate.py --mode both
```

The judge compares predictions with reference summaries using the rubric in [evaluation/judge.py](../evaluation/judge.py). With judging disabled, summaries remain unscored. 

## vLLM Server inference

Install the [vLLM environment](setup.md#vllm-optional), then start a server in another terminal:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 65536 --seed 0 \
  --gpu-memory-utilization 0.90 --tensor-parallel-size 1 --trust-remote-code
```

Change the inference section in `config.yaml`:

```yaml
inference:
  backend: openai
  model: Qwen/Qwen3-VL-8B-Instruct
  base_url: http://localhost:8000/v1
```

Run the README evaluation command. 

## Outputs and reproducibility

Each run saves predictions, `metrics.json`, the resolved config, and provenance under `results/`. 

Evaluation uses native-FPS sampling clamped to 4–8 frames (`target_fps: auto`), five-fold masking followed by OCR blurring, and greedy inference. 