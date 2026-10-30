# Load the dataset

```bash
pip install datasets
```

```python
from datasets import load_dataset

sureon = load_dataset("aleperez24/sureon")
train = sureon["train"].filter(lambda row: row["passes_blind_curation"] is True)
validation = sureon["validation"].filter(lambda row: row["passes_blind_curation"] is True)
test = sureon["test"]

print(len(train), len(validation), len(test))  # 76061, 2081, 340
print(test[0])
```

Train and validation include full pre-curation pools; the filters select the paper's curated subsets. Training and validation are pseudo-labeled; test questions are surgeon-verified. Videos are downloaded separately using the [README instructions](../README.md#quickstart).

## Load a training sample with Qwen3-VL

After [environment setup](setup.md), download the training annotations and videos:

```bash
python scripts/download_dataset.py --split train
python scripts/download_videos.py --split train
python scripts/load_sample.py
```

[load_sample.py](../scripts/load_sample.py) uses [config.yaml](../config.yaml) and the training input format: 4–8 uniformly sampled frames, timestamped image messages, and six randomly drawn multi-model distractors (with baseline fallback). Each run draws fresh choices; use `--index 1` to select another sample. 