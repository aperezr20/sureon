"""Configuration and annotation loading shared by the plain Python scripts."""
import json
import os
from pathlib import Path
from copy import deepcopy

SPLITS = ("train", "validation", "test")

DEFAULTS = {'dataset': {'repo_id': 'aleperez24/sureon', 'revision': 'main', 'parquet_dir': 'data'},
 'videos': {'base_dir': 'videos',
            'cookies_file': None,
            'workers': 2,
            'format': 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best'},
 'curation': {'preblurred_dir': '.cache/preblurred',
              'blur_nonsurgical': True,
              'blur_ocr': True,
              'model_dir': 'models/nonsurgical_object_detection',
              'checkpoint': None,
              'repo_id': 'visurg/LEMON_curation_models',
              'filename': None,
              'revision': 'main',
              'ocr_gpu': True,
              'cache_dir': '.cache/frames'},
 'inference': {'backend': 'openai',
               'attn_implementation': 'sdpa',
               'batch_size': 2,
               'base_url': 'http://localhost:8000/v1',
               'model': 'Qwen/Qwen3-VL-8B-Instruct',
               'api_key_env': 'OPENAI_API_KEY',
               'top_p': 1.0,
               'seed': 42,
               'max_tokens': 8192,
               'concurrency': 64,
               'timeout': 600,
               'retries': 3,
               'is_baseline_model': True},
 'evaluation': {'curated_only': True,
                'split': 'test',
                'mode': 'video',
                'max_frames': 8,
                'min_frames': 4,
                'target_fps': 'auto',
                'seed': 0,
                'output_dir': 'results'},
 'judge': {'enabled': False,
           'provider': 'anthropic',
           'model': 'claude-opus-4-8',
           'api_key_env': 'ANTHROPIC_API_KEY'}}


def read_config(filename):
    import yaml
    path = Path(filename).expanduser().resolve()
    supplied = yaml.safe_load(path.read_text()) or {}
    if not isinstance(supplied, dict):
        raise ValueError("Config must be a YAML mapping")
    config = deepcopy(DEFAULTS)
    for section, values in supplied.items():
        if section in config:
            if not isinstance(values, dict):
                raise ValueError(f"Config section {section} must be a mapping")
            config[section].update(values)
        else:
            config[section] = values
    config["_root"] = str(path.parent)
    return config


def resolve(config, value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else Path(config["_root"]) / path


def selected_splits(split):
    if split == "all":
        return SPLITS
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return (split,)


def load_rows(config, split):
    import pyarrow.parquet as pq
    root = resolve(config, config["dataset"]["parquet_dir"])
    paths = sorted(root.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} Parquet in {root}; run download_dataset.py --split {split}")
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    if any(row["split"] != split for row in rows):
        raise ValueError("Parquet split field does not match filename")
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question_id")
    return rows


def video_path(config, row):
    source = row["source_video_id"]
    if Path(source).name != source or source in {".", ".."}:
        raise ValueError("Unsafe source_video_id")
    return resolve(config, config["videos"]["base_dir"]) / source / f"{source}.mp4"


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sanitized_config(config):
    from urllib.parse import urlsplit, urlunsplit
    clean = json.loads(json.dumps({k:v for k,v in config.items() if not k.startswith("_")}))
    for group, key in [("inference", "base_url"), ("judge", "endpoint")]:
        url = clean.get(group, {}).get(key)
        if url:
            parts = urlsplit(url)
            host = parts.hostname or ""
            if parts.port:
                host += f":{parts.port}"
            clean[group][key] = urlunsplit((parts.scheme, host, parts.path, "", ""))
    return clean


def evaluation_rows(config, split):
    """Use the curated validation subset by default; no need to filter the expert test."""
    rows = load_rows(config, split)
    if split == 'validation' and config.get('evaluation', {}).get('curated_only', True):
        if rows and 'passes_blind_curation' in rows[0]:
            rows = [r for r in rows if r['passes_blind_curation'] is True]
    return rows
