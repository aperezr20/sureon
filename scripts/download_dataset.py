"""Download just the selected annotation splits from Hugging Face."""
import argparse
from evaluation.common import read_config, resolve, selected_splits


def download(config, split):
    from huggingface_hub import HfApi, hf_hub_download
    import shutil
    import hashlib
    dataset = config["dataset"]
    if not dataset.get("repo_id"):
        raise ValueError("Set dataset.repo_id to the published Hugging Face dataset ID")
    api = HfApi()
    revision = api.dataset_info(dataset["repo_id"], revision=dataset.get("revision", "main")).sha
    files = api.list_repo_files(dataset["repo_id"], repo_type="dataset", revision=revision)
    destination = resolve(config, dataset["parquet_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repo_id": dataset["repo_id"], "revision": revision, "files": []}
    for name in selected_splits(split):
        candidates = [f for f in files if f.endswith(".parquet") and f.rsplit("/", 1)[-1].startswith(name + "-")]
        if not candidates:
            raise ValueError(f"No {name}-*.parquet files in the dataset repository")
        basenames = [f.rsplit("/", 1)[-1] for f in candidates]
        if len(set(basenames)) != len(basenames):
            raise ValueError("Ambiguous shard basenames")
        for filename in sorted(candidates):
            cached = hf_hub_download(dataset["repo_id"], filename, repo_type="dataset", revision=revision)
            target = destination / filename.rsplit("/", 1)[-1]
            digest = hashlib.sha256(open(cached, "rb").read()).hexdigest()
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise FileExistsError(f"Existing shard differs: {target}; use a fresh parquet_dir")
            else:
                shutil.copyfile(cached, target)
            manifest["files"].append({"filename": filename, "sha256": digest})
            print(target)
    from evaluation.common import json_write
    json_write(destination / f"download_{split}.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    args = parser.parse_args()
    download(read_config(args.config), args.split)


if __name__ == "__main__":
    main()
