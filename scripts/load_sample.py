"""Load one curated training sample using the SUREON training input format."""
import argparse
import random

from evaluation.common import read_config, resolve, video_path


def training_text(row):
    question, answer = row["question"], row["answer_text"]
    if row["task_format"] != "multiple_choice":
        return question, answer

    def unique(values):
        seen = {answer.strip().lower()}
        result = []
        for value in values:
            key = (value or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result

    pools = {source["source"]: source["distractors"] for source in row["distractor_pools"]}
    pool = unique(
        value for source in ("Llama", "Qwen35-35B", "MedGemma")
        for value in (pools.get(source) or [])
    )
    if len(pool) < 6:
        pool = unique(row["baseline_distractors"])
    distractors = random.Random().sample(pool, 6) if len(pool) >= 6 else pool
    answer = answer[:1].upper() + answer[1:]
    choices = [value[:1].upper() + value[1:] for value in distractors] + [answer]
    random.Random().shuffle(choices)
    question += "\n\nSelect from these options:\n" + "\n".join("- " + value for value in choices)
    return question, answer


def format_seconds(seconds):
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{seconds % 60:06.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", type=int, default=0, help="Index within the curated training subset")
    args = parser.parse_args()

    import numpy as np
    from datasets import load_dataset
    from decord import VideoReader, cpu
    from transformers import AutoProcessor

    config = read_config(args.config)
    shards = sorted(resolve(config, config["dataset"]["parquet_dir"]).glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError("Run python scripts/download_dataset.py --split train first")
    train = load_dataset("parquet", data_files=[str(path) for path in shards], split="train")
    train = train.filter(lambda row: row["passes_blind_curation"] is True)
    row = train[args.index]
    path = video_path(config, row)
    if not path.is_file():
        raise FileNotFoundError(f"Download the training videos first: {path}")

    # Same target_fps=auto, min_frames=4, max_frames=8 as the training run.
    video = VideoReader(str(path), ctx=cpu(0))
    fps = video.get_avg_fps()
    start, end = row["start_seconds"], row["end_seconds"]
    duration = end - start
    if duration <= 0:
        raise ValueError("The sample must have a positive temporal window")
    count = max(4, min(8, int(duration * fps)))
    indices = np.linspace(int(start * fps), int(end * fps), count, dtype=int)
    indices = np.clip(indices, 0, len(video) - 1)
    frames = video.get_batch(indices).asnumpy()
    timestamps = [start + i / (count - 1) * duration for i in range(count)]

    question, answer = training_text(row)
    content = [{"type": "text", "text": (
        "You are viewing frames from a surgical video "
        f"(time range: {row['start_timestamp']} to {row['end_timestamp']})."
    )}]
    for frame, timestamp in zip(frames, timestamps):
        content.extend([
            {"type": "image", "image": frame},
            {"type": "text", "text": f"(timestamp: {format_seconds(timestamp)})"},
        ])
    content[-1]["text"] += "\n\n" + question.replace("<video>", "").strip()
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

    processor = AutoProcessor.from_pretrained(
        config["inference"]["model"], min_pixels=784, max_pixels=50176,
    )
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )
    print("Question:", row["question_id"])
    print(question)
    print("Answer:", answer)
    print("Frames:", len(frames))
    print("Inputs:", {key: tuple(value.shape) for key, value in inputs.items()})


if __name__ == "__main__":
    main()
