import argparse
import hashlib
import json
import os
import random
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from evaluation.common import read_config, resolve, load_rows, evaluation_rows, video_path, json_write, sanitized_config
from evaluation.scoring import score_answer, metrics, _extract_answer

EXPECTED = {"test": 340, "validation": 2081}


def preflight(config, rows, modes, allow_partial=False):
    settings = config["evaluation"]
    minimum, maximum = settings.get("min_frames", 4), settings.get("max_frames", 8)
    
    if not (1 <= minimum <= maximum):
        raise ValueError("Require 1 <= evaluation.min_frames <= evaluation.max_frames")
    fps = settings.get("target_fps")
    
    if fps is not None and fps != "auto" and (not isinstance(fps, (float, int)) or fps <= 0):
        raise ValueError("evaluation.target_fps must be null, auto, or a positive number")
    backend = config["inference"].get("backend", "openai")
    
    if backend not in {"openai", "transformers"}:
        raise ValueError("inference.backend must be openai or transformers")
    
    if not config["inference"].get("model"):
        raise ValueError("Set inference.model")
    
    if backend == "openai" and not config["inference"].get("base_url"):
        raise ValueError("Set inference.base_url for the openai backend")
    
    if int(config["inference"].get("concurrency", 256)) < 1 or int(config["inference"].get("retries", 3)) < 1:
        raise ValueError("Inference concurrency and retries must be positive")
    problems = {}
    for row in rows:
        if row["end_seconds"] <= row["start_seconds"]:
            problems[row["question_id"]] = "invalid_source_window"
        if "video" in modes and not config["curation"].get("preblurred_dir") and not video_path(config, row).is_file():
            problems[row["question_id"]] = "missing_video"
    
    if "video" in modes:
        curation = config["curation"]
        if curation.get("preblurred_dir"):
            from evaluation.preblur_cache import CachedFrames
            cached = CachedFrames(config)
            for row in rows:
                cached.get(row)
        
        elif curation.get("blur_nonsurgical", True):
            from evaluation.blur_frames import checkpoint_identity
            checkpoint_identity(config)
            checkpoint = curation.get("checkpoint")
            if checkpoint and not resolve(config, checkpoint).is_file():
                raise FileNotFoundError("Configured curation.checkpoint does not exist")
            if not checkpoint and not curation.get("filename") and not curation.get("model_dir"):
                raise ValueError("Set curation.checkpoint or an explicit curation.filename in the HF model repository")
    
    if problems and not allow_partial:
        raise ValueError(f"Preflight failed: {Counter(problems.values())}. Download the required full lectures or explicitly use --allow-partial.")
    
    judge = config["judge"]
    if judge.get("enabled"):
        from evaluation.judge import validate_judge_config
        validate_judge_config(judge)
    return problems


def infer(client, settings, messages):
    attempts = int(settings.get("retries", 3))
    for attempt in range(attempts):
        try:
            reply = client.chat.completions.create(
                model=settings["model"], messages=messages,
                max_tokens=int(settings.get("max_tokens", 8192)), temperature=0.0,
                top_p=settings.get("top_p", 1.0), seed=settings.get("seed", 42),
            )
            prediction = reply.choices[0].message.content
            if not isinstance(prediction, str) or not prediction.strip():
                raise ValueError("Empty model response")
            return prediction
        except Exception:
            if attempt + 1 == attempts:
                raise RuntimeError("Inference request failed after configured retries") from None
            time.sleep(min(2 ** attempt, 8))


def run_mode(config, rows, mode, problems, client, processor, judge, model=None, model_processor=None):
    from evaluation.video import build_messages
    print(f"{mode}: preparing {len(rows)} prompts...", flush=True)
    prepared = {}
    def prepare_one(row):
        issue = problems.get(row['question_id'])
        if issue and not (mode == 'blind' and issue == 'missing_video'):
            return row['question_id'], None, None
        try:
            return row['question_id'], build_messages(row, config, mode, processor), None
        except Exception as exc:
            return row['question_id'], None, type(exc).__name__
    
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (identifier, messages, error) in enumerate(pool.map(prepare_one, rows), 1):
            prepared[identifier] = (messages, error)
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(f"{mode}: prepared {index}/{len(rows)} prompts", flush=True)
    
    print(f"{mode}: running inference...", flush=True)
    started = time.monotonic()

    def one(row, prediction=None):
        result = {
            **{k: row[k] for k in ("question_id", "clip_id", "source_video_id", "question_type", "question", "choices", "answer_text")},
            "mode": mode, "prediction": None, "correct": None,
            "judge_verdict": None, "assessment": None, "status": "ok",
        }
        
        issue = problems.get(row["question_id"])
        # Missing video does not invalidate a paired blind pass.
        if issue and not (mode == "blind" and issue == "missing_video"):
            result["status"] = issue
            return result
        try:
            messages, build_error = prepared[row['question_id']]
            if build_error:
                result['status'] = 'failed'
                result['error_type'] = build_error
                return result
            result["prediction"] = prediction if prediction is not None else infer(client, config["inference"], messages)
            result["correct"] = score_answer(row, result["prediction"])
            if row["question_type"] == "sequence_summarization":
                result.update(judge.evaluate(row, _extract_answer(result["prediction"])))
                
        except Exception as exc:
            # Do not put remote response bodies, URLs, or credentials into reports.
            result["status"] = "failed"
            result["error_type"] = type(exc).__name__
        return result
    done = {}
    
    def record(result, index):
        done[result["question_id"]] = result
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"{mode}: {index}/{len(rows)} answers ({time.monotonic() - started:.0f}s)", flush=True)

    if model is not None:
        from evaluation.transformers_backend import generate_batch
        batch_size = int(config["inference"].get("batch_size", 2))
        
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            active = [row for row in batch if prepared[row["question_id"]][0] is not None]
            predictions = generate_batch(
                model, model_processor, [prepared[row["question_id"]][0] for row in active],
                int(config["inference"].get("max_tokens", 8192)),
            ) if active else []
            
            if len(predictions) != len(active) or any(not prediction.strip() for prediction in predictions):
                raise ValueError("Expected one nonempty prediction per question")
            answers = {row["question_id"]: prediction for row, prediction in zip(active, predictions)}
            
            for index, row in enumerate(batch, start + 1):
                record(one(row, answers.get(row["question_id"])), index)
            
            print(f"{mode}: batch complete, {start + len(batch)}/{len(rows)} answers "
                  f"({time.monotonic() - started:.0f}s)", flush=True)
    
    else:
        with ThreadPoolExecutor(max_workers=int(config["inference"].get("concurrency", 256))) as pool:
            jobs = [pool.submit(one, row) for row in rows]
            for index, job in enumerate(as_completed(jobs), 1):
                record(job.result(), index)
    
    return [done[row["question_id"]] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["test", "validation"], help="Override evaluation.split")
    parser.add_argument("--mode", choices=["video", "blind", "both"], help="Override evaluation.mode")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without model/API calls")
    parser.add_argument("--allow-partial", action="store_true", help="Report incomplete runs explicitly")
    parser.add_argument("--accuracy-only", action="store_true", help="Evaluate the nine accuracy categories, excluding summaries")
    parser.add_argument("--limit", type=int, help="Debug subset")
    args = parser.parse_args()
    config = read_config(args.config)
    backend = config["inference"].get("backend", "openai")
    
    if backend == "transformers":
        batch_size = int(config["inference"].get("batch_size", 2))
        if batch_size < 1:
            parser.error("inference.batch_size must be positive")
        config["inference"]["batch_size"] = batch_size
        print(f"Transformers: greedy generation, batch size {batch_size}", flush=True)
    
    settings = config["evaluation"]
    split = args.split or settings.get("split", "test")
    mode = args.mode or settings.get("mode", "video")
    
    if split not in EXPECTED or mode not in {"video", "blind", "both"}:
        parser.error("Evaluation supports test/validation and video/blind/both")
    
    settings.update(split=split, mode=mode)
    rows = evaluation_rows(config, split)
    total = len(rows)
    expected = 7039 if split == "validation" and not settings.get("curated_only", True) else EXPECTED[split]
    
    if total != expected and not args.allow_partial:
        raise ValueError(f"Expected {expected} {split} questions, got {total}")
    accuracy_only = args.accuracy_only or settings.get("accuracy_only", False)
    
    if accuracy_only:
        rows = [r for r in rows if r["question_type"] != "sequence_summarization"]
    eligible_total = len(rows)
    
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    
    modes = ["video", "blind"] if mode == "both" else [mode]
    problems = preflight(config, rows, modes, args.allow_partial)
    print(f"{split}: {len(rows)}/{total} questions; modes={modes}; preflight problems={len(problems)}")
    if args.dry_run:
        print("Preflight passed. No inference, judge requests, or model downloads were performed.")
        return
    
    random.seed(settings.get("seed", 0))
    from evaluation.judge import SummaryJudge
    from evaluation.blur_frames import FrameProcessor
    
    judge = SummaryJudge(config["judge"])
    processor = FrameProcessor(config) if "video" in modes else None
    api_key = os.environ.get(config["inference"].get("api_key_env", "")) or "EMPTY"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = resolve(config, settings["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "config.json", sanitized_config(config))
    
    provenance = {
        "inference_backend": backend,
        "split": split,
        "dataset_questions": total,
        "selected_questions": len(rows),
        "expected_official_questions": EXPECTED[split],
        "expected_selected_population": expected,
        "accuracy_only": accuracy_only,
        "prompt_scheduling": "prebuild_all_then_infer",
        "annotation_shards": [],
        "curation_checkpoint_sha256": getattr(processor, "checkpoint_hash", None),
        "preprocessing_versions": getattr(processor, "versions", {}),
        "judge_prompt_sha256": None,
    }
    
    from evaluation.judge import SEQSUM_JUDGE_PROMPT
    provenance["judge_prompt_sha256"] = hashlib.sha256(SEQSUM_JUDGE_PROMPT.encode()).hexdigest()
    provenance["annotation_shards"] = [
        {"filename": s.name, "sha256": hashlib.sha256(s.read_bytes()).hexdigest()}
        for s in sorted(resolve(config, config["dataset"]["parquet_dir"]).glob(f"{split}-*.parquet"))
    ]
    
    json_write(output / "provenance.json", provenance)
    summaries = {}
    failed = False
    client = model = model_processor = None
    
    if backend == "transformers":
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.manual_seed(int(config["inference"].get("seed", 42)))
        model_processor = AutoProcessor.from_pretrained(config["inference"]["model"])
        model_processor.tokenizer.padding_side = "left"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["inference"]["model"], dtype="auto", device_map="auto",
            attn_implementation=config["inference"].get("attn_implementation", "sdpa"),
        ).eval()
        provenance["attention_implementations"] = {"model": model.config._attn_implementation}
        for name in ("text_config", "vision_config"):
            subconfig = getattr(model.config, name, None)
            if subconfig is not None:
                provenance["attention_implementations"][name] = getattr(subconfig, "_attn_implementation", None)
        json_write(output / "provenance.json", provenance)
        print("Model devices:", getattr(model, "hf_device_map", model.device), flush=True)
    
    else:
        from openai import OpenAI
        client = OpenAI(base_url=config["inference"]["base_url"], api_key=api_key,
                        timeout=config["inference"].get("timeout", 600), max_retries=0)
    
    for current in modes:
        results = run_mode(config, rows, current, problems, client, processor, judge, model, model_processor)
        with (output / (current + "_predictions.jsonl")).open("x") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        summary = metrics(results)
        summary["complete"] = total == expected and len(rows) == eligible_total and summary["failed_questions"] == 0
        summary["standard_blurring"] = current == "blind" or (config["curation"].get("blur_nonsurgical", True) and config["curation"].get("blur_ocr", True))
        summary["full_standard_accuracy_run"] = summary["complete"] and summary["standard_blurring"]
        failed |= summary["failed_questions"] > 0 or summary["judge_failures"] > 0
        summaries[current] = summary
    
    if client is not None:
        client.close()
    
    if set(summaries) == {"video", "blind"}:
        visual = summaries["video"]["accuracy_questions"]["accuracy"]
        blind = summaries["blind"]["accuracy_questions"]["accuracy"]
        summaries["video_minus_blind"] = visual - blind if visual is not None and blind is not None else None
    
    json_write(output / "metrics.json", summaries)
    
    for current in modes:
        score = summaries[current]["accuracy_questions"]
        accuracy = score["accuracy"]
        formatted = f"{accuracy:.2%}" if accuracy is not None else "unavailable"
        print(f"{current}: {score['correct']}/{score['expected']} ({formatted}); "
              f"failures={summaries[current]['failed_questions']}")
    
    print("Saved:", output)
    if failed and not args.allow_partial:
        raise SystemExit("Run has explicit inference/judge failures; inspect reports before reporting results.")


if __name__ == "__main__":
    main()
