import base64
import os
import re
import random
import numpy as np
from .common import video_path

def _parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2].replace(",", "."))
    return h * 3600 + m * 60 + s

def _format_seconds(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def _frame_to_base64(frame_np) -> str:
    import cv2
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")

def _extract_frames(sample: dict, num_distractors: int = 3, max_frames: int = 8,
                    target_fps=None, min_frames: int = 4) -> tuple:
    """Extract numpy frames and timestamps from a sample's video file.

    target_fps:
      None  — adaptive duration-based sampling (default, preserves original behaviour)
      'auto' — sample at the video's native fps, clamped to [min_frames, max_frames]
      float  — sample at exactly target_fps frames/sec
    """
    from decord import VideoReader, cpu
    full_path = os.path.join(sample["data_path"], sample["video"])
    start_sec = _parse_timestamp(sample["start"])
    end_sec = _parse_timestamp(sample["end"])

    vr = VideoReader(full_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    duration = end_sec - start_sec

    if target_fps is None:
        if duration <= 5:
            num_frames = min(int(duration * 2), max_frames)
        elif duration <= 30:
            num_frames = min(int(duration), max_frames)
        else:
            num_frames = min(int(duration * 0.5), max_frames)
        num_frames = max(num_frames, min_frames)
    elif target_fps == 'auto':
        if duration <= 0:
            raise ValueError(
                f"_extract_frames: negative or zero duration ({duration:.3f}s) "
                f"for video {full_path!r} with start={sample['start']!r} end={sample['end']!r}. "
                "Check that start/end timestamps are not swapped in the dataset."
            )
        num_frames = min(max_frames, int(duration * fps))
        num_frames = max(num_frames, min_frames)
    else:
        num_frames = max(1, int(duration * target_fps))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frame_indices = np.clip(frame_indices, 0, len(vr) - 1)
    frames = vr.get_batch(frame_indices).asnumpy()
    timestamps = [
        start_sec + (i / (num_frames - 1)) * duration if num_frames > 1 else start_sec
        for i in range(num_frames)
    ]
    return [frames[i] for i in range(len(frames))], timestamps


_BASELINE_SYSTEM_PROMPT = (
    "You are an expert surgical AI assistant.\n"
    "Select the correct answer from the video frames provided.\n"

    "For multiple choice questions, copy the option text exactly as written.\n"
    "If asked for the order of events, respond with ONLY the letters in order, comma-separated.\n\n"
    "If question is open ended, provide a brief response based on the video without bullet points or markdown"
)



def legacy_sample(row, config):
    question = row['question']
    
    if row['choices']:
        question += '\n\nSelect from these options:\n' + '\n'.join('- ' + choice for choice in row['choices'])
    
    path = video_path(config, row)
    
    return {'id':row['question_id'],'video':path.name,'data_path':str(path.parent),
            'start':row['start_timestamp'],'end':row['end_timestamp'],'qa_type':row['question_type'],
            'conversations':[{'from':'human','value':'<video>\n'+question},
                             {'from':'assistant','value':row['answer_text']}]}


def build_messages(row, config, mode, frames_processor=None):
    sample = legacy_sample(row, config)
    question = re.sub(r'^<(?:video|image)>\s*', '', sample['conversations'][0]['value']).strip()
    
    if config['inference'].get('is_baseline_model', False):
        question = _BASELINE_SYSTEM_PROMPT + question
    
    if mode == 'blind':
        if row['question_type'] == 'temporal_ordering':
            question += '\n\nRespond with ONLY the letters in order, comma-separated.'
        return [{'role':'user','content':[{'type':'text','text':question}]}]
    
    if frames_processor is None:
        raise ValueError('Video prompts require a frame processor')
    
    frames, timestamps = frames_processor.get(row, sample)
    content = [{'type':'text','text':f"You are viewing frames from a surgical video (time range: {sample['start']} to {sample['end']}).\n\n"}]
    
    for frame, timestamp in zip(frames,timestamps):
        content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+_frame_to_base64(frame)}})
        content.append({'type':'text','text':f'(timestamp: {_format_seconds(timestamp)})\n'})
    
    content.append({'type':'text','text':'\n'+question})
    
    return [{'role':'user','content':content}]
