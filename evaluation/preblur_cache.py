import hashlib
import json
from pathlib import Path
import cv2
from .common import resolve


def sampling(config):
    settings = config['evaluation']
    return {k:settings.get(k, default) for k,default in [('max_frames',8),('min_frames',4),('target_fps',None)]}


def legacy_key(sample, config):
    settings = sampling(config)
    raw = '|'.join([sample['data_path'],sample['video'],sample['start'],sample['end'],
                    str(settings['max_frames']),str(settings['target_fps']),str(settings['min_frames'])])
    return hashlib.sha256(raw.encode()).hexdigest()


class CachedFrames:
    def __init__(self, config):
        self.root = resolve(config, config['curation']['preblurred_dir'])
        path = self.root / 'manifest.json'
        if not path.is_file():
            raise FileNotFoundError('Preblur manifest missing. Run preblur.py before starting the model server.')
        self.manifest = json.loads(path.read_text())
        if self.manifest['sampling'] != sampling(config):
            raise ValueError('Preblur sampling differs from evaluation settings; use a matching cache')
        for key in ['blur_nonsurgical','blur_ocr']:
            if self.manifest['curation'][key] != config['curation'].get(key,True):
                raise ValueError('Preblur curation settings differ from evaluation')
        if config['curation'].get('model_dir') or config['curation'].get('checkpoint'):
            from .blur_frames import checkpoint_identity
            _,digest = checkpoint_identity(config)
            if digest != self.manifest['curation']['checkpoint_sha256']:
                raise ValueError('Preblur fold checkpoint identity differs from configuration')

    def get(self, row):
        entry = self.manifest['entries'].get(row['question_id'])
        if not entry:
            raise ValueError(f"No preblurred frames for {row['question_id']}; run preblur.py first")
        if (entry['source_video_id'],entry['start'],entry['end']) != (row['source_video_id'],row['start_timestamp'],row['end_timestamp']):
            raise ValueError('Preblur source/window mismatch')
        key = entry['key']
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid cache key')
        root = self.root / key
        metadata = json.loads((root / 'meta.json').read_text())
        frames = []
        if metadata['timestamps'] != entry['timestamps']:
            raise ValueError('Cached timestamps changed')
        if set(metadata['frame_files']) != set(entry['frame_hashes']):
            raise ValueError('Cached frame list changed')
        for filename in metadata['frame_files']:
            if Path(filename).name != filename:
                raise ValueError('Invalid cached filename')
            path = root / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['frame_hashes'][filename]:
                raise ValueError('Cached image hash mismatch')
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError('Unreadable cached frame')
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if not frames or len(frames) != len(metadata['timestamps']):
            raise ValueError('Invalid cached frame count')
        return frames,metadata['timestamps']
