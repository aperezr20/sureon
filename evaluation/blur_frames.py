"""YOLO non-surgical region and EasyOCR text blurring from SUREON preprocessing."""
import hashlib
import json
import threading
from pathlib import Path
import cv2
import numpy as np
from .common import resolve, video_path

def blur_nonsurgical_regions(frame: np.ndarray, mask_model=None, ocr_reader=None) -> np.ndarray:
    """Run YOLO mask model on a frame and blur detected non-surgical regions."""
    results = mask_model(frame, verbose=False) if mask_model is not None else []
    for result in results:
        if result.masks is None:
            continue
        for mask in result.masks.xy:
            pts = np.array(mask, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            if w <= 0 or h <= 0:
                continue
            roi = frame[y:y+h, x:x+w]
            ksize = max(25, (min(w, h) // 3) | 1)
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    
    # ---- Pass 2: EasyOCR text regions ----
    if ocr_reader is not None:
        results = ocr_reader.readtext(frame, detail=1)
        for (bbox, text, conf) in results:
            if conf < 0.3:
                continue
            pts = np.array(bbox, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            if w <= 0 or h <= 0:
                continue
            roi = frame[y:y+h, x:x+w]
            ksize = max(25, (min(w, h) // 3) | 1)
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (ksize, ksize), 0)

    return frame

def checkpoint_identity(config):
    curation = config['curation']
    
    if not curation.get('blur_nonsurgical', True):
        return [], None
    
    if curation.get('model_dir'):
        paths = [resolve(config,curation['model_dir']) / f'best_fold{i}.pt' for i in range(5)]
    
    elif curation.get('checkpoint'):
        paths = [resolve(config,curation['checkpoint'])]
    
    elif curation.get('filename'):
        from huggingface_hub import hf_hub_download
        paths = [Path(hf_hub_download(curation['repo_id'],curation['filename'],revision=curation.get('revision','main')))]
    
    else:
        raise ValueError('Set curation.model_dir to the five-fold checkpoint directory')
    
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing blurring checkpoint: {path}')
    
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    digest = hashes[0] if len(hashes)==1 else hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    return paths,digest


def blur_fold_chain(frame, models, ocr_reader):
    """Fold0 -> fold4 on successively blurred frames; OCR after the last fold."""
    if not models:
        return blur_nonsurgical_regions(frame,None,ocr_reader)
    for index,model in enumerate(models):
        frame = blur_nonsurgical_regions(frame,model,ocr_reader if index==len(models)-1 else None)
    return frame


class FrameProcessor:
    def __init__(self, config):
        self.config = config
        self.mask_model = None
        self.ocr_reader = None
        self.lock = threading.Lock()
        curation = config['curation']
        self.preblurred = None
        if curation.get('preblurred_dir'):
            from .preblur_cache import CachedFrames
            self.preblurred = CachedFrames(config)
            self.checkpoint_hash = self.preblurred.manifest['curation']['checkpoint_sha256']
            self.versions = self.preblurred.manifest['curation']['versions']
            return
        paths,self.checkpoint_hash = checkpoint_identity(config)
        self.mask_models = []
        if paths:
            from ultralytics import YOLO
            self.mask_models = [YOLO(str(path)) for path in paths]
        self.mask_model = self.mask_models[0] if self.mask_models else None
        if curation.get('blur_ocr', True):
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'],gpu=curation.get('ocr_gpu',False))
        cache = curation.get('cache_dir')
        self.cache = resolve(config,cache) if cache else None
        if self.cache:
            self.cache.mkdir(parents=True,exist_ok=True)
        import importlib.metadata
        self.versions = {}
        for name in ['easyocr','ultralytics','opencv-python-headless','decord']:
            try:
                self.versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.versions[name] = None

    def get(self, row, sample):
        if self.preblurred is not None:
            return self.preblurred.get(row)
        
        from .video import _extract_frames
        
        settings = self.config['evaluation']
        path = video_path(self.config,row)
        stat = path.stat()
        identity = {'source':str(path.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
            'start':row['start_seconds'],'end':row['end_seconds'],
            'max_frames':settings.get('max_frames',8),'min_frames':settings.get('min_frames',4),
            'target_fps':settings.get('target_fps'),'checkpoint':self.checkpoint_hash,
            'ocr':self.ocr_reader is not None,'versions':self.versions,'protocol':'sureon-wacv-v1'}
        key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cached = self.cache / (key+'.npz') if self.cache else None
        
        # Serialize decode/curation/cache writes for thread-safe YOLO and OCR.
        with self.lock:
            if cached and cached.exists():
                with np.load(cached,allow_pickle=False) as archive:
                    return list(archive['frames']),archive['timestamps'].tolist()
            frames,timestamps = _extract_frames(sample,max_frames=settings.get('max_frames',8),
                min_frames=settings.get('min_frames',4),target_fps=settings.get('target_fps'))
            if self.mask_model is not None or self.ocr_reader is not None:
                frames = [blur_fold_chain(frame.copy(),self.mask_models,self.ocr_reader) for frame in frames]
            if cached:
                np.savez_compressed(cached,frames=np.stack(frames),timestamps=np.asarray(timestamps))
            return frames,timestamps
