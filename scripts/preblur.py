"""Preprocess and verify frames before loading the evaluation model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import cv2
from evaluation.common import read_config, load_rows, evaluation_rows, resolve, json_write
from evaluation.video import legacy_sample
from evaluation.blur_frames import FrameProcessor
from evaluation.preblur_cache import CachedFrames, legacy_key, sampling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--split', choices=['test','validation'], default='test')
    parser.add_argument('--accuracy-only', action='store_true')
    parser.add_argument('--check', action='store_true', help='Verify existing frames without loading models')
    args = parser.parse_args()
    config = read_config(args.config)
    rows = evaluation_rows(config,args.split)
    if args.accuracy_only:
        rows = [r for r in rows if r['question_type'] != 'sequence_summarization']
    destination = config['curation'].get('preblurred_dir')
    if not destination:
        raise ValueError('Set curation.preblurred_dir')
    root = resolve(config,destination)
    manifest_path = root / 'manifest.json'
    manifest = None
    cached = None
    if manifest_path.exists():
        cached = CachedFrames(config)
        manifest = cached.manifest
    if args.check:
        if cached is None:
            raise FileNotFoundError('No preblur manifest')
        for row in rows:
            cached.get(row)
        print(f'Verified {len(rows)} preblurred questions; no models loaded.')
        return
    complete = set(manifest['entries']) if manifest else set()
    for row in rows:
        if row['question_id'] in complete:
            cached.get(row)
    pending = [r for r in rows if r['question_id'] not in complete]
    if not pending:
        print(f'All {len(rows)} preblurred questions verified; no models loaded.')
        return
    generation = copy.deepcopy(config)
    generation['curation']['preblurred_dir'] = None
    processor = FrameProcessor(generation)
    if manifest is None:
        manifest = {'sampling':sampling(config),'curation':{
            'checkpoint_sha256':processor.checkpoint_hash,'versions':processor.versions,
            'blur_nonsurgical':config['curation'].get('blur_nonsurgical',True),
            'blur_ocr':config['curation'].get('blur_ocr',True)},'entries':{}}
    elif processor.checkpoint_hash != manifest['curation']['checkpoint_sha256'] or processor.versions != manifest['curation']['versions']:
        raise ValueError('Do not mix checkpoint/library versions in one preblur cache; choose a fresh directory')
    root.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(pending,1):
        sample = legacy_sample(row,config)
        key = legacy_key(sample,config)
        shared = next((entry for entry in manifest['entries'].values() if entry['key']==key), None)
        if shared is not None:
            manifest['entries'][row['question_id']] = dict(shared)
            json_write(manifest_path,manifest)
            print(f'Reused shared clip {index}/{len(pending)}')
            continue
        frames,timestamps = processor.get(row,sample)
        directory = root / key
        directory.mkdir(exist_ok=True)
        filenames = []
        hashes = {}
        for number,frame in enumerate(frames):
            name = f'frame_{number:03d}.jpg'
            path = directory / name
            if path.exists():
                raise FileExistsError(f'Untracked cache file exists: {path}; choose a fresh preblur directory')
            if not cv2.imwrite(str(path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]):
                raise OSError('Could not write preblur frame')
            filenames.append(name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        json_write(directory/'meta.json',{'frame_files':filenames,'timestamps':timestamps})
        manifest['entries'][row['question_id']] = {'key':key,'source_video_id':row['source_video_id'],
            'start':row['start_timestamp'],'end':row['end_timestamp'],'timestamps':timestamps,'frame_hashes':hashes}
        json_write(manifest_path,manifest)
        print(f'Preblurred {index}/{len(pending)}')
    checker = CachedFrames(config)
    for row in rows:
        checker.get(row)
    print(f'Preblur complete: {len(rows)} questions. This process exits before model loading.')


if __name__ == '__main__':
    main()
