import argparse
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def wsl_path(path):
    value = path.as_posix()
    if len(value) > 2 and value[1] == ':':
        return '/mnt/' + value[0].lower() + value[2:]
    return value


def inventory(source):
    records = []
    rejected = []
    transcript = source / 'transcript.txt'
    forbidden = re.compile('エッチ|セックス|おっぱい|射精|精液|オナニー|勃起|乳首|ちんぽ|喘ぎ')
    for line_number, line in enumerate(transcript.read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = line.split('|', 3)
        if len(fields) != 4:
            rejected.append({'line': line_number, 'reason': 'format'})
            continue
        filename = fields[0].replace(chr(92), '/').split('/')[-1]
        audio = source / 'reference_audio' / filename
        text = fields[3].strip()
        if not audio.is_file():
            rejected.append({'file': filename, 'reason': 'missing'})
            continue
        info = sf.info(audio)
        if not 1.2 <= info.duration <= 12 or len(text) < 6 or forbidden.search(text):
            rejected.append({'file': filename, 'reason': 'duration_or_text'})
            continue
        waveform, sample_rate = sf.read(audio, dtype='float32', always_2d=True)
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        peak = float(np.max(np.abs(waveform)))
        clipping = float(np.mean(np.abs(waveform) >= 0.999))
        if rms < 0.003 or clipping > 0.01:
            rejected.append({'file': filename, 'reason': 'silence_or_clipping'})
            continue
        records.append({'id': audio.stem, 'path': audio.as_posix(), 'wsl_path': wsl_path(audio),
                        'text': text, 'language': fields[2], 'duration': info.duration,
                        'sample_rate': sample_rate, 'channels': info.channels, 'peak': peak,
                        'rms_db': 20 * math.log10(rms), 'clipping': clipping})
    return records, rejected


def prepare(source, output):
    records, rejected = inventory(source)
    references = [item for item in records if 4 <= item['duration'] <= 9 and len(item['text']) >= 15]
    references.sort(key=lambda item: (abs(item['duration'] - 6), item['id']))
    if len(references) < 2:
        raise RuntimeError('Not enough usable references; download must finish first')
    neutral = next((item for item in references if not re.search('[!！?？]', item['text'])), references[0])
    lively = next((item for item in references if item['id'] != neutral['id'] and re.search('[!！?？]', item['text'])), next(item for item in references if item['id'] != neutral['id']))
    ref_dir = output / 'refs'
    ref_dir.mkdir(exist_ok=True)
    selected = {}
    for label, record in [('A', neutral), ('B', lively)]:
        record_id = record['id']
        destination = ref_dir / f'ref_{label}_{record_id}.wav'
        shutil.copy2(record['path'], destination)
        selected[label] = dict(record, preview_path=destination.as_posix())
    anchor_pool = [item for item in records if item['id'] not in {neutral['id'], lively['id']} and 2 <= item['duration'] <= 10]
    random.Random(20261001).shuffle(anchor_pool)
    anchors = []
    seen_texts = set()
    for record in anchor_pool:
        if record['text'] not in seen_texts:
            anchors.append(record)
            seen_texts.add(record['text'])
        if len(anchors) == 300:
            break
    if len(anchors) < 300:
        raise RuntimeError(f'Only {len(anchors)} Japanese anchors available')
    (output / 'refs.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'jp_anchors.json').write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'source': source.as_posix(), 'transcript_lines': len((source / 'transcript.txt').read_text(encoding='utf-8-sig').splitlines()),
               'downloaded_wavs': len(list((source / 'reference_audio').glob('*.wav'))), 'eligible': len(records),
               'eligible_minutes': sum(item['duration'] for item in records) / 60, 'anchors': len(anchors),
               'rejected': rejected, 'reference_selection': 'Text/duration heuristics; human audition required'}
    (output / 'inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'rejected'}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/nimi_sora_hf/9nine/nimi_sora'))
    parser.add_argument('--output', type=Path, default=Path('data/zihan_cn_20261001'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
