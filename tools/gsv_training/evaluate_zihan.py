import argparse
import asyncio
import gc
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from generate_zihan_preview import SAMPLES


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'output' / 'zihan_20261001'
DATA = ROOT / 'data' / 'zihan_cn_20261001'
MODELS = ROOT / 'models' / 'zihan'
EMOTIONS = {'neutral': '平淡', 'happy': '活泼', 'gentle': '温柔', 'teasing': '撒娇甜',
            'shy': '害羞', 'surprised': '惊讶', 'angry': '生气', 'sad': '悲伤'}


def checkpoint_epoch(path, kind):
    pattern = r'-e([0-9]+)' if kind == 'gpt' else r'_e([0-9]+)_'
    match = re.search(pattern, path.name)
    return int(match.group(1)) if match else -1


def select_references():
    records = json.loads((DATA / 'teacher_quality.json').read_text(encoding='utf-8'))
    destination = ROOT / 'reference_audio' / 'zihan'
    destination.mkdir(exist_ok=True)
    references = {}
    for emotion, label in EMOTIONS.items():
        candidates = [item for item in records if item['passed'] and item['emotion'] == emotion and 3.2 <= item['duration'] <= 9.5]
        if not candidates:
            raise RuntimeError(f'No usable reference for emotion {emotion}')
        selected = min(candidates, key=lambda item: (item['cer'], abs(item['duration'] - 5.5)))
        source = DATA / 'cn_teacher' / selected['file']
        target = destination / f'{emotion}.wav'
        shutil.copy2(source, target)
        references[label] = {'audio': target.relative_to(ROOT).as_posix(), 'text': selected['text'],
                             'language': 'zh', 'teacher_cer': selected['cer']}
    (destination / 'references.json').write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding='utf-8')
    return references


async def evaluate():
    from gsv_tts import TTS
    references = select_references()
    originals = json.loads((DATA / 'refs.json').read_text(encoding='utf-8'))
    selected_teacher = json.loads((DATA / 'selection.json').read_text(encoding='utf-8'))['selected']
    original = originals['B' if selected_teacher == 'B_reference' else 'A']['preview_path']
    gpt = {checkpoint_epoch(path, 'gpt'): path for path in (MODELS / 'gpt').glob('*.ckpt')}
    sovits = {checkpoint_epoch(path, 'sovits'): path for path in (MODELS / 'sovits').glob('*.pth')}
    combinations = [(5, 10), (10, 10), (15, 10), (10, 20), (15, 20)]
    destination = OUTPUT / 'evaluation'
    destination.mkdir(exist_ok=True)
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8')) if (destination / 'manifest.json').exists() else []
    completed = {item['id'] for item in manifest if (destination / item['file']).is_file()}
    models = {}
    for gpt_epoch, sovits_epoch in combinations:
        if gpt_epoch not in gpt or sovits_epoch not in sovits:
            continue
        variant = f's1e{gpt_epoch}_s2e{sovits_epoch}'
        models[variant] = {'gpt': gpt[gpt_epoch].relative_to(ROOT).as_posix(),
                           'sovits': sovits[sovits_epoch].relative_to(ROOT).as_posix()}
        engine = TTS(models_dir=str(ROOT / 'gsv_models'), use_bert=True)
        engine.load_gpt_model(str(gpt[gpt_epoch]))
        engine.load_sovits_model(str(sovits[sovits_epoch]))
        for reference in references.values():
            audio = str(ROOT / reference['audio'])
            engine.cache_spk_audio(audio)
            engine.cache_prompt_audio(prompt_audio_paths=audio, prompt_audio_texts=reference['text'], prompt_language='zh')
        for index, (emotion, text) in enumerate(SAMPLES, 1):
            item_id = f'{variant}_{index:02d}'
            if item_id in completed:
                continue
            reference = references[EMOTIONS.get(emotion, '平淡')]
            audio = str(ROOT / reference['audio'])
            torch.manual_seed(1234 + index)
            started = time.perf_counter()
            first_chunk = None
            chunks = []
            async for clip in engine.infer_stream_async(spk_audio_path=audio, prompt_audio_path=audio,
                prompt_audio_text=reference['text'], prompt_language='zh', text=text, text_language='zh',
                stream_mode='sentence', boost_first_chunk=False, debug=False):
                if first_chunk is None:
                    first_chunk = time.perf_counter() - started
                chunks.append(clip.audio_data)
            if not chunks:
                raise RuntimeError(f'No inference output for {item_id}')
            waveform = np.concatenate(chunks)
            if not np.isfinite(waveform).all():
                raise RuntimeError(f'Non-finite inference audio for {item_id}')
            duration = len(waveform) / 32000
            target = destination / variant / f'{index:02d}_{emotion}.wav'
            target.parent.mkdir(exist_ok=True)
            sf.write(target, waveform, 32000, subtype='PCM_16')
            inference_seconds = time.perf_counter() - started
            teacher_similarity = engine.verify_speaker(audio, str(target))
            original_similarity = engine.verify_speaker(original, str(target))
            if teacher_similarity is None or original_similarity is None or not np.isfinite([teacher_similarity, original_similarity]).all():
                raise RuntimeError(f'Invalid speaker verification score for {item_id}')
            manifest.append({'id': item_id, 'variant': variant, 'emotion': emotion, 'text': text,
                'file': target.relative_to(destination).as_posix(), 'duration': duration, 'sample_rate': 32000,
                'rms': float(np.sqrt(np.mean(waveform ** 2))), 'peak': float(np.max(np.abs(waveform))),
                'first_chunk_seconds': first_chunk, 'rtf': inference_seconds / max(duration, 0.001),
                'speaker_similarity_to_teacher': teacher_similarity,
                'speaker_similarity_to_jp': original_similarity})
            temporary = destination / 'manifest.tmp'
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(destination / 'manifest.json')
            print(f'{item_id}: {duration:.2f}s, first chunk {first_chunk:.3f}s', flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    if len(models) < 2:
        raise RuntimeError('Fewer than two checkpoint combinations available for comparison')
    (destination / 'models.json').write_text(json.dumps(models, indent=2), encoding='utf-8')


def publish():
    report = json.loads((OUTPUT / 'evaluation_quality.json').read_text(encoding='utf-8'))
    models = json.loads((OUTPUT / 'evaluation/models.json').read_text(encoding='utf-8'))
    scores = {}
    for variant in models:
        records = [item for item in report if item['variant'] == variant]
        if len(records) != len(SAMPLES):
            continue
        scores[variant] = {'mean_cer': sum(item['cer'] for item in records) / len(records),
                           'passed': sum(item['passed'] for item in records),
                           'mean_jp_speaker_similarity': sum(item['speaker_similarity_to_jp'] for item in records) / len(records)}
    qualified = [variant for variant in scores if scores[variant]['passed'] >= 8 and scores[variant]['mean_cer'] <= 0.15]
    if not qualified:
        raise RuntimeError(f'No checkpoint passed fixed-set validation: {scores}')
    minimum = min(scores[variant]['mean_cer'] for variant in qualified)
    close = [variant for variant in qualified if scores[variant]['mean_cer'] <= minimum + 0.025]
    selected = max(close, key=lambda variant: (scores[variant]['mean_jp_speaker_similarity'],
                   *[int(value) for value in re.findall('[0-9]+', variant)[::-1]]))
    references = json.loads((ROOT / 'reference_audio/zihan/references.json').read_text(encoding='utf-8'))
    voice = {'id': 'zihan', 'label': '新海天中文', 'text_language': 'zh', 'references': references,
             **models[selected], 'selected_checkpoint': selected, 'scores': scores,
             'subjective_audition': 'pending',
             'automatic_gate': 'Chinese ASR + acoustic sanity; SV similarity tie-break is not human likeness acceptance'}
    temporary = MODELS / 'voice.tmp'
    temporary.write_text(json.dumps(voice, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(MODELS / 'voice.json')
    (OUTPUT / 'final_selection.json').write_text(json.dumps(voice, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(voice, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        publish()
        subprocess.run([sys.executable, '-u', '-B', str(ROOT / 'tools/verify_zihan_service.py')],
                       cwd=ROOT, check=True)
    else:
        asyncio.run(evaluate())


if __name__ == '__main__':
    main()
