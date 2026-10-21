import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = [
    ('neutral', '今天的安排已经整理好了，我们先把最重要的事情做好吧。'),
    ('happy', '你终于回来啦！今天有没有遇到什么有趣的事情呀？'),
    ('gentle', '先休息一下吧，喝口水再继续，我会在这里陪着你的。'),
    ('teasing', '哎呀，你该不会又忘记保存了吧？下次可要记得哦。'),
    ('shy', '那个，其实我一直想说，能和你一起聊天，我很开心。'),
    ('surprised', '真的吗？你居然已经全部做完了！这也太厉害了吧！'),
    ('angry', '不可以再熬夜了！答应过我的事情，你可不能反悔。'),
    ('sad', '今天好像有一点不顺利，不过没关系，我们慢慢来吧。'),
    ('numbers', '现在是晚上九点半，还有三项任务，明天早上八点提醒你。'),
    ('long', '如果觉得今天的任务太多，就先选一件最容易完成的事情，不用着急，我会陪你一点一点把它们做完。'),
]
VARIANTS = {
    'A_natural': ('A', 'You are a helpful assistant. 请用自然、清晰的普通话说话。<|endofprompt|>'),
    'B_reference': ('B', 'You are a helpful assistant. 请用自然、清晰的普通话说话。<|endofprompt|>'),
    'C_lively': ('A', 'You are a helpful assistant. 请用自然、清晰的普通话说话，语气轻快灵动，保留参考声音的特点，不要夸张，不要夹嗓。<|endofprompt|>'),
}
EMOTION_INSTRUCTIONS = {
    'neutral': '语气自然平和。', 'happy': '语气开心轻快。', 'gentle': '语气温柔，带一点关心。',
    'teasing': '语气俏皮，轻轻打趣。', 'shy': '语气有一点害羞，但不要夸张。',
    'surprised': '语气自然惊讶。', 'angry': '语气认真，略带生气但不要喊叫。',
    'sad': '语气轻柔，带一点难过但不要哭泣。',
}


def load_model(cosy_root, model_dir):
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(cache / 'numba'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    sys.path.insert(0, str(cosy_root))
    sys.path.insert(0, str(cosy_root / 'third_party' / 'Matcha-TTS'))
    from cosyvoice.cli.cosyvoice import AutoModel
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in teacher environment')
    torch.cuda.init()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    print('Loading existing CosyVoice3 teacher...', flush=True)
    return AutoModel(model_dir=str(model_dir), fp16=True)


def synthesize(model, text, instruction, reference, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    chunks = [item['tts_speech'].detach().cpu() for item in model.inference_instruct2(
        tts_text=text, instruct_text=instruction, prompt_wav=str(reference), stream=False)]
    if not chunks:
        raise RuntimeError('Teacher returned empty speech')
    waveform = torch.cat(chunks, dim=1).squeeze().numpy()
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.95:
        waveform = waveform * (0.95 / peak)
    return waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'zihan_cn_20261001')
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'zihan_20261001' / 'preview')
    parser.add_argument('--cosy-root', type=Path, default=Path('E:/PROJECT/Airilife_voice/CosyVoice'))
    parser.add_argument('--model-dir', type=Path, default=Path('E:/PROJECT/Airilife_voice/pretrained_models/Fun-CosyVoice3-0.5B'))
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--variant', choices=list(VARIANTS), default='A_natural')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    references = json.loads((args.data / 'refs.json').read_text(encoding='utf-8'))
    variants = [args.variant] if args.corpus else list(VARIANTS)
    samples = json.loads(args.corpus.read_text(encoding='utf-8')) if args.corpus else [
        {'id': f'preview_{index:02d}', 'emotion': emotion, 'text': text}
        for index, (emotion, text) in enumerate(SAMPLES, 1)]
    model = load_model(args.cosy_root, args.model_dir)
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    completed = {item['id'] for item in manifest if (args.output / item['file']).is_file()}
    for variant in variants:
        reference_label, instruction = VARIANTS[variant]
        reference = Path(references[reference_label]['preview_path'])
        for index, sample in enumerate(samples):
            sample_id = sample['id']
            item_id = f'{variant}_{sample_id}'
            if item_id in completed:
                continue
            text = sample['text']
            emotion = sample['emotion']
            sample_instruction = instruction.replace('<|endofprompt|>', '') + EMOTION_INSTRUCTIONS.get(emotion, '') + '<|endofprompt|>'
            destination = args.output / variant / f'{sample_id}_{emotion}.wav'
            destination.parent.mkdir(exist_ok=True)
            started = time.time()
            waveform = synthesize(model, text, sample_instruction, reference, 1234 + index)
            duration = len(waveform) / model.sample_rate
            if not np.isfinite(waveform).all() or duration < 1.0 or duration > 30:
                raise RuntimeError(f'Invalid teacher audio: {item_id}, duration={duration}')
            sf.write(destination, waveform, model.sample_rate, subtype='PCM_16')
            manifest.append({'id': item_id, 'variant': variant, 'emotion': emotion, 'text': text,
                             'file': destination.relative_to(args.output).as_posix(), 'duration': duration,
                             'sample_rate': model.sample_rate, 'reference': reference.as_posix(),
                             'instruction': sample_instruction, 'seed': 1234 + index,
                             'peak': float(np.max(np.abs(waveform))),
                             'rms': float(np.sqrt(np.mean(waveform ** 2))), 'semantic_check': 'pending'})
            temporary = manifest_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(manifest_path)
            print(f'{item_id}: {duration:.2f}s generated in {time.time() - started:.1f}s', flush=True)
    for variant in variants:
        playlist = ['#EXTM3U'] + [item['file'] for item in manifest if item['variant'] == variant]
        (args.output / f'{variant}.m3u8').write_text(chr(10).join(playlist) + chr(10), encoding='utf-8')
    print(f'Completed {len(manifest)} teacher samples in {args.output}', flush=True)


if __name__ == '__main__':
    main()
