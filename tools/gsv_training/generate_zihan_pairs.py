import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gen_multilingual import load_cv, synth


ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = 'You are a helpful assistant. 请用中文说这句话。<|endofprompt|>'
TRANSLATIONS = {
    'sr0871': '要不要拿我家那个混蛋老哥，试着克服一下男性恐惧症呀！',
    'sr0292': '我本来还想着，能不能把爸爸的书房给我呢。结果哥哥居然搬出去了，真是没想到啊。',
    'sr0566': '真是的，简直就是公主嘛。还真有点羡慕，我也想被那样捧着呀。',
    'sr0027': '哎，是谁呀？你说想介绍给我的人。话说，到底怎么回事？哥哥突然这么认真，怪吓人的。',
    'sr0956': '就是啊，都怪老哥……不对，都怪翔大人啦。',
    'sr0812': '好嘞！那明天约吃午饭这事，就算答应了，对吧？',
    'sr0797': '对方实在太能说了，我也就只能在旁边嗯嗯啊啊地应着啦。',
    'sr0461': '那个，能不能别说得那么感慨呀？你确定？我会哭哦？我真会哭哦？',
}


def prepare_pairs(output):
    anchor_path = ROOT / 'data' / 'zihan_cn_20261001' / 'jp_anchors.json'
    anchors = {item['id']: item for item in json.loads(anchor_path.read_text(encoding='utf-8'))}
    source_root = (ROOT / 'data' / 'nimi_sora_hf' / '9nine' / 'nimi_sora' / 'reference_audio').resolve()
    pairs = []
    for index, (sample_id, translation) in enumerate(TRANSLATIONS.items(), 1):
        anchor = anchors[sample_id]
        reference = Path(anchor['path']).resolve()
        reference.relative_to(source_root)
        if not reference.is_file() or reference.stem != sample_id:
            raise ValueError(f'Invalid source reference: {reference}')
        prefix = f'{index:02d}_{sample_id}'
        original = output / f'{prefix}_JP原声.wav'
        chinese = output / f'{prefix}_CN中文.wav'
        source_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        shutil.copy2(reference, original)
        if hashlib.sha256(original.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError(f'Source copy mismatch: {sample_id}')
        text = f'{sample_id}\n日语原文：{anchor["text"]}\n中文译文：{translation}\n参考音频：{reference}\n'
        (output / f'{prefix}_文本对照.txt').write_text(text, encoding='utf-8-sig')
        pairs.append({
            'id': sample_id, 'jp_text': anchor['text'], 'zh_text': translation,
            'reference': str(reference), 'source_sha256': source_hash,
            'jp_audio': str(original), 'cn_audio': str(chinese), 'status': 'pending',
        })
    if len({item['source_sha256'] for item in pairs}) != 8:
        raise ValueError('Expected eight distinct Japanese references')
    return pairs


def write_manifest(output, pairs):
    manifest = {
        'model': 'Fun-CosyVoice3-0.5B', 'method': 'inference_instruct2',
        'reused_pipeline': 'tools/gen_multilingual.py', 'instruction': INSTRUCTION,
        'emotion_instructions': False, 'subjective_status': 'awaiting_user_audition',
        'completed': sum(item['status'] == 'generated' for item in pairs), 'pairs': pairs,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('E:/Downloads/新海天_日中逐句配对试听_20261002'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob('*_CN中文.wav')):
        raise FileExistsError('Chinese previews already exist; use a new output directory')
    pairs = prepare_pairs(args.output)
    write_manifest(args.output, pairs)
    torch.set_num_threads(4)
    model = load_cv()
    for index, pair in enumerate(pairs, 1):
        started = time.time()
        seed = 20261002 + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f'[{index}/8] {pair["id"]}: {pair["zh_text"]}', flush=True)
        try:
            audio = synth(model, 'zh', pair['zh_text'], reference=pair['reference'], instruction=INSTRUCTION)
            waveform = audio.detach().cpu().numpy().reshape(-1)
            duration = len(waveform) / model.sample_rate
            peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
            if not np.all(np.isfinite(waveform)) or not 0.5 <= duration <= 40 or not 0.01 <= peak < 1.0:
                raise RuntimeError(f'Invalid audio: duration={duration:.3f}, peak={peak:.3f}')
            sf.write(pair['cn_audio'], waveform, model.sample_rate, subtype='PCM_16')
            pair.update(status='generated', duration_s=round(duration, 3), sample_rate=model.sample_rate,
                        peak=round(peak, 5), seed=seed, seconds=round(time.time() - started, 2),
                        cn_sha256=hashlib.sha256(Path(pair['cn_audio']).read_bytes()).hexdigest())
            write_manifest(args.output, pairs)
            print(f'[{index}/8] OK {duration:.2f}s -> {pair["cn_audio"]}', flush=True)
        except Exception as error:
            pair.update(status='failed', error=str(error))
            write_manifest(args.output, pairs)
            raise
    note = ('本目录是新生成的8组逐句日中配对试听，不是之前那批情绪模板数据。\n'
            '同编号的JP原声.wav是日语原音；CN中文.wav是对应台词的中文译文合成。\n'
            '每条中文都用同编号的日语原音作为参考，8条使用8个不同参考。\n'
            '复用流萤多语言流程：CosyVoice3 / inference_instruct2，仅指定中文，没有情绪或年龄指令。\n'
            '文本对照.txt保留日语原文和中文译文；manifest.json记录来源和生成参数。\n'
            '这是教师音频试听，未启动批量生成或GPT-SoVITS训练。请先验收音色，再决定是否扩量。\n')
    (args.output / '试听说明.txt').write_text(note, encoding='utf-8-sig')
    print(f'COMPLETE 8/8: {args.output}', flush=True)


if __name__ == '__main__':
    main()
