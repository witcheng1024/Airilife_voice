import argparse
import json
import re
from pathlib import Path


def normalize(text):
    return re.sub('[^a-z0-9一-龥]', '', text.lower())


def character_error(expected, actual):
    expected, actual = normalize(expected), normalize(actual)
    previous = list(range(len(actual) + 1))
    for row, character in enumerate(expected, 1):
        current = [row]
        for column, observed in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (character != observed)))
        previous = current
    return previous[-1] / max(1, len(expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cer', type=float)
    parser.add_argument('--backend', choices=['paraformer', 'sensevoice', 'whisper'], default='paraformer')
    args = parser.parse_args()
    if args.max_cer is None:
        args.max_cer = 0.12 if args.samples.name == 'cn_teacher' else 0.25
    manifest = json.loads((args.samples / 'manifest.json').read_text(encoding='utf-8'))
    if args.backend in {'sensevoice', 'paraformer'}:
        from funasr import AutoModel
        model_name = 'iic/SenseVoiceSmall' if args.backend == 'sensevoice' else 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        model = AutoModel(model=model_name, device='cuda', hub='ms', trust_remote_code=False,
                          disable_update=True, disable_pbar=True)
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel('small', device='cuda', compute_type='int8_float16',
                             download_root='/home/witcheng/.cache/zihan_whisper', cpu_threads=4)
    report = json.loads(args.output.read_text(encoding='utf-8')) if args.output.exists() else []
    for item in report:
        item['passed'] = item['cer'] <= args.max_cer and 1.2 <= item['duration'] <= 12 and item['rms'] >= 0.005
        item['max_cer'] = args.max_cer
    completed = {item['id'] for item in report}
    for item in manifest:
        if item['id'] in completed:
            continue
        if args.backend in {'sensevoice', 'paraformer'}:
            results = model.generate(input=str(args.samples / item['file']), cache={}, language='zh',
                                     use_itn=True, batch_size_s=30)
            recognized = re.sub('<[|].*?[|]>', '', results[0]['text'])
        else:
            segments, info = model.transcribe(str(args.samples / item['file']), language='zh', beam_size=3, vad_filter=True, condition_on_previous_text=False)
            recognized = ''.join(segment.text for segment in segments)
        error = character_error(item['text'], recognized)
        passed = error <= args.max_cer and 1.2 <= item['duration'] <= 12 and item['rms'] >= 0.005
        result = dict(item, recognized=recognized, cer=error, passed=passed,
                      max_cer=args.max_cer, semantic_check=args.backend)
        report.append(result)
        temporary = args.output.with_suffix('.tmp')
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(args.output)
        item_id = item['id']
        print(f'{item_id}: CER={error:.3f} pass={passed} recognized={recognized}', flush=True)
    passed_count = sum(item['passed'] for item in report)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    print(f'ASR passed {passed_count}/{len(report)}', flush=True)


if __name__ == '__main__':
    main()
