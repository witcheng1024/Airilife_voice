import argparse
import ctypes
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from prepare_zihan import wsl_path
from zihan_corpus import build_corpus


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'output' / 'zihan_20261001'
DATA = ROOT / 'data' / 'zihan_cn_20261001'
WSL_PYTHON = '/home/witcheng/miniforge3/envs/tts/bin/python'


def status(phase, state='running', **details):
    record = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'phase': phase, 'state': state, **details}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / 'status.tmp'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(OUTPUT / 'status.json')
    with (OUTPUT / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + chr(10))
    print(json.dumps(record, ensure_ascii=False), flush=True)


def phase(label, command, timeout=21600):
    status(label)
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
                       CUDA_LAUNCH_BLOCKING='0', HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='120')
    with (OUTPUT / f'{label}.log').open('a', encoding='utf-8') as log:
        subprocess.run([str(value) for value in command], cwd=ROOT, env=environment,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=timeout, check=True)
    status(label, 'completed')


def wsl(script, *arguments):
    libraries = '/home/witcheng/miniforge3/envs/tts/lib/python3.11/site-packages/nvidia/cudnn/lib:/home/witcheng/miniforge3/envs/tts/lib/python3.11/site-packages/nvidia/cublas/lib'
    return ['wsl.exe', '-d', 'Ubuntu24.04', '--exec', 'env', 'HF_HUB_DISABLE_XET=1',
            'HF_HUB_DOWNLOAD_TIMEOUT=120', f'LD_LIBRARY_PATH={libraries}', WSL_PYTHON, '-u', '-B', wsl_path(script), *arguments]


def assemble_training(variant):
    quality = json.loads((DATA / 'teacher_quality.json').read_text(encoding='utf-8'))
    passed = [item for item in quality if item['passed'] and item['variant'] == variant]
    if len(passed) < 600:
        raise RuntimeError(f'Only {len(passed)} Chinese samples passed ASR; refusing low-quality training')
    anchors = json.loads((DATA / 'jp_anchors.json').read_text(encoding='utf-8'))[:min(300, round(len(passed) * 0.375))]
    audio_dir = DATA / 'train_audio'
    audio_dir.mkdir(exist_ok=True)
    rows = []
    for item in passed:
        source = DATA / 'cn_teacher' / item['file']
        destination = audio_dir / source.name
        shutil.copy2(source, destination)
        rows.append('|'.join([wsl_path(destination), 'zihan', 'zh', item['text']]))
    for item in anchors:
        source = Path(item['path'])
        destination = audio_dir / source.name
        shutil.copy2(source, destination)
        rows.append('|'.join([wsl_path(destination), 'zihan', 'ja', item['text']]))
    (DATA / 'train.list').write_text(chr(10).join(rows) + chr(10), encoding='utf-8')
    summary = {'variant': variant, 'chinese': len(passed), 'japanese': len(anchors), 'total': len(rows),
               'asr_max_cer': 0.12, 'teacher_samples': len(quality), 'subjective_audition': 'pending'}
    (DATA / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-pid', type=int)
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--teacher-python', type=Path, default=Path('E:/PROJECT/Airilife_voice/.venv/Scripts/python.exe'))
    parser.add_argument('--gsv-python', type=Path, default=Path('D:/miniforge3/envs/gsv/python.exe'))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'worker.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    (OUTPUT / 'worker.pid').write_text(str(os.getpid()), encoding='utf-8')
    try:
        if args.finalize_only:
            phase('publish', [args.gsv_python, '-u', '-B', ROOT / 'tools/evaluate_zihan.py', '--publish'])
            status('finished', 'completed', voice='zihan', subjective_audition='pending')
            return
        if args.preview_pid:
            status('waiting_existing_preview', pid=args.preview_pid)
            handle = ctypes.windll.kernel32.OpenProcess(0x100000, False, args.preview_pid)
            if handle:
                result = ctypes.windll.kernel32.WaitForSingleObject(handle, 1800000)
                ctypes.windll.kernel32.CloseHandle(handle)
                if result != 0:
                    raise RuntimeError('Existing preview worker did not finish within 30 minutes')
        if not (DATA / 'refs.json').is_file():
            phase('prepare', [args.teacher_python, '-B', ROOT / 'tools/prepare_zihan.py'])
        build_corpus(DATA / 'corpus.json')
        phase('preview_generate', [args.teacher_python, '-u', '-B', ROOT / 'tools/generate_zihan_preview.py'])
        phase('preview_validate', wsl(ROOT / 'tools/validate_zihan_audio.py', '--samples',
                                     wsl_path(OUTPUT / 'preview'), '--output', wsl_path(OUTPUT / 'preview_quality.json')), timeout=3600)
        quality = json.loads((OUTPUT / 'preview_quality.json').read_text(encoding='utf-8'))
        scores = {}
        for variant in {item['variant'] for item in quality}:
            records = [item for item in quality if item['variant'] == variant]
            scores[variant] = {'mean_cer': sum(item['cer'] for item in records) / len(records),
                               'passed': sum(item['passed'] for item in records)}
        best = min(scores, key=lambda variant: (scores[variant]['mean_cer'], -scores[variant]['passed']))
        variant = 'A_natural' if scores['A_natural']['mean_cer'] <= scores[best]['mean_cer'] + 0.05 else best
        if scores[variant]['passed'] < 7:
            raise RuntimeError(f'No sufficiently intelligible preview variant: {scores}')
        (DATA / 'selection.json').write_text(json.dumps({'selected': variant, 'scores': scores,
            'selection_basis': 'ASR intelligibility, conservative natural style; subjective voice likeness unverified'}, indent=2), encoding='utf-8')
        phase('teacher_generate', [args.teacher_python, '-u', '-B', ROOT / 'tools/generate_zihan_preview.py',
                                  '--corpus', DATA / 'corpus.json', '--variant', variant, '--output', DATA / 'cn_teacher'])
        phase('teacher_validate', wsl(ROOT / 'tools/validate_zihan_audio.py', '--samples',
                                     wsl_path(DATA / 'cn_teacher'), '--output', wsl_path(DATA / 'teacher_quality.json'), '--max-cer', '0.12'))
        summary = assemble_training(variant)
        status('dataset_ready', 'completed', **summary)
        phase('training', wsl(ROOT / 'tools/train_zihan_wsl.py', '--root', wsl_path(ROOT)), timeout=43200)
        phase('evaluation', [args.gsv_python, '-u', '-B', ROOT / 'tools/evaluate_zihan.py'], timeout=7200)
        phase('evaluation_validate', wsl(ROOT / 'tools/validate_zihan_audio.py', '--samples',
                                        wsl_path(OUTPUT / 'evaluation'), '--output', wsl_path(OUTPUT / 'evaluation_quality.json')), timeout=3600)
        phase('publish', [args.gsv_python, '-u', '-B', ROOT / 'tools/evaluate_zihan.py', '--publish'])
        status('finished', 'completed', voice='zihan', subjective_audition='pending')
    except Exception as error:
        status('failed', 'failed', error=str(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
