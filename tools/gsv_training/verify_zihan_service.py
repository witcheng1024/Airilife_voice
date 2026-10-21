import argparse
import concurrent.futures
import json
import math
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'output' / 'zihan_20261001'
TEXT = '早上好呀，今天也要记得好好吃饭。累了就休息一下，我会陪着你的。'


def get_json(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return json.load(response)


def compatible(base):
    try:
        voices = get_json(base, '/api/voices')['voices']
        return {'firefly', 'zihan'} <= {item['id'] for item in voices if item['ready']}
    except (OSError, ValueError, KeyError):
        return False


def connect():
    for port in range(9880, 9886):
        base = f'http://127.0.0.1:{port}'
        if compatible(base):
            return base, None
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                continue
        with (OUTPUT / 'tts_service.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', '-B', str(ROOT / 'tools/tts_server.py'),
                '--host', '127.0.0.1', '--port', str(port)], cwd=ROOT,
                env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        (OUTPUT / 'tts_service.pid').write_text(str(process.pid), encoding='utf-8')
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('TTS service exited; inspect tts_service.log')
            if compatible(base):
                return base, process
            time.sleep(2)
        process.terminate()
        process.wait(timeout=15)
        raise RuntimeError('TTS service readiness timeout')
    raise RuntimeError('No free TTS port; existing services were left untouched')


def synthesize(base, speaker, mode, name):
    payload = {'text': TEXT, 'emotion': '平淡', 'speaker': speaker, 'stream_mode': mode}
    request = urllib.request.Request(base + '/api/tts', data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        if response.headers.get('X-TTS-Speaker') != speaker:
            raise RuntimeError(f'Speaker header mismatch for {speaker}')
        if not response.headers.get('Content-Type', '').startswith('audio/L16'):
            raise RuntimeError('Unexpected audio format')
        first = response.read(2048)
        first_chunk = time.perf_counter() - started
        pcm = first + response.read()
    elapsed = time.perf_counter() - started
    if len(pcm) % 2 or len(pcm) < 32000 * 2:
        raise RuntimeError(f'Empty, truncated or malformed PCM for {speaker}')
    samples = struct.unpack(f'<{len(pcm) // 2}h', pcm)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768
    if rms < 0.005:
        raise RuntimeError(f'Near-silent PCM for {speaker}')
    target = OUTPUT / 'service_checks' / f'{name}.wav'
    target.parent.mkdir(exist_ok=True)
    with wave.open(str(target), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(32000)
        audio.writeframes(pcm)
    return {'speaker': speaker, 'mode': mode, 'file': str(target), 'text': TEXT,
            'duration': len(samples) / 32000, 'rms': rms, 'first_pcm_seconds': first_chunk,
            'total_seconds': elapsed, 'rtf_with_switch': elapsed / (len(samples) / 32000)}


def run_checks(base):
    results = []
    for speaker, mode, name in [('firefly', 'sentence', '01_firefly'),
                                 ('zihan', 'sentence', '02_zihan'),
                                 ('firefly', 'sentence', '03_firefly_return'),
                                 ('zihan', 'token', '04_zihan_token')]:
        results.append(synthesize(base, speaker, mode, name))
        if get_json(base, '/health')['speaker'] != speaker:
            raise RuntimeError('Active speaker did not change')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(synthesize, base, speaker, 'sentence', f'05_concurrent_{speaker}')
                   for speaker in ['zihan', 'firefly']]
        results.extend(future.result(timeout=900) for future in futures)
    invalid = urllib.request.Request(base + '/api/tts', data=json.dumps(
        {'text': TEXT, 'speaker': 'not_a_voice'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(invalid, timeout=10).close()
    except urllib.error.HTTPError as error:
        if error.code != 400:
            raise
    else:
        raise RuntimeError('Invalid speaker was accepted')
    cancel = urllib.request.Request(base + '/api/tts', data=json.dumps(
        {'text': TEXT * 8, 'speaker': 'firefly', 'stream_mode': 'token'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(cancel, timeout=600) as response:
        if not response.read(2048):
            raise RuntimeError('Cancellation probe returned no PCM')
    results.append(synthesize(base, 'zihan', 'sentence', '06_after_cancel'))
    results.append(synthesize(base, 'firefly', 'sentence', '07_default_restored'))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    process = None
    base = args.base
    try:
        if not base:
            base, process = connect()
        connection = {'base': base, 'demo': base + '/demo',
                      'owned_pid': process.pid if process else None, 'state': 'checking'}
        (OUTPUT / 'service_connection.json').write_text(json.dumps(connection, indent=2), encoding='utf-8')
        results = run_checks(base)
        connection.update(state='passed', checks=results, final_health=get_json(base, '/health'),
                          subjective_audition='pending', completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        (OUTPUT / 'service_connection.json').write_text(json.dumps(connection, ensure_ascii=False, indent=2), encoding='utf-8')
        selection = json.loads((OUTPUT / 'final_selection.json').read_text(encoding='utf-8'))
        dataset = json.loads((ROOT / 'data/zihan_cn_20261001/dataset_summary.json').read_text(encoding='utf-8'))
        score = selection['scores'][selection['selected_checkpoint']]
        report = [
            '# 新海天中文：自动训练与接入结果',
            '',
            '自动模型筛选和 HTTP 检查已通过；尚未由用户人耳验收音色、自然度和情绪。',
            '',
            f"- 训练数据：中文 {dataset['chinese']} 条，日文 anchor {dataset['japanese']} 条。",
            f"- 选中组合：`{selection['selected_checkpoint']}`。",
            f"- 固定测试集：{score['passed']}/10 通过，平均 CER {score['mean_cer']:.4f}。",
            f"- 原声声纹 cosine：{score['mean_jp_speaker_similarity']:.4f}（仅作辅助，不代表人耳认可）。",
            f"- GPT：`{selection['gpt']}`；SoVITS：`{selection['sovits']}`。",
            f"- 试听页面：{base}/demo；音色列表：{base}/api/voices。",
            '- 请求用 `speaker=zihan` 或 `speaker=firefly`；默认保持流萤。',
            '- 已验证双向切换、token/sentence、并发排队、无效 ID 和断连后继续合成。',
            '- 本目录 `service_checks/` 是实际 HTTP 试听 WAV；`evaluation/` 是五组 checkpoint 的对比样本。',
            '- 切换音色包含冷加载时间，不能把冷切换延迟当成稳态首字延迟。',
        ]
        (OUTPUT / 'RESULT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        print(json.dumps(connection, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        (OUTPUT / 'service_failure.json').write_text(json.dumps({'base': base, 'error': str(error)}, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
