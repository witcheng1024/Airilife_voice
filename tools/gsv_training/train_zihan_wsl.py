import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml


GS = Path('/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS')
VERSION = 'v2ProPlus'


def run(command, log, environment, cwd=GS):
    print('Running', command, flush=True)
    with log.open('a', encoding='utf-8') as output:
        subprocess.run(command, cwd=cwd, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True)


def configure(root, experiment, epochs_s2, epochs_s1, batch_size):
    pretrained = root / 'gsv_models'
    required = {'generator': pretrained / 's2Gv2ProPlus.pth',
                'discriminator': pretrained / 'v2Pro' / 's2Dv2ProPlus.pth',
                'gpt': pretrained / 's1v3.ckpt',
                'bert': pretrained / 'chinese-roberta-wwm-ext-large' / 'pytorch_model.bin',
                'hubert': pretrained / 'chinese-hubert-base' / 'pytorch_model.bin',
                'speaker': pretrained / 'sv' / 'pretrained_eres2netv2w24s4ep4.ckpt',
                'g2pw': GS / 'GPT_SoVITS/text/G2PWModel/g2pW.onnx'}
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size < 1024:
            raise RuntimeError(f'Missing pretrained {label}: {path}')
    pretrained_g = torch.load(required['generator'], map_location='cpu', weights_only=True)
    embedding = pretrained_g['weight']['enc_p.text_embedding.weight']
    if embedding.shape[0] != 732:
        raise RuntimeError(f'Generator phoneme vocabulary mismatch: {embedding.shape}')
    destination = root / 'models' / 'zihan'
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((GS / 'GPT_SoVITS/configs/s2v2ProPlus.json').read_text(encoding='utf-8'))
    config['model']['version'] = VERSION
    config['data']['exp_dir'] = str(experiment)
    config.update(version=VERSION, name=experiment.name, s2_ckpt_dir=str(experiment), save_weight_dir=str(destination / 'sovits'))
    config['train'].update(epochs=epochs_s2, batch_size=batch_size, grad_ckpt=True, gpu_numbers='0',
                           save_every_epoch=5, if_save_latest=True, if_save_every_weights=True,
                           pretrained_s2G=str(required['generator']), pretrained_s2D=str(required['discriminator']))
    (destination / 'sovits').mkdir(exist_ok=True)
    (destination / 'gpt').mkdir(exist_ok=True)
    (experiment / f'logs_s2_{VERSION}').mkdir(exist_ok=True)
    (experiment / 'tmp_s2.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    config_s1 = {
        'data': {'max_eval_sample': 8, 'max_sec': 54, 'num_workers': 4, 'pad_val': 1024},
        'inference': {'top_k': 15},
        'model': {'EOS': 1024, 'dropout': 0, 'embedding_dim': 512, 'head': 16, 'hidden_dim': 512,
                  'linear_units': 2048, 'n_layer': 24, 'phoneme_vocab_size': 732, 'random_bert': 0, 'vocab_size': 1025},
        'optimizer': {'decay_steps': 40000, 'lr': 0.01, 'lr_end': 0.0001, 'lr_init': 0.00001, 'warmup_steps': 2000},
        'output_dir': str(experiment / f'logs_s1_{VERSION}'), 'pretrained_s1': str(required['gpt']),
        'train': {'batch_size': 4, 'epochs': epochs_s1, 'exp_name': experiment.name, 'gradient_clip': 1.0,
                  'half_weights_save_dir': str(destination / 'gpt'), 'if_dpo': False, 'if_save_every_weights': True,
                  'if_save_latest': True, 'precision': '16-mixed', 'save_every_n_epoch': 1, 'seed': 1234},
        'train_phoneme_path': str(experiment / '2-name2text.txt'),
        'train_semantic_path': str(experiment / '6-name2semantic.tsv')}
    (experiment / 'tmp_s1.yaml').write_text(yaml.safe_dump(config_s1, allow_unicode=True), encoding='utf-8')
    return required, destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', default='zihan_cn_20261001')
    parser.add_argument('--s2-epochs', type=int, default=20)
    parser.add_argument('--s1-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('WSL CUDA unavailable')
    experiment = GS / 'exp' / args.experiment
    experiment.mkdir(parents=True, exist_ok=True)
    lock = (experiment / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = args.root / 'data' / args.experiment
    rows = (data / 'train.list').read_text(encoding='utf-8').splitlines()
    if len(rows) < 700 or any(not Path(row.split('|', 1)[0]).is_file() for row in rows):
        raise RuntimeError('Training list incomplete or contains missing audio')
    quality = json.loads((data / 'teacher_quality.json').read_text(encoding='utf-8'))
    accepted = {Path(item['file']).name for item in quality if item['passed'] and item['cer'] <= 0.12}
    chinese = [Path(row.split('|')[0]).name for row in rows if row.split('|')[2] == 'zh']
    if len(chinese) < 600 or not set(chinese) <= accepted:
        raise RuntimeError('Training list does not satisfy the independent Chinese ASR quality gate')
    dataset_summary_path = data / 'dataset_summary.json'
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding='utf-8'))
    dataset_summary.update(asr_max_cer=0.12, asr_backend='paraformer')
    dataset_summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    fingerprint = hashlib.sha256()
    for row in rows:
        audio = Path(row.split('|', 1)[0])
        fingerprint.update(f'{row}|{audio.stat().st_size}|{audio.stat().st_mtime_ns}\n'.encode('utf-8'))
    fingerprint_path = experiment / 'dataset.sha256'
    if fingerprint_path.exists() and fingerprint_path.read_text().strip() != fingerprint.hexdigest():
        raise RuntimeError('Dataset changed; use a new experiment instead of reusing features/checkpoints')
    fingerprint_path.write_text(fingerprint.hexdigest(), encoding='utf-8')
    required, destination = configure(args.root, experiment, args.s2_epochs, args.s1_epochs, args.batch_size)
    environment = os.environ.copy()
    environment.update(inp_text=str(data / 'train.list'), inp_wav_dir=str(data / 'train_audio'),
                       exp_name=args.experiment, opt_dir=str(experiment), i_part='0', all_parts='1',
                       _CUDA_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0', is_half='True', version=VERSION,
                       bert_pretrained_dir=str(required['bert'].parent), cnhubert_base_dir=str(required['hubert'].parent),
                       sv_path=str(required['speaker']), pretrained_s2G=str(required['generator']),
                       s2config_path=str(GS / 'GPT_SoVITS/configs/s2v2ProPlus.json'), PYTHONIOENCODING='utf-8')
    for label, script, marker in [
        ('text', '1-get-text.py', '2-name2text.txt'),
        ('hubert', '2-get-hubert-wav32k.py', 'hubert.complete'),
        ('speaker', '2-get-sv.py', 'speaker.complete'),
        ('semantic', '3-get-semantic.py', '6-name2semantic.tsv')]:
        if (experiment / marker).exists():
            continue
        run([sys.executable, '-u', '-s', str(GS / 'GPT_SoVITS/prepare_datasets' / script)], experiment / f'log_{label}.txt', environment)
        if label == 'text':
            (experiment / marker).write_bytes((experiment / '2-name2text-0.txt').read_bytes())
        elif label == 'semantic':
            (experiment / marker).write_bytes((experiment / '6-name2semantic-0.tsv').read_bytes())
        else:
            (experiment / marker).write_text('done', encoding='utf-8')
    expected = {Path(row.split('|', 1)[0]).name for row in rows}
    phoneme_names = {row.split(chr(9), 1)[0] for row in (experiment / '2-name2text.txt').read_text(encoding='utf-8').splitlines()}
    semantic_names = {row.split(chr(9), 1)[0] for row in (experiment / '6-name2semantic.tsv').read_text(encoding='utf-8').splitlines()}
    if len(expected & phoneme_names & semantic_names) < len(expected) * 0.97:
        raise RuntimeError('Feature coverage below 97%; refusing incomplete training')
    training_environment = dict(environment, PYTHONPATH=str(GS) + ':' + str(GS / 'GPT_SoVITS'))
    config_path = experiment / 'tmp_s2.json'
    for attempt in range(2):
        try:
            run([sys.executable, '-u', '-s', 's2_train.py', '--config', str(config_path)], experiment / 'log_s2_train.txt', training_environment, GS / 'GPT_SoVITS')
            break
        except subprocess.CalledProcessError:
            recent = (experiment / 'log_s2_train.txt').read_text(encoding='utf-8', errors='replace')[-12000:]
            config = json.loads(config_path.read_text(encoding='utf-8'))
            if 'out of memory' not in recent.lower() or config['train']['batch_size'] <= 1 or attempt == 1:
                raise
            config['train']['batch_size'] = 1
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    config_s1_path = experiment / 'tmp_s1.yaml'
    for attempt in range(3):
        try:
            run([sys.executable, '-u', '-s', str(GS / 'GPT_SoVITS/s1_train.py'), '--config_file', str(config_s1_path)], experiment / 'log_s1_train.txt', training_environment)
            break
        except subprocess.CalledProcessError:
            recent = (experiment / 'log_s1_train.txt').read_text(encoding='utf-8', errors='replace')[-12000:]
            config = yaml.safe_load(config_s1_path.read_text(encoding='utf-8'))
            if 'out of memory' not in recent.lower() or config['train']['batch_size'] <= 1 or attempt == 2:
                raise
            config['train']['batch_size'] = max(1, config['train']['batch_size'] // 2)
            config_s1_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    final_s2 = [path for path in (destination / 'sovits').glob('*.pth')
                if re.search(rf'_e{args.s2_epochs}_', path.name) and path.stat().st_size > 1024]
    final_s1 = [path for path in (destination / 'gpt').glob('*.ckpt')
                if re.search(rf'-e{args.s1_epochs}(?:\.|$)', path.name) and path.stat().st_size > 1024]
    if not final_s2 or not final_s1:
        raise RuntimeError('Training exited without final-epoch inference weights')
    summary = {'experiment': str(experiment), 'version': VERSION, 'samples': len(rows),
               's1_epochs': args.s1_epochs, 's2_epochs': args.s2_epochs, 'models': str(destination)}
    (destination / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
