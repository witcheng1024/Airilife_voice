import json
import logging
import re
from pathlib import Path


def _resource(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Voice resource is outside project: {value}')
    return path


def load_speakers(root, gpt, sovits, emotions):
    root = Path(root)
    speakers = {'firefly': {
        'id': 'firefly', 'label': '流萤', 'gpt': Path(gpt), 'sovits': Path(sovits),
        'text_language': 'auto',
        'references': {emotion: {'audio': root / 'models' / 'airi' / 'references' / values[0],
                                  'text': values[1], 'language': 'zh'}
                       for emotion, values in emotions.items()},
    }}
    for manifest in sorted((root / 'models').glob('*/voice.json')):
        try:
            config = json.loads(manifest.read_text(encoding='utf-8'))
            if not isinstance(config, dict):
                raise ValueError('Voice manifest must be an object')
            speaker_id = config['id']
            if not isinstance(speaker_id, str) or not re.fullmatch('[a-zA-Z0-9_-]+', speaker_id) or speaker_id in speakers:
                raise ValueError('Invalid or duplicate speaker id')
            references = config['references']
            if not isinstance(references, dict) or not references:
                raise ValueError('References must be a nonempty object')
            if any(not isinstance(item, dict) or not isinstance(item.get('text'), str)
                   or not item['text'].strip() for item in references.values()):
                raise ValueError('Empty references or prompt text')
            speaker = dict(config)
            speaker['label'] = str(config.get('label', speaker_id))
            speaker['gpt'] = _resource(root, config['gpt'])
            speaker['sovits'] = _resource(root, config['sovits'])
            speaker['references'] = {
                emotion: dict(item, audio=_resource(root, item['audio']))
                for emotion, item in references.items()
            }
            speakers[speaker_id] = speaker
        except (ValueError, KeyError, TypeError) as error:
            logging.getLogger('airi_tts').warning('Skip invalid voice manifest %s: %s', manifest, error)
    return speakers


def is_ready(speaker):
    return (speaker['gpt'].is_file() and speaker['sovits'].is_file()
            and all(item['audio'].is_file() for item in speaker['references'].values()))


def public_speakers(speakers):
    return [{'id': speaker['id'], 'label': speaker['label'], 'ready': is_ready(speaker),
             'emotions': list(speaker['references'])} for speaker in speakers.values()]
