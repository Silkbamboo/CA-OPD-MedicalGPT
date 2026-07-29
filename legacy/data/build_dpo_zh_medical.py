# -*- coding: utf-8 -*-
"""Build a Chinese DPO train/val split for the medical SFT -> DPO route."""
import json
import random
from pathlib import Path

SEED = 42
VAL_RATIO = 0.01

SRC = Path('/root/MedicalGPT/data/reward_raw/dpo_zh.jsonl')
OUT_ROOT = Path('/root/MedicalGPT/data/dpo_mix/zh_for_medical_general')
TRAIN_DIR = OUT_ROOT / 'train'
VAL_DIR = OUT_ROOT / 'val'
TRAIN_FILE = TRAIN_DIR / 'dpo_en_zh_20k_preference_zh.jsonl'
VAL_FILE = VAL_DIR / 'dpo_en_zh_20k_preference_zh.jsonl'
MANIFEST = OUT_ROOT / 'mixture_manifest.json'


def main():
    if not SRC.exists():
        raise FileNotFoundError(f'Missing source file: {SRC}')
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    with SRC.open('r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]

    required = {'system', 'history', 'question', 'response_chosen', 'response_rejected'}
    cleaned = []
    for row in rows:
        if not required.issubset(row):
            continue
        if not row['question'] or not row['response_chosen'] or not row['response_rejected']:
            continue
        cleaned.append({
            'system': row.get('system', ''),
            'history': row.get('history', []) or [],
            'question': row['question'],
            'response_chosen': row['response_chosen'],
            'response_rejected': row['response_rejected'],
        })

    rng = random.Random(SEED)
    rng.shuffle(cleaned)
    val_count = max(100, int(len(cleaned) * VAL_RATIO)) if cleaned else 0
    val_rows = cleaned[:val_count]
    train_rows = cleaned[val_count:]

    def dump(path: Path, items):
        with path.open('w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    dump(TRAIN_FILE, train_rows)
    dump(VAL_FILE, val_rows)

    manifest = {
        'source': str(SRC),
        'seed': SEED,
        'val_ratio': VAL_RATIO,
        'counts': {
            'total': len(cleaned),
            'train': len(train_rows),
            'val': len(val_rows),
        },
        'files': {
            'train': str(TRAIN_FILE),
            'val': str(VAL_FILE),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
