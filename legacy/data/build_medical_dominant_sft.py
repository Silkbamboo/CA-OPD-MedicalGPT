import argparse
import json
import math
import random
from pathlib import Path

SOURCES = {
    'medical': [
        ('shibing624_medical_sft', Path('/root/MedicalGPT/data/finetune/shibing624_medical_sft.jsonl')),
        ('huatuogpt_sft_220k', Path('/root/MedicalGPT/data/finetune/huatuogpt_sft_220k.jsonl')),
    ],
    'general': [
        ('general_sharegpt_gpt4', Path('/root/MedicalGPT/data/finetune/general_sharegpt_gpt4.jsonl')),
        ('general_alpaca_zh', Path('/root/MedicalGPT/data/finetune/general_alpaca_zh.jsonl')),
        ('general_sharegpt_cn_38k', Path('/root/MedicalGPT/data/finetune/general_sharegpt_cn_38k.jsonl')),
        ('general_stanford_alpaca_50k', Path('/root/MedicalGPT/data/finetune/general_stanford_alpaca_50k.jsonl')),
    ],
}


def count_lines(path: Path) -> int:
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def allocate_counts(group, target_total):
    total_available = sum(item['available'] for item in group)
    if target_total > total_available:
        raise ValueError(f'target_total={target_total} exceeds available={total_available}')

    allocated = []
    running = 0
    for idx, item in enumerate(group):
        if idx == len(group) - 1:
            n = target_total - running
        else:
            n = math.floor(target_total * item['available'] / total_available)
            running += n
        allocated.append(n)

    current = sum(allocated)
    remainder = target_total - current
    if remainder > 0:
        order = sorted(range(len(group)), key=lambda i: group[i]['available'], reverse=True)
        for i in order[:remainder]:
            allocated[i] += 1

    for item, n in zip(group, allocated):
        item['sampled'] = n
    return group


def choose_indices(available: int, sampled: int, val_ratio: float, rng: random.Random):
    selected = sorted(rng.sample(range(available), sampled))
    val_count = max(1, round(sampled * val_ratio)) if sampled > 0 else 0
    val_positions = set(rng.sample(range(sampled), val_count)) if val_count else set()
    train_indices = []
    val_indices = []
    for pos, idx in enumerate(selected):
        if pos in val_positions:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    return train_indices, val_indices


def write_selected(src, out: Path, selected_indices):
    src = Path(src)
    selected = set(selected_indices)
    written = 0
    with src.open('r', encoding='utf-8') as fin, out.open('w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i in selected and line.strip():
                fout.write(line)
                written += 1
    return written


parser = argparse.ArgumentParser(description='Build a medical-dominant SFT mixture for MedicalGPT.')
parser.add_argument('--medical_ratio', type=float, default=0.85)
parser.add_argument('--general_ratio', type=float, default=0.15)
parser.add_argument('--general_keep', choices=['all'], default='all')
parser.add_argument('--val_ratio', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', default='/root/MedicalGPT/data/finetune_mix/medical_dominant_85_15')
args = parser.parse_args()

if abs(args.medical_ratio + args.general_ratio - 1.0) > 1e-8:
    raise ValueError('medical_ratio + general_ratio must equal 1.0')
if args.general_keep != 'all':
    raise ValueError('only --general_keep all is supported')

rng = random.Random(args.seed)
out_dir = Path(args.output_dir)
train_dir = out_dir / 'train'
val_dir = out_dir / 'val'
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

medical_group = [{'name': name, 'path': str(path), 'available': count_lines(path)} for name, path in SOURCES['medical']]
general_group = [{'name': name, 'path': str(path), 'available': count_lines(path)} for name, path in SOURCES['general']]
medical_available = sum(item['available'] for item in medical_group)
general_available = sum(item['available'] for item in general_group)

general_target = general_available
medical_target = round(general_target * args.medical_ratio / args.general_ratio)
if medical_target > medical_available:
    raise ValueError(
        f'medical target {medical_target} exceeds available {medical_available}; reduce medical_ratio or add more general data'
    )

medical_group = allocate_counts(medical_group, medical_target)
general_group = allocate_counts(general_group, general_target)

train_counts = {'medical': 0, 'general': 0}
val_counts = {'medical': 0, 'general': 0}

for group_name, group in [('medical', medical_group), ('general', general_group)]:
    for item in group:
        train_idx, val_idx = choose_indices(item['available'], item['sampled'], args.val_ratio, rng)
        train_out = train_dir / f"{item['name']}.jsonl"
        val_out = val_dir / f"{item['name']}.jsonl"
        item['train_written'] = write_selected(item['path'], train_out, train_idx)
        item['val_written'] = write_selected(item['path'], val_out, val_idx)
        train_counts[group_name] += item['train_written']
        val_counts[group_name] += item['val_written']

manifest = {
    'seed': args.seed,
    'medical_ratio': args.medical_ratio,
    'general_ratio': args.general_ratio,
    'general_keep': args.general_keep,
    'medical_available': medical_available,
    'general_available': general_available,
    'medical_target_before_split': medical_target,
    'general_target_before_split': general_target,
    'train_counts_after_split': train_counts,
    'val_counts_after_split': val_counts,
    'medical_sources': medical_group,
    'general_sources': general_group,
    'train_file_dir': str(train_dir),
    'validation_file_dir': str(val_dir),
    'recommended_sft_command': (
        'python supervised_finetuning.py '
        '--train_file_dir ' + str(train_dir) + ' '
        '--validation_file_dir ' + str(val_dir)
    ),
}
(out_dir / 'mixture_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
