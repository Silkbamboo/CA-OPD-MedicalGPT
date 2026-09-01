# Data Protocol v2

## Principle

Public datasets are upstream sources; the project owns the cleaning, normalization,
deduplication, role assignment, prompt-only export, manifests and capability boundaries. Raw
benchmark questions, labels and full processed corpora are not distributed in this repository.

## Pinned upstream sources

| Source | Pinned revision | License recorded by the project | Roles |
|---|---|---|---|
| Medical-O1 Chinese | `fc2c9e8a37b38f38da6d449564a8c350b244aef4` | Apache-2.0 | SFT and medical OPD |
| CMB | `935fbc09edf1303d89872b21265ff597f426ac0d` | Apache-2.0 | SFT MCQ bridge and medical OPD |
| MedQA-zh | source revisions are bound in generated manifests | unknown | medical controller/confirmation/final candidates |
| COIG family | `9f25758ec94f82762fb9c09a5c60e908cfb83632` | checked per subsource | general anchors |
| GPT4-LLM Chinese Alpaca | `80cda626ea305004be42426671c66efebbf22144` | CC-BY-NC-4.0 | general anchors, noncommercial research only |
| C-Eval | `617524a00b307ff6f9933702f724131fe12ca7ce` | CC BY-NC-SA 4.0 | general controller/final candidates |

License metadata is evidence about the reviewed revision, not legal advice. Unknown or
unresolved subsources fail closed. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Roles and access

| Role | Size in the frozen experiment | Training access | Label access |
|---|---:|---|---|
| `medical_sft_train` | 9,600 in SFT-v3 (7,200 CMB + 2,400 Medical-O1) | SFT only | SFT target is available |
| `medical_opd_o1` | prompt pool | OPD only | physically removed |
| `medical_opd_cmb` | prompt pool | OPD only | physically removed |
| `general_anchors` | 3,793 (3,200 GPT4-LLM Chinese Alpaca + 593 COIG-LeetCode) | IDT/CA Base route only | physically removed |
| `medical_controller_dev` | 300 | no | evaluator process only |
| `general_controller_dev` | 209 | no | evaluator process only |
| `medical_teacher_confirmation_dev` | 600 | no | one prediction-first join per registered comparison |
| final capabilities | frozen, never evaluated | forbidden | unopened in this project |

The 600-question development confirmation was frozen before the B2 candidate existed. It had
previously been used once to confirm the SFT Teacher, so it is not called an untouched final
test. P10 was the first B2 access to that capability.

## Normalized schema

Adapters normalize source-specific fields to a common schema containing stable IDs, source
revision and license, upstream split, target role, question, ordered options, answer/reasoning
where permitted, normalized representations, content hash, near-duplicate group and quality
flags. The implementation lives in [`src/data`](../src/data).

Normalization preserves clinically important tokens such as dosage, units, negation and
positive/negative status. Option order is never sorted.

## Leakage controls

1. Normalize and exact-deduplicate before splitting.
2. Keep every near-duplicate `group_id` inside one role.
3. Build a denylist from controller, confirmation and final normalized hashes.
4. Reject any training or OPD export that intersects that denylist.
5. Strip `answer`, `answer_idx`, `label`, `reasoning`, `response`, `solution`, `output` and
   `completion` from OPD records.
6. Give final evaluation a separate capability and explicit authorization; it is not a normal
   `split=final` option.

The public fixtures under [`tests/fixtures`](../tests/fixtures) are synthetic and exist only to
exercise adapters and leakage guards.

## Manual-audit boundary

The formal build passed automated schema, exact-hash, role-overlap and supervision-field gates.
Its near-duplicate scan produced 433 candidate pairs; 23 crossed protected roles and were
conservatively resolved with no unresolved cross-role candidate remaining. The planned manual
row-by-row audit was explicitly waived for the time-constrained interview MVP. The precise status
is therefore `formal_ready_mvp_waived`, not fully human-audited data. This limitation applies to
all reported training results.
