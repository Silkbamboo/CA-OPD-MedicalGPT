# GRPO v9 Evaluation Report

Pred file: `/root/MedicalGPT/eval/v9_eval_results/v9_preds_300.jsonl`

## specialized_150  (n=150)

| metric | v9 | v8 baseline | delta |
|--------|---:|------------:|------:|
| Overall | 0.5932 | 0.5785 | ++0.0147 |
| Triage | 0.4267 | 0.6557 | -0.2290 |
| RedFlag | 0.5733 | 0.4553 | ++0.1180 |
| Actionability | 0.7167 | 0.7207 | -0.0040 |
| Unsafe Rate | 0.0000 | 0.0133 | -0.0133 |
| Triage Exact Acc | 0.3133 | 0.4667 | -0.1534 |
| UE No-Undertriage | 0.8154 | 0.8500 | -0.0346 |

### By Triage Label

| triage | overall | exact_acc | undertriage_rate | count |
|--------|--------:|----------:|-----------------:|------:|
| home_observe | 0.4370 | 0.1633 | 0.1633 | 49 |
| outpatient | 0.5845 | 0.0833 | 0.2222 | 36 |
| urgent | 0.7103 | 0.3333 | 0.2308 | 39 |
| emergency | 0.7239 | 0.8846 | 0.1154 | 26 |

### Worst 5 Categories (by overall)

| category | overall | triage | redflag | actionability | unsafe_rate | n |
|----------|--------:|-------:|--------:|--------------:|------------:|--:|
| missed_antihypertensive_dose | 0.2846 | 0.0769 | 0.0000 | 0.6154 | 0.0000 | 13 |
| mild_upper_respiratory | 0.4350 | 0.4000 | 0.0000 | 0.7000 | 0.0000 | 10 |
| pediatric_fever | 0.4433 | 0.0769 | 0.5385 | 0.7115 | 0.0000 | 13 |
| mental_health | 0.4962 | 0.2692 | 0.5000 | 0.6538 | 0.0000 | 13 |
| diarrhea_dehydration | 0.5846 | 0.1538 | 0.8462 | 0.8462 | 0.0000 | 13 |

### Best 5 Categories

| category | overall | triage | redflag | actionability | unsafe_rate | n |
|----------|--------:|-------:|--------:|--------------:|------------:|--:|
| persistent_headache | 0.6702 | 0.3077 | 1.0000 | 0.7885 | 0.0000 | 13 |
| drug_allergy | 0.7077 | 0.6538 | 0.7308 | 0.6538 | 0.0000 | 13 |
| pregnancy_bleeding | 0.7096 | 0.5385 | 0.6923 | 0.8846 | 0.0000 | 13 |
| warfarin_bleeding | 0.7135 | 0.5000 | 1.0000 | 0.6538 | 0.0000 | 13 |
| cardiac_emergency | 0.7785 | 0.9231 | 0.5641 | 0.6731 | 0.0000 | 13 |

## natural_150  (n=150)

| metric | v9 | v8 baseline | delta |
|--------|---:|------------:|------:|
| Overall | 0.3026 | 0.3069 | -0.0043 |
| Triage | 0.1333 | 0.2167 | -0.0834 |
| RedFlag | 0.2689 | 0.2267 | ++0.0422 |
| Actionability | 0.3283 | 0.5600 | -0.2317 |
| Unsafe Rate | 0.0000 | 0.0400 | -0.0400 |
| Triage Exact Acc | 0.0533 | 0.1133 | -0.0600 |
| UE No-Undertriage | 0.1875 | 0.3800 | -0.1925 |

### By Triage Label

| triage | overall | exact_acc | undertriage_rate | count |
|--------|--------:|----------:|-----------------:|------:|
| home_observe | 0.2161 | 0.0556 | 0.6806 | 72 |
| outpatient | 0.3151 | 0.0000 | 0.7391 | 46 |
| urgent | 0.5295 | 0.1818 | 0.7273 | 22 |
| emergency | 0.3688 | 0.0000 | 1.0000 | 10 |

### Worst 5 Categories (by overall)

| category | overall | triage | redflag | actionability | unsafe_rate | n |
|----------|--------:|-------:|--------:|--------------:|------------:|--:|
| medication_consult | 0.2216 | 0.0909 | 0.0909 | 0.2500 | 0.0000 | 11 |
| dermatology_consult | 0.2273 | 0.0909 | 0.1364 | 0.2273 | 0.0000 | 11 |
| cardiovascular_consult | 0.2278 | 0.0417 | 0.1528 | 0.2917 | 0.0000 | 12 |
| pediatric_consult | 0.2703 | 0.0938 | 0.2500 | 0.2812 | 0.0000 | 16 |
| general_medical_consult | 0.2732 | 0.1429 | 0.2143 | 0.2500 | 0.0000 | 14 |

### Best 5 Categories

| category | overall | triage | redflag | actionability | unsafe_rate | n |
|----------|--------:|-------:|--------:|--------------:|------------:|--:|
| chronic_disease_consult | 0.3250 | 0.2500 | 0.1000 | 0.4000 | 0.0000 | 10 |
| neurology_consult | 0.3383 | 0.1562 | 0.3438 | 0.3594 | 0.0000 | 16 |
| respiratory_consult | 0.3449 | 0.1364 | 0.3409 | 0.4205 | 0.0000 | 22 |
| gynecology_pregnancy_consult | 0.3549 | 0.1944 | 0.3889 | 0.3194 | 0.0000 | 18 |
| mental_health_consult | 0.4188 | 0.2500 | 0.5625 | 0.3125 | 0.0000 | 8 |

## combined_300  (n=300)

| metric | v9 | v8 baseline | delta |
|--------|---:|------------:|------:|
| Overall | 0.4479 | nan | N/A |
| Triage | 0.2800 | nan | N/A |
| RedFlag | 0.4211 | nan | N/A |
| Actionability | 0.5225 | nan | N/A |
| Unsafe Rate | 0.0000 | nan | N/A |
| Triage Exact Acc | 0.1833 | nan | N/A |
| UE No-Undertriage | 0.6082 | nan | N/A |

### By Triage Label

| triage | overall | exact_acc | undertriage_rate | count |
|--------|--------:|----------:|-----------------:|------:|
| home_observe | 0.3056 | 0.0992 | 0.4711 | 121 |
| outpatient | 0.4334 | 0.0366 | 0.5122 | 82 |
| urgent | 0.6451 | 0.2787 | 0.4098 | 61 |
| emergency | 0.6252 | 0.6389 | 0.3611 | 36 |

### Worst 5 Categories (by overall)

| category | overall | triage | redflag | actionability | unsafe_rate | n |
|----------|--------:|-------:|--------:|--------------:|------------:|--:|
| medication_consult | 0.2216 | 0.0909 | 0.0909 | 0.2500 | 0.0000 | 11 |
| dermatology_consult | 0.2273 | 0.0909 | 0.1364 | 0.2273 | 0.0000 | 11 |
| cardiovascular_consult | 0.2278 | 0.0417 | 0.1528 | 0.2917 | 0.0000 | 12 |
| pediatric_consult | 0.2703 | 0.0938 | 0.2500 | 0.2812 | 0.0000 | 16 |
| general_medical_consult | 0.2732 | 0.1429 | 0.2143 | 0.2500 | 0.0000 | 14 |

### Best 5 Categories

| category | overall | triage | redflag | actionability | unsafe_rate | n |
|----------|--------:|-------:|--------:|--------------:|------------:|--:|
| persistent_headache | 0.6702 | 0.3077 | 1.0000 | 0.7885 | 0.0000 | 13 |
| drug_allergy | 0.7077 | 0.6538 | 0.7308 | 0.6538 | 0.0000 | 13 |
| pregnancy_bleeding | 0.7096 | 0.5385 | 0.6923 | 0.8846 | 0.0000 | 13 |
| warfarin_bleeding | 0.7135 | 0.5000 | 1.0000 | 0.6538 | 0.0000 | 13 |
| cardiac_emergency | 0.7785 | 0.9231 | 0.5641 | 0.6731 | 0.0000 | 13 |

