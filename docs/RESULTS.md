# Results and interpretation

## Summary

- SFT-v3 produced a confirmed medical improvement on the frozen 600-question development
  confirmation set.
- The same SFT checkpoint also improved the selected General Controller, so the expected
  forgetting premise was not observed in this experiment.
- Medical OPD showed a small development-set peak at step 240, but an independent B2
  confirmation on 600 questions produced exactly the same accuracy as Base.
- The equal-budget 120-step CA-OPD arm did not outperform fixed IDT.

Machine-readable aggregate summaries are in [`artifacts/results`](../artifacts/results).

## SFT-v3 confirmation

| Route | Correct / 600 | Accuracy | Delta vs Base | Paired 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| B0 | 443 | 73.83% | - | - | - |
| B1 SFT-v3 step450 | 467 | 77.83% | +4.00pp | `[+1.17,+7.00]pp` | 0.0116 |

Paired outcomes were 54 improved, 30 regressed and 516 unchanged. This supports the narrow
claim that the frozen SFT route improved accuracy on this development-confirmation protocol.
It is not a final-test or clinical-validity claim.

## Equal-budget controller results

| Route | Medical correct / 300 | General correct / 209 | Medical | General |
|---|---:|---:|---:|---:|
| B0 | 219 | 128 | 73.00% | 61.244% |
| B1 SFT-v3 | 240 | 139 | 80.00% | 66.507% |
| B2 step120 | 217 | 128 | 72.33% | 61.244% |
| IDT step120 | 217 | 127 | 72.33% | 60.766% |
| CA step120 | 216 | 126 | 72.00% | 60.287% |

At step120, CA minus IDT was -0.33pp Medical and -0.48pp General; both paired confidence
intervals crossed zero. The evidence therefore does not support CA superiority.

## B2 dose-response curve

| Accepted step | Medical correct / 300 | General correct / 209 |
|---:|---:|---:|
| 120 | 217 | 128 |
| 150 | 218 | 126 |
| 180 | 216 | 124 |
| 200 | 221 | 123 |
| 240 | 223 | 126 |
| 270 | 218 | 124 |
| 300 | 218 | 126 |

The pre-registered development rule selected step240. Its Medical delta over B0 was 4/300
(+1.33pp), paired CI `[-1.00,+4.00]pp`, exact McNemar `p=0.424`. Later checkpoints declined,
which is why the development point was treated as a candidate rather than a result.

## Independent B2 confirmation

| Route | Correct / 600 | Accuracy |
|---|---:|---:|
| B0 | 443 | 73.8333% |
| B2 step240 | 443 | 73.8333% |

- delta: `0.00pp`;
- improved / regressed / unchanged: `10 / 10 / 580`;
- paired bootstrap 95% CI: `[-1.50,+1.50]pp`;
- exact McNemar `p=1.0`.

The registered decision is `b2_step240_confirmation_not_supported`. The project stopped B2
and did not try another checkpoint, seed, prompt template or response length after observing
this result.

## What may and may not be claimed

Supported:

- the SFT result under the frozen development-confirmation protocol;
- implementation of B2, IDT and CA-OPD under a shared production loop;
- robust training/evaluation infrastructure and an honestly reported negative OPD result.

Not supported:

- stable Medical OPD improvement over Base;
- CA-OPD superiority over IDT;
- general-capability preservation under an improved medical result;
- final-test, multi-seed, clinical or state-of-the-art performance.
