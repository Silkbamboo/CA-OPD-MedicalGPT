# Engineering notes

The project reached a complete experimental decision through several fail-closed incidents.
The items below are condensed technical summaries; raw operator logs, private runbooks and
interview notes are intentionally not public.

## 1. DataParallel aggregation OOM

The first weighted SFT-v2 path gathered full-vocabulary logits on GPU0. Both cards had free
capacity in aggregate, but the asymmetric gather left only a few MiB on rank 0 and failed
before optimizer step 1.

The repair used one process per GPU, local loss computation and global denominator reduction
with DDP. Effective batch, supervision weights and seed remained fixed. SFT-v3 later ran 600
steps with peak memory around 13.60/13.50 GiB.

## 2. LoRA prompt-logprob drift

The Base route was repeatable after warm-up, while the vLLM 0.11 LoRA `prompt_logprobs` path
showed score drift far above the registered `1e-4` tolerance. Candidate token IDs, finite
scores, adapter hashes and label isolation were verified. Warm-up and engine cleanup did not
remove the drift.

The scoring definition was kept unchanged and moved to a Transformers direct-logit backend.
The vLLM route remained diagnostic. The exact low-level cause was not proven, so the project
does not label this as a confirmed upstream bug.

## 3. Extreme per-prompt gradient

At a fixed historical batch, three raw prompt gradient norms were 0.3342, 0.3076 and 0.1445;
one CMB trajectory was 126.9449. A minimum response length would have rejected many legitimate
short CMB answers and silently changed the source weighting.

The shared protocol instead bounds each of four prompt contributions at 0.25 before summing,
then applies the existing global clip of 1.0. The same fixed-token candidate produced a
post-update ratio maximum of 1.5054 instead of the earlier extreme tail. All diagnostic
candidates were rolled back before formal fresh-v0 training.

## 4. Runtime package/schema mismatch

A fixed-token qualification proved the intended update mathematics, but a subsequent canary
failed before model load because the protocol file lacked a configuration section required by
the production kernel. The evidence file was immutable and SHA-bound, so it was not edited in
place. A new version copied the same eight mathematical mappings and added only the missing
runtime schema, then repeated the qualification.

## 5. Transactional training and recovery

Every candidate update snapshots the Student, optimizer, scheduler, RNG, data cursor and
sampler version. Health gates either commit the complete transition or restore all states.
Checkpoints use completeness markers and hashes; a failure-only adapter is not resume-eligible.

These controls increase engineering reliability. They do not turn a statistically unsupported
capability result into a positive one.

## 6. Development peak did not replicate

The 300-question Controller selected step240 at +4 questions over Base. The 600-question
prediction-first confirmation produced 10 improvements and 10 regressions. This is the most
important scientific failure mode in the project: a plausible development trend was not a
reproducible gain. The protocol therefore stopped without opening final data or tuning against
the confirmation result.
