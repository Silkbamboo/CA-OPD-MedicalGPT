# Sanitized protocol snapshots

These files preserve the hyperparameters and protocol identities used by the reported
experiments, while replacing machine-specific absolute paths with repository-relative
`artifacts/...` placeholders.

They are evidence and reconstruction templates, not a way to reuse the historical hashes on a
different model, dataset or Git commit. A new run must rebuild manifests/packages and receive a
new identity. Model weights, processed data, labels and checkpoints are not distributed.
