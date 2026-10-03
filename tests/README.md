# Mothra tests

See [runtime/security validation](RUNTIME_VALIDATION.md) for the current
Python 3.12 environment, historical-reference provenance, both compatibility
modes and reproduction instructions.

The original joblib CVE-2022-21797 regression remains covered; the toxicity
model now uses numeric-only NPZ inference. [TLS/IDNA isolation evidence](TRUST_VALIDATION.md)
is retained, and those checks also pass in the final environment.
