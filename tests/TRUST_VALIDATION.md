# certifi / idna security update

Validated 2026-10-03 on macOS ARM64, Python 3.9.25, in two new independent venvs.
Only certifi and idna changed; `trust-environment-comparison.json` records all
14 installed package versions, OpenSSL version and CA subjects in both environments.

- certifi 2022.5.18.1 -> 2026.7.22: remove distrust-listed roots, including
  TrustCor, e-Tugra and GLOBALTRUST, following the CA trust-store updates.
  See the official [GLOBALTRUST advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-248v-346w-9cwc)
  and the dated OSV evidence in `trust-osv-snapshot.json`.
- idna 3.3 -> 3.20: address malformed-input resource consumption, including
  [CVE-2024-3651](https://github.com/kjd/idna/security/advisories/GHSA-jjg7-2v4v-x38h).
  The selected version remains compatible with Python 3.9 and existing Requests.

OSV matched advisories for the original versions and none for either exact new
version at this snapshot. This does not establish that the complete application
is free of vulnerabilities or that a particular application path was exploited.

## Validation

Both environments pass dependency consistency checks. Requests 2.27.1,
urllib3 1.26.9, charset-normalizer 2.0.12, requests-oauthlib 1.3.1,
oauthlib 3.2.0, google-auth 2.6.6 and google-auth-oauthlib 0.4.6 remain unchanged.

Five compatibility checks pass in both environments:

1. Five ASCII/Unicode hostnames encode/decode and produce identical prepared
   Requests URLs (including German, Japanese and Arabic).
2. Invalid bidi/label-length/hyphen/empty-label hostnames are rejected.
3. A localhost HTTPS fixture with an explicitly trusted test certificate supports
   JSON GET, POST and a redirect without changing payloads.
4. Default CA verification rejects the untrusted localhost fixture, and the
   supplied certifi trust store loads successfully.
5. Existing Google anonymous-auth transport performs the local HTTPS request;
   OAuth URL construction works with a dummy client and fixed state. No real
   credentials, token refresh, external service or external message is involved.

The sixth test asserts that TrustCor/e-Tugra/GLOBALTRUST certificates are absent
from the new trust store. It fails with the old version and passes after the
update. All six updated-version tests pass. This is an intended trust-policy
change: connections relying solely on removed roots may cease to validate.
Unicode database updates can change behavior for newly introduced characters;
the domain fixtures are a compatibility sample, not exhaustive IDNA equivalence.

## Reproduction

From the repository root, use a new Python 3.9 environment:

```sh
uv venv --python /path/to/python3.9 /tmp/mothra-trust-updated
uv --no-config pip install --python /tmp/mothra-trust-updated/bin/python \
  requests==2.27.1 urllib3==1.26.9 charset-normalizer==2.0.12 \
  certifi==2026.7.22 idna==3.20 requests-oauthlib==1.3.1 oauthlib==3.2.0 \
  google-auth==2.6.6 google-auth-oauthlib==0.4.6
/tmp/mothra-trust-updated/bin/python -m unittest discover -s tests -p test_trust_dependencies.py -v
uv pip check --python /tmp/mothra-trust-updated/bin/python
```

An `openssl` executable supporting `req -addext` is required for the local
one-day fixture. Tests bind only to localhost and make no external requests.
For the old comparison use a second new environment and replace only certifi
with 2022.5.18.1 and idna with 3.3. Five tests pass and the distrusted-root test
intentionally fails. Do not use that old environment for production.

## Remaining work

This PR does not change Python, TensorFlow/Keras, the RNN, toxicity predictor,
CUDA, Vina, Open Babel, or the other HTTP dependencies. Full requirements,
container execution, training and docking were not tested. Existing environments
were not modified. These auth/HTTP integration checks do not establish
compatibility of every untested optional consumer or endpoint.

Current Python-3.9-compatible Requests 2.32.5 and urllib3 2.6.3 still matched
OSV advisories at the investigation date. The latest patched candidates require
Python 3.10 or later, so those updates must accompany a separately validated
runtime migration. Python 3.9 itself is end of life. The saved RNN JSON identifies
Keras 2.9.0, matching requirements but conflicting with README's older guidance;
do not use the README's Keras 2.0.5 / TensorFlow 1.15.2 statements as the numerical
reference for that migration.
