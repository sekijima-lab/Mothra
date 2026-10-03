# Historical Mothra runtime investigation

The user requested that the RNN update remain on hold while the numerical
cause is investigated (2026-10-03). This document records the investigation before the user subsequently approved
default-on, switchable compatibility mode. Final integration and validation are
described in ../RUNTIME_VALIDATION.md.

## Scope and baseline

Historical model: bundled Keras 2.9 JSON/HDF5, 64-token vocabulary, two
256-unit GRUs, dropout 0.2 and TimeDistributed softmax. All nine initial weight
tensors and the complete vocabulary match exactly. CPU comparison uses one
thread and inference mode (dropout disabled).

Baseline: Python 3.9.25, tensorflow-macos 2.9.1, Keras 2.9.0, NumPy 1.22.4,
RDKit 2022.03.4; h5py 3.7.0 substitutes for the unavailable arm64 3.6.0 wheel.
Candidate: Python 3.12.15, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3,
RDKit 2026.03.6. The platform is macOS arm64. This is not an exact historical
Linux/CUDA reconstruction. Existing research environments were not modified.

## Strict comparison and independent checks

The original thresholds remain unchanged in criteria.json. With duplicate
embedding gradients summed before Adam's second-moment update:

- 256 molecules, 81 positions, 64 probabilities: max absolute difference
  4.5180320739746094e-5 exceeds 1e-5; mean difference 5.993537222082068e-9
  passes 1e-6. All argmax tokens match.
- Loss matches; max gradient difference 1.156982034444809e-5 exceeds 1e-5.
- One Adam step: max weight difference 4.410743713378906e-6 passes 1e-4.
  Without duplicate aggregation, the embedding update differs by 1.4901e-4.
- All nine fixed-seed generation cases have identical expansion tokens and
  completed sequences. This is a finite fixture comparison, not a guarantee
  for every seed or cumulative training trajectory.
- 2,048 molecular fingerprints, toxicity probabilities and acceptance decisions
  match exactly; QED max difference 1.11e-16 and SA scores match.
- 63 three-dimensional hypervolume fixtures have maximum difference 7.11e-15
  and identical candidate rankings.
- The 32-molecule docking/filter fixture accepts the same seven molecules and
  returns identical scores. Open Babel and Vina subprocesses were mocked;
  real docking was not validated.
- Training CLI completed one epoch on 64 molecules with TensorBoard enabled,
  an isolated scratch configuration and `.keras`/weights output. Two optimizer
  tests cover duplicate sparse rows over multiple updates and serialization.

## Cause: CPU activation kernels

On identical historical inputs and hidden states, both input and recurrent
matrix products match element-for-element in 14 probes (two GRUs, seven time
points). Differences first appear in sigmoid and tanh. A 100,001-point
float32 grid from -20 to 20 gives maximum differences of 1.1920929e-7 for
sigmoid and 2.9802322e-7 for tanh.

Standalone C++ programs compiled against each TensorFlow wheel's own Eigen
headers reproduce that wheel's sigmoid and tanh grid **exactly**. The legacy build does not define EIGEN_VECTORIZE_FMA, whereas the modern
build does. This affects the selected tanh clamp; it does not mean that all
legacy arithmetic is unfused: its ARM64 NEON pmadd uses vfmaq_f32. The tanh
rational approximation and its coefficients also changed:

- Legacy: Eigen/src/Core/MathFunctionsImpl.h, generic_fast_tanh_float.
- Modern: Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h, ptanh_float.

Replacing only modern GRU activations with the legacy Eigen functions in a
**diagnostic** native library reduces the complete model's max probability
difference to 5.9604645e-8 and mean difference to 2.0069538e-10. Maximum gradient
difference becomes 6.2864274e-9 and the Adam weight difference 4.7683716e-7.
All original numerical criteria then pass. This intervention identifies the
activation implementation as the cause of the failing comparisons.

The diagnostic library is not a proposed production dependency. A subsequent diagnostic reproduces the old approximation using TensorFlow
operations with float64 intermediate multiply-adds. Its complete-model
probability, gradient and Adam differences are the same as the native
intervention above, satisfying the original thresholds without a native
extension. On the 100,001-point activation grid, sigmoid matches completely;
tanh differs at one scalar-tail point (2.38e-7), while the vectorized grid
elements match. This is a diagnostic compatibility implementation, not yet
a production integration. Serialization, performance, extreme inputs,
GPU behavior, all-data training, real docking and Linux containers remain
unverified. The user subsequently authorized production integration with old-compatible
mode as the default.

## Saved evidence

The workspace `mothra-runtime-validation/` retains old/new prediction,
intermediate-layer, gradient, weight, fingerprint and primitive-kernel arrays;
`audit/mothra-*.log` retains execution logs. Strict failures are retained in
comparison-dedup.json. The initial investigation withheld the migration from PR #3. The subsequent
authorized integration is described in ../RUNTIME_VALIDATION.md.
