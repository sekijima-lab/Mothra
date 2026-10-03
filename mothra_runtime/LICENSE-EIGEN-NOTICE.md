# Eigen attribution

`activations.py` adapts the float32 polynomial coefficients, range reduction
and CPU activation formulas in Eigen headers distributed with
`tensorflow-macos==2.9.1`:

- `Eigen/src/Core/MathFunctionsImpl.h` (`generic_fast_tanh_float`)
- `Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h` (`pexp_float`, `pldexp_generic`)
- `Eigen/src/Core/functors/UnaryFunctors.h` (`scalar_logistic_op`)

Copyright (C) 2007 Julien Pommier
Copyright (C) 2014 Pedro Gonnet
Copyright (C) 2009-2019 Gael Guennebaud

The adapted source file is provided under the Mozilla Public License, v. 2.0:
https://www.mozilla.org/MPL/2.0/ . The adaptation uses TensorFlow operations,
float64 intermediate multiply-adds and analytic Keras gradients; it does not
include or compile the Eigen library.
