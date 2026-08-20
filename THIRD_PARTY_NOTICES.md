# Third-party test provenance

Selected algebra inputs in `tests/test_mathsteps_regressions.py` are adapted from the
`google/mathsteps` test suite (commit `9d370c30f2d9c2b3650883e942f97c8abe6805f8`,
retrieved August 20, 2026).

MathSteps is Copyright 2017 Evy Kassirer and is distributed under the Apache License,
Version 2.0. The adapted tests use StepSolver syntax and assert StepSolver's independently
implemented result, derivation, verification, and domain-constraint models. No MathSteps
runtime code is included.

Source: https://github.com/google/mathsteps

License: https://www.apache.org/licenses/LICENSE-2.0

## SymPy

Selected calculus and summation inputs in `tests/test_sympy_calculus_regressions.py`
are adapted from the SymPy test suite and documentation at commit
`81b519fabdbbc8e82db154dd271100ec7fb7ef32` (retrieved August 20, 2026).

SymPy is Copyright 2006-2023 SymPy Development Team and is distributed under
the BSD 3-Clause license. The adapted tests use StepSolver syntax and assert
StepSolver's independently implemented human derivations and verification records.
No SymPy test harness or implementation code is included.

Source: https://github.com/sympy/sympy

License: https://github.com/sympy/sympy/blob/master/LICENSE
