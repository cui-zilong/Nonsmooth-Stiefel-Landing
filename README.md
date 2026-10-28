# Nonsmooth Stiefel Landing Methods

MATLAB code for the paper:

> Z. Cui, T. Hu, B. Gao, and R. Gu, “Retraction-Free Nonsmooth Optimization on the Stiefel Manifold: Subgradient and Proximal Landing Methods,” [arXiv:2610.30617](https://arxiv.org/abs/2610.30617), 2026.

## Contents

```text
algorithms/          SubLanding, PGLanding, and IPGLanding
algorithms/baselines comparison methods
demos/               quick start and paper-reproduction scripts
utilities/           sparse-PCA data generators
data/                Colon data download instructions
licenses/            upstream notices and license texts
```

## Quick start

Open MATLAB in the repository root and run:

```matlab
run(fullfile('demos','demo_quick_start.m'))
```

The quick demo solves a small random sparse-PCA instance and checks
IPGLanding's stopping residuals. It is not used for the paper tables.

## Reproduce the paper

| Paper result | Script |
|---|---|
| Table 2 and Figure 1 | `demos/demo_Table2_Figure1.m` |
| Table 3: random SPCA | `demos/demo_Table3_Random.m` |
| Table 4: structured synthetic SPCA | `demos/demo_Table4_Synthetic.m` |
| Table 5: Colon data | `demos/demo_Table5_Colon.m` |

For example:

```matlab
run(fullfile('demos','demo_Table3_Random.m'))
```

The paper uses 10 timing repetitions. The variable `number_of_repetitions` near the beginning of each comparison script can be changed for a shorter test.

For Figure 1 alone, set `run_only_part_two = true` in `demo_Table2_Figure1.m`.
Output files are written beside the corresponding demo; Figure 1 is saved
in its `figures_SPCA_RieSub_SubLanding` subfolder.

The Colon CSV is not distributed. Follow `data/README.md` before running Table 5.

## Environment and conventions

The reported experiments used MATLAB R2025b on an Intel Core Ultra 9 275HX processor with 32 GB RAM. No external MATLAB package is required.

- Iteration counts are updates of the returned primal variable, excluding
  stopping checks and rejected attempts. PGLanding also limits subproblem
  attempts to 30000, including rejected steps.
- Spectral-norm calculations performed inside an algorithm are included in its CPU time.
- ManPG-Ada constructs its SSN index matrices once per solver call, within
  the timed region; they are not cached across repetitions.
- Comparison runs disable trajectory storage. SLPG keeps only the recent
  11 stepsizes needed by its update; optional outputs contain final summaries.
- Generated CSV, PDF, PNG, MAT, and log files are ignored by Git.

Tables 3--5 use a common RieSub warm start (250 updates), a ManPG-Ada
reference, and tolerance `1e-4` for the objective gap, iterate change, and
orthogonality residual. SOC and RADMM also check splitting residuals.
The distance to the reference is checked after termination. Failed runs
are not repeated; their timing fields are `NaN` in the CSV.

The comparison demos set IPGLanding to `t=2*t0`, `eta0=1.20`,
`lambda=5/12`, `dual_theta=1.95`, and backtracking factor `0.4`, where
`t0=1/norm(L,2)`. PGLanding uses `t=t0` and at most 20 SSN steps.
SLPG uses a BB cap of `2.5*t0` and no final polar postprocessing.

Times depend on the machine and MATLAB version.

## License and attribution

Unless otherwise stated, project-authored files are released under the MIT License in `LICENSE`.

`SLPG.m` is a MATLAB adaptation based on PySTOP 0.2.2 and is distributed under LGPL-3.0-or-later. It is excluded from the repository-level MIT License. ManPG-derived files remain MIT-licensed with attribution. See `THIRD_PARTY_NOTICE.md` and `licenses/`.

The Colon data are not distributed and are not covered by the repository license.
