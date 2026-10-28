# Third-party notice

## ManPG-derived MATLAB code (MIT)

`ManPGAda.m`, `SOC.m`, and `RieSub.m` are adapted from the official ManPG MATLAB package. The semismooth Newton implementation embedded in `PGLanding.m` also builds on its MIT-licensed solver structure.

Official source: <https://github.com/chenshixiang/ManPG>

Reference: S. Chen, S. Ma, A. M.-C. So, and T. Zhang, “Proximal Gradient Method for Nonsmooth Optimization over the Stiefel Manifold,” *SIAM Journal on Optimization*, 30(1), 210--239, 2020.

Modifications include interfaces, stopping rules, diagnostics, timing, and experiment integration.

The upstream README, including its author credit and MIT declaration, is
preserved in `licenses/ManPG-README.md`. The MIT permission text is in
`LICENSE`. The instrumented RieSub loop in `demos/demo_Table2_Figure1.m`
has the same source attribution.

## SLPG / PySTOP (LGPL-3.0-or-later)

`algorithms/baselines/SLPG.m` is a MATLAB adaptation based on PySTOP 0.2.2. PySTOP identifies its license as LGPL-3.0-or-later. This file is excluded from the repository-level MIT License and remains subject to LGPL-3.0-or-later.

Original package: <https://pypi.org/project/pystop/0.2.2/>

PySTOP authors: Nachuan Xiao, Lei Wang, Bin Gao, Xin Liu, and Ya-xiang Yuan.
The MATLAB adaptation adds SPCA-specific operations, stopping criteria,
an optional BB stepsize cap, and diagnostics (modified 2026-10-26).

The license designation follows PySTOP 0.2.2's package metadata.
The upstream GPLv3 text is preserved in `licenses/GPL-3.0.txt`;
the additional LGPLv3 terms are in `licenses/LGPL-3.0-or-later.txt`.

## Independent comparison implementations (MIT)

`RADMM.m` is an independent project implementation based on the update equations in the RADMM paper. It is not the authors' official implementation. The authors' repository is available at <https://github.com/JasonJiaxiangLi/RADMM>.

`LSALM.m` is an independent project implementation based on Algorithm 1 of L. Zhu, W. Ding, S. Liu, and A. M.-C. So, “Primal-Dual Methods for Nonsmooth Nonconvex Optimization with Orthogonality Constraints,” arXiv:2604.04130, 2026. It is not an official implementation supplied by those authors.

The original papers should be cited when numerical results using these comparison methods are reported.

## Data

The Colon data file is not redistributed. Attribution and download instructions are provided in `data/README.md`.
