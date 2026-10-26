# Superseded hand-rolled components

Moved out of the live path, not deleted, because this repo has no version control.

| file | superseded by | why |
|---|---|---|
| `wbc.py` | SONIC | QP inverse dynamics with hand-typed task weights. SONIC tracks at 0.132 rad and survives 7500+ steps; this plateaued at 110. |
| `mpc.py` | SONIC | Predictive-sampling MPC. Plateaued at ~110 steps invariant to optimiser, horizon (4-16), sigma (0.03-0.30), cost weights and 12x more samples. |
| `dynamics_fix.py` | GMR | ZMP/cart-table correction + FootPlanter, hand-tuned to the G1. GMR's task-space IK retargeting replaced it: 12/18 clips track 20 s vs 3/18. |

`estimator.py` (leg odometry) was NOT moved: it is what makes state estimation
honest on hardware (0.94 cm height, 11.3 cm/s velocity from encoders+IMU only)
and nothing supersedes it.

| `fit_coupling.py` | learned `PhaseCoupling` in `rpc/generator/lmp.py` | Fitted inter-limb Kuramoto coupling to the corpus as a preprocessing step. Failed: the periodic autoencoder encodes each window independently, so the phase track has no usable derivative (d(phi)/dt ~5x noisier than the predicted rate; R^2 ~ 0 and mostly negative). Coupling is now a learned parameter inside the generator instead. |
