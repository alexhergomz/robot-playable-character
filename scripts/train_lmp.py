#!/usr/bin/env python3
"""Stage 3: train the Local-Motion-Phase generator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpc.generator.lmp import (CoupledLMPGenerator, LMPGenerator,
                              UncertaintyWeighting, skate_loss)


class Norm:
    def __init__(self, a):
        self.mean = a.mean(0)
        self.std = a.std(0) + 1e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/robot_lmp")
    ap.add_argument("--out", default="runs/lmp")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--skate-weight", type=float, default=1.0,
                    help="ignored unless --fixed-weights; otherwise learned")
    ap.add_argument("--coupling", action="store_true",
                    help="learn pairwise phase coupling between channels; the "
                         "correction enters the predicted frequency, so a pair "
                         "that needs none simply learns a strength of zero")
    ap.add_argument("--skate", dest="no_skate", action="store_false", default=True,
                    help="include the stance-foot term.  Off by default: ablated "
                         "over 20 epochs it showed no benefit -- the arm that "
                         "never optimises it reached a *better* validation skate "
                         "value (83.68 vs 85.58 cm^2) and equal closed-loop "
                         "contact-point slip, because reconstructing pose and root "
                         "accurately already implies feet that do not slide.")
    ap.add_argument("--fixed-weights", action="store_true",
                    help="use the old uniform+typed weighting instead of learning it")
    ap.add_argument("--phase-target", action="store_true",
                    help="replace the authored gait one-hot with the measured "
                         "speed->phase target (exogenous, from the corpus)")
    ap.add_argument("--no-gait", action="store_true",
                    help="drop the authored gait one-hot from the input; gait is then "
                         "carried by the measured phase manifold instead")
    args = ap.parse_args()

    D = Path(args.data)
    man = json.load(open(D / "manifest.json"))
    load = lambda s, k: np.load(D / f"{s}_{k}.npy")
    Xtr, Ytr, Mtr, Ftr = (load("train", k) for k in ("X", "Y", "M", "F"))
    Xva, Yva, Mva, Fva = (load("val", k) for k in ("X", "Y", "M", "F"))

    J, C = man["n_joints"], man["phase_channels"]
    nf = man["n_feet"]

    # ---- assemble inputs: state | manifold | foot state ------------------
    # The gait one-hot is an authored label: three copies of a hand-typed
    # speed->gait heuristic existed in this repo with different constants, and
    # their disagreement stalled locomotion below 0.65 m/s.  Dropping it makes
    # the network read gait off the measured phase manifold (amplitude and
    # frequency), which is what physically distinguishes a walk from a run.
    goff = 0
    gsl = None
    for nm, k in man["in_blocks"]:
        if nm == "traj_gait":
            gsl = (goff, goff + k)
        goff += k
    keep_cols = np.ones(man["n_in"], bool)
    if args.no_gait:
        keep_cols[gsl[0]:gsl[1]] = False
        print(f"dropping traj_gait {gsl} -> {int(keep_cols.sum())} state inputs")

    # Exogenous phase target: what the corpus says a motion at this speed
    # measurably looks like.  It plays the role the gait one-hot played -- an
    # input that comes from intent rather than from the network's own previous
    # output -- but every number in it is a measurement.  Without some such
    # anchor the manifold feeds back on itself and amplitude collapses: the
    # gait-free model stalled below 0.8 m/s despite a better validation loss.
    ptab = None
    if args.phase_target:
        import json as _json
        from pathlib import Path as _P
        ptab = _json.loads((_P("rpc/control/phase_target.json")).read_text())
        ptab = {"speed": np.asarray(ptab["speed"]),
                "amplitude": np.asarray(ptab["amplitude"]),
                "frequency": np.asarray(ptab["frequency"])}

    def phase_target(X):
        soff = 0
        for nm, k in man["in_blocks"]:
            if nm == "root_lin_vel":
                break
            soff += k
        v = np.linalg.norm(X[:, soff:soff + 2], axis=1)
        a = np.stack([np.interp(v, ptab["speed"], ptab["amplitude"][:, i])
                      for i in range(C)], 1)
        f = np.stack([np.interp(v, ptab["speed"], ptab["frequency"][:, i])
                      for i in range(C)], 1)
        return np.concatenate([a, f], 1)

    def mk_in(X, M, F):
        parts = [X[:, keep_cols], M, F.reshape(len(F), -1)]
        if ptab is not None:
            parts.insert(1, phase_target(X))
        return np.concatenate(parts, 1).astype(np.float32)

    # ---- assemble targets: old blocks minus dphase, plus manifold + feet --
    off, sl = 0, {}
    for name, n in man["out_blocks"]:
        sl[name] = slice(off, off + n)
        off += n
    keep = [k for k, _ in man["out_blocks"] if k != "dphase"]

    def mk_out(Y, M, F, nxt_M, nxt_F):
        parts = [Y[:, sl[k]] for k in keep]
        parts += [nxt_M, nxt_F.reshape(len(nxt_F), -1)[:, :nf * 3]]
        return np.concatenate(parts, 1).astype(np.float32)

    # next-frame manifold / foot targets (last sample of each clip repeats)
    def shift(A):
        B = A.copy()
        B[:-1] = A[1:]
        return B

    Otr = mk_out(Ytr, Mtr, Ftr, shift(Mtr), shift(Ftr)[..., :3])
    Ova = mk_out(Yva, Mva, Fva, shift(Mva), shift(Fva)[..., :3])
    Itr, Iva = mk_in(Xtr, Mtr, Ftr), mk_in(Xva, Mva, Fva)

    off, osl = 0, {}
    for name in keep:
        n = dict(man["out_blocks"])[name]
        osl[name] = slice(off, off + n); off += n
    osl["manifold"] = slice(off, off + 4 * C); off += 4 * C
    osl["foot_pos"] = slice(off, off + nf * 3); off += nf * 3
    n_out = off
    n_in = Itr.shape[1]
    print(f"in {n_in}  out {n_out}  ({len(Itr)} train / {len(Iva)} val)")
    print(f"  blocks: " + ", ".join(f"{k}[{osl[k].stop-osl[k].start}]" for k in osl))

    # The amplitude-vs-speed curve the runtime uses to set phase amplitude from
    # commanded speed.  Measured from *this* model's own training data and
    # stored in its checkpoint, so a model trained on different clips cannot
    # silently inherit another dataset's curve.
    soff = 0
    for nm, k in man["in_blocks"]:
        if nm == "root_lin_vel":
            break
        soff += k
    vtr = np.linalg.norm(Xtr[:, soff:soff + 2], axis=1)
    edges = np.array([0, .15, .3, .45, .6, .8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    csp, camp = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (vtr >= lo) & (vtr < hi)
        if sel.sum() < 200:
            continue
        csp.append((lo + hi) / 2)
        camp.append(Mtr[sel, 2 * C:3 * C].mean(0))
    amp_curve = {"speed": np.asarray(csp), "amplitude": np.asarray(camp)}
    print(f"  amplitude curve: {len(csp)} bins, "
          f"{np.mean(camp[0]):.2f} at {csp[0]:.2f} m/s -> "
          f"{np.mean(camp[-1]):.2f} at {csp[-1]:.2f} m/s")

    xn, yn = Norm(Itr), Norm(Otr)
    if args.coupling:
        net = CoupledLMPGenerator(n_in, n_out, gate_dim=4 * C, hidden=args.hidden,
                                  n_experts=args.experts, dropout=args.dropout,
                                  n_phase_channels=C, manifold_slice=osl["manifold"])
    else:
        net = LMPGenerator(n_in, n_out, gate_dim=4 * C, hidden=args.hidden,
                           n_experts=args.experts, dropout=args.dropout)
    net.set_normalizers(xn, yn)

    Itr_t = torch.from_numpy((Itr - xn.mean) / xn.std).float()
    Otr_t = torch.from_numpy((Otr - yn.mean) / yn.std).float()
    Gtr_t = torch.from_numpy(Mtr).float()
    Ftr_t = torch.from_numpy(Ftr[..., :3]).float()
    Ctr_t = torch.from_numpy(Ftr[..., 3]).float()
    Iva_t = torch.from_numpy((Iva - xn.mean) / xn.std).float()
    Ova_t = torch.from_numpy((Ova - yn.mean) / yn.std).float()
    Gva_t = torch.from_numpy(Mva).float()
    Fva_t = torch.from_numpy(Fva[..., :3]).float()
    Cva_t = torch.from_numpy(Fva[..., 3]).float()
    ys = torch.from_numpy(yn.std).float()
    ym = torch.from_numpy(yn.mean).float()

    TERMS = tuple(osl.keys()) + (() if args.no_skate else ("skate",))
    uw = UncertaintyWeighting(TERMS)
    params = list(net.parameters()) + ([] if args.fixed_weights else list(uw.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = 1e9
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Itr_t))
        tot = sk = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            pred = net(Itr_t[b], Gtr_t[b], normalized=True)
            rec = torch.nn.functional.mse_loss(pred, Otr_t[b])
            skt = skate_loss(pred, osl, ys, ym, Ftr_t[b], Ctr_t[b])
            if args.fixed_weights:
                loss = rec + (0.0 if args.no_skate else args.skate_weight * skt)
            else:
                terms = {k: torch.nn.functional.mse_loss(pred[:, v], Otr_t[b][:, v])
                         for k, v in osl.items()}
                if not args.no_skate:
                    terms["skate"] = skt
                loss = uw(terms)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += rec.item() * len(b); sk += skt.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            vp = net(Iva_t, Gva_t, normalized=True)
            vr = torch.nn.functional.mse_loss(vp, Ova_t).item()
            vs = skate_loss(vp, osl, ys, ym, Fva_t, Cva_t).item()
        if vr + vs < best:
            best = vr + vs
            torch.save({"state": net.state_dict(), "args": vars(args), "manifest": man,
                        "out_slices": {k: [v.start, v.stop] for k, v in osl.items()},
                        "x_mean": xn.mean, "x_std": xn.std,
                        "y_mean": yn.mean, "y_std": yn.std,
                        "loss_weights": uw.weights(),
                        "amplitude_curve": {"speed": amp_curve["speed"].tolist(),
                                            "amplitude": amp_curve["amplitude"].tolist()},
                        "n_in": n_in, "n_out": n_out,
                        "no_gait": bool(args.no_gait),
                        "no_skate": bool(args.no_skate),
                        "coupling": bool(args.coupling),
                        "phase_target": bool(args.phase_target),
                        "keep_cols": keep_cols},
                       Path(args.out) / "generator.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            w = uw.weights()
            top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
            if args.coupling:
                K = net.coupling.strength()
                g = man.get("limb_groups"); pg = man.get("channels_per_group", 1)
                if g:
                    agg = K.reshape(len(g), pg, len(g), pg).abs().mean(dim=(1, 3))
                    print("        coupling K: " + "  ".join(
                        f"{g[i]}<-{g[j]}={agg[i, j]:.3f}"
                        for i in range(len(g)) for j in range(len(g)) if i != j
                        and agg[i, j] > 0.02), flush=True)
            print(f"  ep {ep:3d}  train rec {tot/len(perm):.4f} skate {sk/len(perm)*1e4:7.2f}"
                  f"   val rec {vr:.4f} skate {vs*1e4:7.2f} cm^2"
                  + ("" if args.fixed_weights else
                     "   w: " + " ".join(f"{k}={v:.2f}" for k, v in top)), flush=True)
    print(f"wrote {args.out}/generator.pt")


if __name__ == "__main__":
    main()
