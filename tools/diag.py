"""Diagnose: Läufe mit festen Seeds, Ursache jedes Fehlschlags protokollieren.

Aufruf aus dem Repo-Root (in der pixi-Umgebung):

    python tools/diag.py --seeds 0:40
    python tools/diag.py --seeds 0:40 --set '{"v_unseen": [2.6, 4.0, 3.5, 0.0, 0.0]}'
    python tools/diag.py --seeds 0:40 --oracle 2,3   # Analyse: echte Lage dieser Gates sofort
    python tools/diag.py --seeds 0:40 --seeded False  # Reset ohne Seed (wie scripts/sim.py)

--set überschreibt Attribute, die der Controller im __init__ setzt (vor ihrer Verwendung).
Pro Lauf eine JSON-Zeile, am Ende SUMMARY und CAUSES.
"""

from __future__ import annotations  # noqa: I001

import lsy_drone_racing  # noqa: F401  (must be imported before scipy)

import contextlib
import io
import json
from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config, load_controller

REPO = Path(__file__).resolve().parents[1]


def _ids(arg: object) -> list[int]:
    """'2,3' / (2, 3) / 2 -> [2, 3]."""
    if arg in ("", None):
        return []
    if isinstance(arg, (list, tuple)):
        return [int(x) for x in arg]
    return [int(x) for x in str(arg).split(",") if x.strip()]


def main(
    controller: str = "best_cont.py",
    config: str = "level2.toml",
    seeds: str = "0:40",
    out: str = "",
    set: str | dict = "",  # noqa: A002
    oracle: str = "",
    oracle_obst: str = "",
    quiet: bool = True,
    seeded: bool = True,
) -> None:
    """Run seeded episodes and print per-run diagnostics."""
    cfg = load_config(REPO / "config" / config)
    cfg.sim.render = False
    base = load_controller(REPO / "lsy_drone_racing/control" / controller)
    over = (json.loads(set) if isinstance(set, str) else dict(set)) if set else {}
    for k, v in over.items():
        if v in ("false", "true"):
            over[k] = v == "true"
        elif isinstance(v, list):
            over[k] = tuple(v)

    class Ctrl(base):  # Overrides: Zuweisungen im __init__ abfangen
        def __setattr__(self, k: str, v: object) -> None:
            object.__setattr__(self, k, over.get(k, v))

    og, oo = _ids(oracle), _ids(oracle_obst)
    env = JaxToNumpy(
        gymnasium.make(
            cfg.env.id,
            freq=cfg.env.freq,
            sim_config=cfg.sim,
            sensor_range=cfg.env.sensor_range,
            control_mode=cfg.env.control_mode,
            track=cfg.env.track,
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=cfg.env.seed,
        )
    )
    a, b = map(int, str(seeds).split(":"))
    quiet_ctx = (
        (lambda: contextlib.redirect_stdout(io.StringIO())) if quiet else contextlib.nullcontext
    )
    results = []
    for seed in range(a, b):
        obs, info = env.reset(seed=seed) if seeded else env.reset()  # ungeseedet wie sim.py
        u = env.unwrapped
        true_g = np.asarray(u.data.gates_pos)[0]
        true_q = np.asarray(u.data.gates_quat)[0]
        true_rot = R.from_quat(true_q).as_matrix()
        true_o = np.asarray(u.data.obstacles_pos)[0]
        nom_g = np.asarray(u.data.nominal_gates_pos)[0]  # level 3: the sampled track
        nom_o = np.asarray(u.data.nominal_obstacles_pos)[0]
        with quiet_ctx():
            ctrl = Ctrl(obs, info, cfg)
        err, known_at, i = [], {}, 0
        while True:
            act = ctrl.compute_control(obs, info)
            prev = obs
            obs, r, term, trunc, info = env.step(act)
            if og or oo:  # nur Analyse: echte Lage ausgewählter Objekte sofort bekannt
                obs = {k: np.array(v, copy=True) for k, v in obs.items()}
                for g in og:
                    obs["gates_pos"][g], obs["gates_quat"][g] = true_g[g], true_q[g]
                    obs["gates_visited"][g] = True
                for o in oo:
                    obs["obstacles_pos"][o] = true_o[o]
                    obs["obstacles_visited"][o] = True
            with quiet_ctx():
                fin = ctrl.step_callback(act, obs, r, term, trunc, info)
            ref = getattr(ctrl, "_last_ref", None)
            if ref is not None:
                err.append(float(np.linalg.norm(np.asarray(prev["pos"]) - ref)))
            for g, vis in enumerate(np.asarray(obs["gates_visited"])):
                if vis and g not in known_at:
                    known_at[g] = (i, float(np.linalg.norm(true_g[g] - np.asarray(obs["pos"]))))
            i += 1
            if term or trunc or fin:
                break
        with quiet_ctx():
            ctrl.episode_callback()
        p, v = np.asarray(prev["pos"]), np.asarray(prev["vel"])
        ngp = int(obs["n_gates_passed"])
        ok = ngp >= len(obs["gate_sequence"])
        d_obs = np.linalg.norm(true_o[:, :2] - p[:2], axis=1)
        d_gate = np.linalg.norm(true_g - p, axis=1)
        cause, extra = "ok", {}
        if not ok:
            if trunc:
                cause = "timeout"
            elif d_obs.min() < 0.15:
                cause = f"stange{int(d_obs.argmin())}"
            elif d_gate.min() < 0.5:
                cause = f"gate{int(d_gate.argmin())}-rahmen"
            elif p[2] < 0.1:
                cause = "boden"
            else:
                cause = "grenze/sonst"
            g = int(d_gate.argmin())
            ka = known_at.get(g)
            extra = {
                "g": g,
                "loc_cm": np.round(100 * true_rot[g].T @ (p - true_g[g])).astype(int).tolist(),
                "err_cm": int(100 * err[-1]) if err else None,
                "known_before_s": None if ka is None else round((i - ka[0]) / cfg.env.freq, 2),
            }
        res = {
            "seed": seed,
            "ok": bool(ok),
            "t": i / cfg.env.freq,
            "gates": ngp,
            "fallback": getattr(ctrl, "_plan_fallback", None),
            "relaxed": getattr(ctrl, "_plan_relaxed", None),
            "plan_s": round(float(getattr(ctrl, "_plan_time", 0.0)), 1),
            "solve_p95_ms": round(float(np.percentile(ctrl._solve_ms, 95)), 1)
            if getattr(ctrl, "_solve_ms", None)
            else None,
            "cause": cause,
            "pos": np.round(p, 2).tolist(),
            "speed": round(float(np.linalg.norm(v)), 2),
            **extra,
            "gate_off_cm": np.round(100 * np.linalg.norm(true_g - nom_g, axis=1))
            .astype(int)
            .tolist(),
            "obst_off_cm": np.round(100 * np.linalg.norm((true_o - nom_o)[:, :2], axis=1))
            .astype(int)
            .tolist(),
        }
        results.append(res)
        print(json.dumps(res), flush=True)
    t_ok = [r["t"] for r in results if r["ok"]]
    n_ok = len(t_ok)
    print(f"SUMMARY ok={n_ok}/{len(results)} t_avg={np.mean(t_ok) if t_ok else float('nan'):.3f}")
    causes: dict[str, int] = {}
    for r in results:
        if not r["ok"]:
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    print("CAUSES", causes)
    if out:
        Path(out).write_text(json.dumps(results))


if __name__ == "__main__":
    fire.Fire(main)
