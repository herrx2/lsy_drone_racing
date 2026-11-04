"""best controller: Zeitoptimale Bahn durch die Gates + MPCC (Fortschritt maximieren im Tunnel)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.linalg import solveh_banded
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from typing import Any

    from crazyflow import Sim
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


# Bordregler setzt die Beschleunigung ~1,46-fach um (im Sim identifiziert, aus my_controller_2.py)
THRUST_SCALE = 1.611 * 0.0393 / 0.0434
GRAVITY = np.array([0.0, 0.0, 9.81])

# ---- Drohnenmodell (aus my_controller_2.py, im Crazyflow-Sim identifiziert) ----------------
# Lage ohne Yaw über zwei Kippwinkel der Körper-z-Achse:
# z = [sin(p) cos(r), -sin(r), cos(r) cos(p)].
# Lageregler an Bord (im Sim identifiziert), pro Kippachse mit Winkelbeschleunigung al:
#   T = K1 * (tilt_cmd - tilt) + K2 * (rate_cmd - dtilt) - KD * al,  |T| <= AL_MAX
#   dal = (T - al) / TAU_AL
# Ohne Raten-Vorsteuerung dominiert die Dämpfung K2 (langsamer Pol K1 / K2 ~ 7 rad/s). Das MPC plant
# deshalb T direkt und schickt Lage + Rate so, dass die innere Schleife genau diese Bahn abfährt.
MPC_K1 = 800.0
MPC_K2 = 117.0
MPC_KD = 1.33
MPC_AL_MAX = 170.0  # max. Winkelbeschleunigung [rad/s^2]
MPC_TAU_AL = 0.051  # Rotor-Verzögerung [s]
MPC_CMD_TILT_MAX = 1.3  # Grenze fürs gesendete Lagekommando [rad]
MPC_MASS = 0.0434  # Nennmasse [kg] (zufällig +-0.005, fängt der Störgrößenbeobachter ab)
MPC_KF = 0.978  # umgesetzter / kommandierter Gesamtschub
MPC_TAU = 0.057  # Verzögerung des Schubs (1. Ordnung) [s]
MPC_F_MAX = 0.74  # Schub-Kommando [N]; Reserve für die Lagemomente regelt die Sättigungsgrenze
MPC_F_MIN = 0.15
MPC_F_ABS = 0.76  # oberste Schubgrenze auch für schwere Drohnen [N]; Motoren enden bei 0.8 N
MPC_TILT_MAX = 1.14  # max. Gesamtneigung [rad]
MPC_TILT_Z = 3000.0  # Strafe auf die Neigungsgrenze (linear, quadratisch x 100)
MPC_TILT_LEAD = 0.08  # Neigungsgrenze gilt für tilt + LEAD * Kipprate (Bremsweg) [s]
MPC_RESCUE_TILT = 1.3  # ab dieser Gesamtneigung MPC aussetzen und aufrichten [rad]
# Motoren: Schub f / 4 +- Momentanteil muss in [THR_MIN, THR_MAX] bleiben, sonst geht das Moment
# verloren und die Drohne kippt ungebremst weiter. T = 288 rad/s^2 entspricht 0.049 N pro Motor.
MPC_THR_MAX = 0.2  # [N pro Motor]
MPC_THR_MIN = 0.0214
MPC_THR_PER_T = 0.049 / 288.0
MPC_SAT_Z = 300.0  # Strafe auf die Sättigungsgrenze (linear, quadratisch x 10)
MPC_DF_MAX = 15.0  # max. Änderungsrate des Schub-Kommandos [N/s]
DRAG = np.array([0.0216, 0.0216, 0.0248]) / 0.0434  # lineare Luftreibung [1/s]
# Schubpfad im Mellinger: F = mass_thrust / pwm_max * 4 * thrust_max * m_ctrl * |a_cmd + g|
MELLINGER_F_PER_ACC = 132000.0 / 65535.0 * 0.8 * 0.0393

# Kollisionsgeometrie aus dem Simulator
DRONE_RADIUS = 0.10  # Kollisionskugel der Drohne (0.086 m) plus Reserve [m]
POLE_RADIUS = 0.015
GATE_BOXES = (  # (Mitte, halbe Kantenlängen) im Gate-System, x = Durchflugrichtung
    (np.array([0.0, 0.0, 0.28]), np.array([0.01, 0.36, 0.08])),  # oben
    (np.array([0.0, 0.0, -0.28]), np.array([0.01, 0.36, 0.08])),  # unten
    (np.array([0.0, -0.28, 0.0]), np.array([0.01, 0.08, 0.36])),  # links
    (np.array([0.0, 0.28, 0.0]), np.array([0.01, 0.08, 0.36])),  # rechts
    (np.array([0.0, 0.0, -0.86]), np.array([0.05, 0.05, 0.50])),  # Ständer
)
# Dieselben Teile als Kapseln (Achse von a nach b, Radius) für Planer und Bahnverformung
GATE_CAPSULES = (
    ((0.0, -0.28, 0.28), (0.0, 0.28, 0.28), 0.08),
    ((0.0, -0.28, -0.28), (0.0, 0.28, -0.28), 0.08),
    ((0.0, -0.28, -0.28), (0.0, -0.28, 0.28), 0.08),
    ((0.0, 0.28, -0.28), (0.0, 0.28, 0.28), 0.08),
    ((0.0, 0.0, -0.36), (0.0, 0.0, -1.36), 0.07),
)

NX, NU, NP = 16, 4, 23  # MPCC: Zustände, Eingänge, Parameter pro Stufe


def _tilt_axis(tilt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Körper-z-Achse (..., 3) aus den Kippwinkeln (..., 2) = [roll, pitch]."""
    r, p = tilt[..., 0], tilt[..., 1]
    return np.stack((np.sin(p) * np.cos(r), -np.sin(r), np.cos(r) * np.cos(p)), axis=-1)


def _tilt_state(
    quat: NDArray[np.float64], ang_vel: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Kippwinkel und ihre Raten aus Lage und Drehrate."""
    rot = Rotation.from_quat(quat)
    z = rot.apply([0.0, 0.0, 1.0])
    z_dot = np.cross(rot.apply(ang_vel), z)
    hor2 = max(z[0] ** 2 + z[2] ** 2, 1e-6)
    tilt = np.array([np.arcsin(np.clip(-z[1], -1.0, 1.0)), np.arctan2(z[0], z[2])])
    rate = np.array([-z_dot[1] / np.sqrt(hor2), (z[2] * z_dot[0] - z[0] * z_dot[2]) / hor2])
    return tilt, rate


def _capsules(
    gate_pos: NDArray[np.float64], gate_rot: NDArray[np.float64]
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], float]]:
    """Rahmen und Ständer eines Gates als Kapseln in Weltkoordinaten: (a, b, Radius)."""
    return [
        (gate_pos + gate_rot @ np.array(a), gate_pos + gate_rot @ np.array(b), r)
        for a, b, r in GATE_CAPSULES
    ]


def _capsule_dist(
    pts: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Abstand der Punkte (n, 3) zur Strecke a-b."""
    d = b - a
    t = np.clip((pts - a) @ d / (d @ d), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * d), axis=1)


# ======================================================================================
# Zeitoptimale Punktmassen-Bahn
# ======================================================================================
def _time_optimal(
    start: NDArray[np.float64],
    start_v: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_rot: NDArray[np.float64],
    seq: NDArray[np.integer],
    dirs: NDArray[np.float64],
    poles: NDArray[np.float64],
    guess: list[NDArray[np.float64]],
    unseen: list[float | None],
    cfg: dict[str, Any],
) -> dict[str, NDArray] | None:
    """Zeitoptimale Bahn durch die Gates in der Reihenfolge seq.

    Punktmasse mit Schubbeschleunigung f als Zustand (Ruck als Eingang), |f| <= t_max,
    Neigung <= tilt_max, lineare Luftreibung. Pro Gate ein Segment mit eigener Dauer; das
    Segmentende liegt in der Gate-Öffnung, die Flugrichtung höchstens theta vom Gate-Normalen-
    vektor entfernt. guess: Startlösung (Punkte pro Segment, je n+1). unseen[i]: Tempolimit in
    den letzten unseen_len Metern vor Gate i, weil es dort erst erkannt wird. Start in start mit
    Geschwindigkeit start_v (Zustand am Ende des senkrechten Abhebens), Schub dort frei.
    """
    n = cfg["n"]
    nseg = len(seq)
    opti = ca.Opti()
    ts = opti.variable(nseg)
    xs = [opti.variable(9, n + 1) for _ in range(nseg)]
    us = [opti.variable(3, n) for _ in range(nseg)]
    cost = ca.sum1(ts)
    caps = [(g, c) for g in range(len(gates_pos)) for c in _capsules(gates_pos[g], gates_rot[g])]
    gvec = GRAVITY
    cos_t2 = np.cos(cfg["theta"]) ** 2
    cos_tilt2 = np.cos(cfg["tilt_max"]) ** 2
    for i in range(nseg):
        h = ts[i] / n
        opti.subject_to(opti.bounded(0.15, ts[i], 4.0))
        gp = guess[i]
        seg_len = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))]
        for k in range(n):
            p, v, f = xs[i][0:3, k], xs[i][3:6, k], xs[i][6:9, k]
            u = us[i][:, k]
            a = f - gvec - DRAG * v
            opti.subject_to(xs[i][0:3, k + 1] == p + v * h + a * h**2 / 2 + u * h**3 / 6)
            opti.subject_to(xs[i][3:6, k + 1] == v + a * h + u * h**2 / 2)
            opti.subject_to(xs[i][6:9, k + 1] == f + u * h)
            opti.subject_to(opti.bounded(-cfg["jerk"], u, cfg["jerk"]))
            cost += 1e-6 * ca.sumsqr(u) * h
        for k in range(n + 1):
            p, v, f = xs[i][0:3, k], xs[i][3:6, k], xs[i][6:9, k]
            opti.subject_to(ca.sumsqr(f) <= cfg["t_max"] ** 2)
            opti.subject_to(f[2] >= 0.0)
            opti.subject_to(f[2] ** 2 >= cos_tilt2 * ca.sumsqr(f))
            if i == 0 and k <= n // 3:  # Start: mindestens mit Steigung climb vom Boden weg
                dxy2 = ca.sumsqr(p[0:2] - start[0:2])
                opti.subject_to(p[2] >= start[2])
                opti.subject_to((p[2] - start[2]) ** 2 >= cfg["climb"] ** 2 * dxy2)
            else:
                opti.subject_to(p[2] >= cfg["z_min"])
            for o in poles:
                if np.min(np.linalg.norm(gp[:, :2] - o[:2], axis=1)) < 0.9:
                    opti.subject_to(ca.sumsqr(p[0:2] - o[0:2]) >= cfg["r_pole"] ** 2)
            for g, (a, b, r) in caps:
                if np.min(np.linalg.norm(gp - gates_pos[g], axis=1)) < 0.9:
                    d = b - a
                    t = ca.fmin(ca.fmax(ca.dot(p - a, d) / float(d @ d), 0.0), 1.0)
                    opti.subject_to(ca.sumsqr(p - (a + t * d)) >= (r + cfg["r_frame"]) ** 2)
            if unseen[i] is not None and seg_len[-1] - seg_len[k] < cfg["unseen_len"]:
                opti.subject_to(ca.sumsqr(v) <= unseen[i] ** 2)
            if unseen[i] is not None and seg_len[-1] - seg_len[k] < cfg["approach_len"]:
                # Einflug-Kegel: fast waagrecht und nicht zu schräg. Verschiebt sich das Gate
                # längs seiner Achse, ändert sich die Bahn dann kaum
                rot = gates_rot[seq[i]]
                v_n = ca.dot(v, rot[:, 0] * dirs[i])
                opti.subject_to(
                    opti.bounded(
                        -np.tan(cfg["approach_lat"][i]) * v_n,
                        ca.dot(v, rot[:, 1]),
                        np.tan(cfg["approach_lat"][i]) * v_n,
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -np.tan(cfg["approach_vert"]) * v_n,
                        ca.dot(v, rot[:, 2]),
                        np.tan(cfg["approach_vert"]) * v_n,
                    )
                )
        if i > 0:
            opti.subject_to(xs[i][:, 0] == xs[i - 1][:, n])
        # Segmentende = Durchflug durch Gate seq[i]
        rot = gates_rot[seq[i]]
        nrm, gy, gz = rot[:, 0] * dirs[i], rot[:, 1], rot[:, 2]
        rel = xs[i][0:3, n] - (gates_pos[seq[i]] + cfg["gate_z"] * gz)
        ve = xs[i][3:6, n]
        opti.subject_to(ca.dot(rel, nrm) == 0.0)
        opti.subject_to(
            opti.bounded(cfg["bias"][i] - cfg["lat"], ca.dot(rel, gy), cfg["bias"][i] + cfg["lat"])
        )
        opti.subject_to(opti.bounded(-0.03, ca.dot(rel, gz), 0.03))
        opti.subject_to(ca.dot(ve, nrm) >= 0.0)
        opti.subject_to(ca.dot(ve, nrm) ** 2 >= cos_t2 * ca.sumsqr(ve))
    opti.subject_to(xs[0][0:3, 0] == start)
    opti.subject_to(xs[0][3:6, 0] == start_v)
    opti.minimize(cost)

    for i in range(nseg):
        gp = guess[i]
        length = float(np.sum(np.linalg.norm(np.diff(gp, axis=0), axis=1)))
        t_i = max(length / 2.5, 0.3)
        opti.set_initial(xs[i][0:3, :], gp.T)
        opti.set_initial(xs[i][3:6, :], (np.gradient(gp, axis=0) / (t_i / n)).T)
        opti.set_initial(xs[i][6:9, :], np.tile(gvec, (n + 1, 1)).T)
        opti.set_initial(ts[i], t_i)
    opti.solver(
        "ipopt",
        {"print_time": False, "expand": True},
        {"max_iter": 1500, "print_level": 0, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4},
    )
    try:
        sol = opti.solve()
    except RuntimeError:
        return None
    pts, vel, tt, cross = [], [], [], []
    t0 = 0.0
    for i in range(nseg):
        x = np.asarray(sol.value(xs[i]))
        t_i = float(sol.value(ts[i]))
        k0 = 0 if i == 0 else 1
        pts.append(x[0:3, k0:].T)
        vel.append(x[3:6, k0:].T)
        tt.append(t0 + np.linspace(0.0, t_i, n + 1)[k0:])
        t0 += t_i
        cross.append(sum(len(p) for p in pts) - 1)
    return {
        "P": np.concatenate(pts),
        "V": np.concatenate(vel),
        "t": np.concatenate(tt),
        "cross": np.array(cross),
    }


_PLAN_CACHE = {}


# ======================================================================================
# MPCC
# ======================================================================================
def _mpcc_model(name: str) -> AcadosModel:
    """Definiere das MPCC-Modell.

    x = [p, tilt, v, dtilt, al, f, f_cmd, s, ds], u = [T (2), Schub-Kommando-Rate, dds].

    Parameter pro Stufe: [Stör-Beschleunigung (3), Pfadpunkt (3), Tangente (3), s_lin, Tunnelradius,
    Tempolimit, Schubfaktor (geschätzt, fängt die zufällige Masse ab)].
    Der Pfad wird um s_lin linearisiert: p_ref(s) ~ Pfadpunkt + Tangente * (s - s_lin).
    """
    p, v = ca.SX.sym("p", 3), ca.SX.sym("v", 3)
    tilt, dtilt = ca.SX.sym("tilt", 2), ca.SX.sym("dtilt", 2)
    al = ca.SX.sym("al", 2)
    f, f_cmd = ca.SX.sym("f"), ca.SX.sym("f_cmd")
    s, ds = ca.SX.sym("s"), ca.SX.sym("ds")
    u = ca.SX.sym("u", NU)
    par = ca.SX.sym("par", NP)
    dist, p_lin, t_lin, s_lin, r_tun, v_lim = par[0:3], par[3:6], par[6:9], par[9], par[10], par[11]
    kap = par[12]
    # Gate-Öffnung (MPCC++): Mitte, Gate-y- und Gate-z-Achse, halbe Breite (groß = inaktiv)
    g_c, g_y, g_z, g_w = par[13:16], par[16:19], par[19:22], par[22]

    cr, sr = ca.cos(tilt[0]), ca.sin(tilt[0])
    cp, sp = ca.cos(tilt[1]), ca.sin(tilt[1])
    z_body = ca.vertcat(sp * cr, -sr, cr * cp)
    acc = z_body * MPC_KF * kap * f / MPC_MASS - GRAVITY - DRAG * v + dist

    # Fehler zum (linearisierten) Pfad: Lag = längs, Contour = quer
    e = p - (p_lin + t_lin * (s - s_lin))
    e_lag = ca.dot(t_lin, e)
    e_con = e - e_lag * t_lin

    model = AcadosModel()
    model.name = name
    model.x = ca.vertcat(p, tilt, v, dtilt, al, f, f_cmd, s, ds)
    model.u = u
    model.p = par
    model.f_expl_expr = ca.vertcat(
        v, dtilt, acc, al, (u[:2] - al) / MPC_TAU_AL, (f_cmd - f) / MPC_TAU, u[2], ds, u[3]
    )
    model.cost_y_expr = ca.vertcat(e_con, e_lag, ds, dtilt, u)
    model.cost_y_expr_e = ca.vertcat(e_con, e_lag)
    # im Tunnel bleiben; echte Geschwindigkeit längs des Pfads begrenzen
    model.con_h_expr = ca.vertcat(
        ca.sqrt(ca.dot(e_con, e_con) + 1e-6) - r_tun,
        ca.dot(t_lin, v) - v_lim,
        np.cos(MPC_TILT_MAX)
        - ca.cos(tilt[0] + MPC_TILT_LEAD * dtilt[0]) * ca.cos(tilt[1] + MPC_TILT_LEAD * dtilt[1]),
        ca.sqrt(ca.dot(p - g_c, g_y) ** 2 + 1e-6) - g_w,
        ca.sqrt(ca.dot(p - g_c, g_z) ** 2 + 1e-6) - g_w,
    )
    model.con_h_expr_e = model.con_h_expr
    # Reserve der Motoren (normiert auf 0.05 N): Moment und Gesamtschub teilen sich den Bereich
    d_m = MPC_THR_PER_T * (ca.sqrt(u[0] ** 2 + 1.0) + ca.sqrt(u[1] ** 2 + 1.0))
    model.con_h_expr = ca.vertcat(
        model.con_h_expr_e,
        (f_cmd / 4.0 + d_m - MPC_THR_MAX) / 0.05,
        (MPC_THR_MIN - f_cmd / 4.0 + d_m) / 0.05,
    )
    return model


def _mpcc_solver(
    n: int,
    dt: float,
    w: NDArray[np.float64],
    w_e: NDArray[np.float64],
    ds_ref: float,
    dds_max: float,
    tun_z: float,
    tun_Z: float,
    z_min: float,
) -> AcadosOcpSolver:
    """MPCC bauen (oder aus dem Temp-Ordner laden). Schreibt nie ins Repo."""
    key = hashlib.sha1(
        repr(
            (
                n,
                dt,
                list(map(float, w)),
                list(map(float, w_e)),
                ds_ref,
                dds_max,
                tun_z,
                tun_Z,
                z_min,
                MPC_AL_MAX,
                MPC_TAU_AL,
                MPC_MASS,
                MPC_KF,
                MPC_TAU,
                MPC_F_MAX,
                MPC_F_MIN,
                MPC_TILT_MAX,
                MPC_TILT_Z,
                MPC_TILT_LEAD,
                MPC_DF_MAX,
                DRAG.tolist(),
                MPC_SAT_Z,
                "mpcc_gate_v5",
            )
        ).encode()
    ).hexdigest()[:10]
    name = f"lsy_my_mpcc_{key}"
    root = os.path.join(tempfile.gettempdir(), name)
    json_file = os.path.join(root, "ocp.json")
    lib = os.path.join(root, "c_generated_code", f"libacados_ocp_solver_{name}.so")
    if os.path.exists(json_file) and os.path.exists(lib):
        return AcadosOcpSolver(
            None, json_file=json_file, build=False, generate=False, verbose=False
        )

    ocp = AcadosOcp()
    ocp.model = _mpcc_model(name)
    ocp.code_export_directory = os.path.join(root, "c_generated_code")
    ocp.solver_options.N_horizon = n
    ocp.solver_options.tf = n * dt
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(w)
    ocp.cost.W_e = np.diag(w_e)
    # Fortschritt: ds soll möglichst groß sein -> (ds - ds_ref)^2 mit großem ds_ref
    ocp.cost.yref = np.r_[np.zeros(4), ds_ref, np.zeros(2 + NU)]
    ocp.cost.yref_e = np.zeros(4)
    ocp.parameter_values = np.zeros(NP)
    ocp.constraints.x0 = np.zeros(NX)
    ocp.constraints.lbu = np.array([-MPC_AL_MAX, -MPC_AL_MAX, -MPC_DF_MAX, -dds_max])
    ocp.constraints.ubu = np.array([MPC_AL_MAX, MPC_AL_MAX, MPC_DF_MAX, dds_max])
    ocp.constraints.idxbu = np.arange(NU)
    # Zustände: Höhe (weich), Kippwinkel (weich), Schub-Kommando, Bahngeschwindigkeit (weich)
    ocp.constraints.idxbx = np.array([2, 3, 4, 13, 15])
    ocp.constraints.lbx = np.array([z_min, -MPC_TILT_MAX, -MPC_TILT_MAX, MPC_F_MIN, 0.0])
    ocp.constraints.ubx = np.array([2.4, MPC_TILT_MAX, MPC_TILT_MAX, MPC_F_MAX, 10.0])
    # Gesamtneigung zusätzlich zu den Einzelachsen: 0.9 rad pro Achse wären zusammen 68 Grad
    ocp.constraints.idxsbx = np.array([0, 1, 2, 4])
    ocp.constraints.lh = np.full(7, -100.0)
    ocp.constraints.uh = np.zeros(7)
    ocp.constraints.idxsh = np.arange(7)
    # Strafkosten der weichen Grenzen:
    # [Boden, tilt, tilt, ds, Tunnel, Tempo, Neigung, Gate (2), Sättigung (2)]
    z_e = np.array([2000.0, 100.0, 100.0, 100.0, tun_z, 1000.0, MPC_TILT_Z, tun_z, tun_z])
    z2_e = np.array([2e5, 1000.0, 1000.0, 1000.0, tun_Z, 1e4, 100 * MPC_TILT_Z, tun_Z, tun_Z])
    ocp.cost.zl = ocp.cost.zu = np.r_[z_e, MPC_SAT_Z, MPC_SAT_Z]
    ocp.cost.Zl = ocp.cost.Zu = np.r_[z2_e, 10 * MPC_SAT_Z, 10 * MPC_SAT_Z]
    # Gleiche Grenzen auch im letzten Knoten
    ocp.constraints.idxbx_e = ocp.constraints.idxbx.copy()
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
    ocp.constraints.idxsbx_e = ocp.constraints.idxsbx.copy()
    ocp.constraints.lh_e = np.full(5, -100.0)
    ocp.constraints.uh_e = np.zeros(5)
    ocp.constraints.idxsh_e = np.arange(5)
    ocp.cost.zl_e = ocp.cost.zu_e = z_e
    ocp.cost.Zl_e = ocp.cost.Zu_e = z2_e
    opts = ocp.solver_options
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.hessian_approx = "GAUSS_NEWTON"
    opts.integrator_type = "ERK"
    opts.sim_method_num_stages = 4
    opts.sim_method_num_steps = 1
    opts.nlp_solver_type = "SQP_RTI"
    opts.qp_solver_iter_max = 50
    opts.qp_solver_warm_start = 1
    os.makedirs(root, exist_ok=True)
    return AcadosOcpSolver(ocp, json_file=json_file, verbose=False)


class PathGeo:
    """Kubischer Spline durch die Wegpunkte, nach der Bogenlänge s parametriert."""

    def __init__(self, waypoints: NDArray[np.float64]) -> None:
        """Initialisiere den nach Bogenlänge parametrisierten Pfad."""
        waypoints = np.asarray(waypoints, dtype=float)
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        if len(seg) == 0 or np.any(seg < 1e-6):
            raise ValueError("Path needs distinct consecutive waypoints")
        knots = np.r_[0.0, np.cumsum(seg)]
        raw = CubicSpline(knots, waypoints)
        # 2.5-mm-Raster + alle Knoten, dann nach der echten Bogenlänge umparametrieren
        u = np.unique(
            np.r_[np.linspace(0, knots[-1], max(1001, int(np.ceil(knots[-1] / 0.0025)) + 1)), knots]
        )
        speed = np.linalg.norm(raw.derivative()(u), axis=1)
        if np.min(speed) < 1e-7:
            raise ValueError("Path has a stationary tangent")
        arc = np.r_[0.0, cumulative_trapezoid(speed, u)]
        self.spl = CubicSpline(arc, raw(u))
        self.d1 = self.spl.derivative()
        self.length = float(arc[-1])
        self.knot_s = np.interp(knots, u, arc)  # Bogenlänge der Wegpunkte
        self.ss = np.r_[np.arange(0.0, self.length, 0.01), self.length]
        self.pts = self.spl(self.ss)

    def __call__(self, s: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Werte den Pfad an einer oder mehreren Bogenlängen aus."""
        return self.spl(np.clip(s, 0.0, self.length))

    def tangent(self, s: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Berechne die normierte Tangente entlang des Pfads."""
        d = self.d1(np.clip(s, 0.0, self.length))
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)

    def project(
        self, p: NDArray[np.float64], s_guess: float, back: float = 0.3, ahead: float = 0.8
    ) -> float:
        """Projiziere einen Punkt im lokalen Suchbereich auf den Pfad."""
        lo = float(np.clip(s_guess - back, 0, self.length))
        hi = float(np.clip(s_guess + ahead, lo, self.length))
        if hi - lo < 1e-10:
            return lo
        i0 = int(np.searchsorted(self.ss, lo))
        i1 = int(np.searchsorted(self.ss, hi, side="right"))
        grid = np.r_[lo, self.ss[i0:i1], hi]
        sq = np.sum((self(grid) - p) ** 2, axis=1)
        k = int(np.argmin(sq))
        left, right = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        if right - left < 1e-10:
            return float(grid[k])
        res = minimize_scalar(
            lambda x: float(np.sum((self(x) - p) ** 2)),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-5},
        )
        return float(res.x if res.fun < sq[k] else grid[k])


class MyController(Controller):
    """Zeitoptimale Bahn durch die Gates + MPCC (Fortschritt maximieren im Tunnel)."""

    def __init__(self, obs: dict[str, Any], info: dict[str, Any], config: ConfigDict) -> None:
        """Initialisiere Planer, MPCC und den Zustand des Controllers."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # ---- Zeitoptimale Bahn ------------------------------------------------------------
        self.plan_n = 30  # Knoten pro Segment
        self.plan_t_max = 15.1  # Schubgrenze [m/s^2]
        self.plan_tilt = 0.982  # max. Neigung [rad]
        self.plan_jerk = 150.0  # max. Ruck pro Achse [m/s^3]
        self.plan_theta = 1.077  # max. Winkel Flugrichtung <-> Gate-Normale beim Durchflug [rad]
        self.plan_lat = 0.03  # Durchflugpunkt: max. seitlicher Abstand zur Gate-Mitte [m]
        self.plan_r_pole = 0.165  # Abstand zur Stangenmitte [m]
        self.plan_r_frame = 0.10  # Abstand zu den Rahmen-Kapseln [m]
        self.plan_z_min = 0.15  # Mindesthöhe nach dem Start [m]
        self.plan_climb = 0.5  # Start: Höhe >= climb * horizontaler Abstand
        # Tempo vor noch unbekannten Gates [m/s], pro Durchflug (0 = kein Limit)
        self.v_unseen = (2.57, 4.0, 3.5, 3.24, 0.0)
        self.unseen_len = 0.7  # ... auf den letzten Metern vor dem Gate (Sensorreichweite)
        self.approach_len = 0.4  # Einflug-Kegel vor unbekannten Gates: Länge [m]
        # ... max. Winkel zur Gate-Achse seitlich [rad], pro Durchflug
        self.approach_lat = (0.5, 0.5, 0.5, 0.5, 0.5)
        self.approach_vert = 0.1  # ... und in der Höhe [rad]
        # Durchflugpunkt seitlich versetzt (Gate-y) [m]: Gate 0 wird schräg angeflogen, die
        # Korrektur hinkt dort immer in Richtung des linken Balkens hinterher
        self.gate_bias = (0.078, 0.0, 0.0, 0.0, 0.0)
        self.plan_via = {}  # Hilfspunkte der Startlösung: {Durchflug-Index: (x, y, z)}
        self.gate_z = 0.05  # Durchflughöhe relativ zur Gate-Mitte [m]
        self.run_out = 1.0  # Auslauf hinter dem letzten Gate [m]
        # Startlösungen für den Planer: (Hilfspunkt-Abstand am Gate, Stangen-, Rahmenabstand)
        self.plan_starts = ((0.3, 0.25, 0.5), (0.5, 0.3, 0.5), (0.2, 0.35, 0.6))

        # ---- Bahnverformung ---------------------------------------------------------------
        self.def_ds = 0.05  # Raster der Verformung [m]
        self.gate_rigid = 0.10  # Bereich um den Durchflug, der sich starr mit dem Gate bewegt [m]
        self.gate_tol = 0.04  # Gate-Verschiebung bis hierhin wird nicht nachgeführt [m]
        self.push_pole = 0.165  # Bahn mindestens so weit von Stangen [m]
        self.push_frame = 0.19  # ... und von den Rahmen-Kapselachsen (inkl. Rahmenbreite) [m]

        # ---- MPCC ---------------------------------------------------------------------------
        self.mpc_n = 40  # Horizont (40 x 20 ms = 0.8 s)
        self.v_max = 7.0  # max. Bahngeschwindigkeit [m/s]
        self.topt_vf = 0.9  # Tempolimit = topt_vf * zeitoptimales Profil + 0.3 m/s
        self.tunnel_max = 0.15  # max. Tunnelradius im freien Raum [m]
        self.tunnel_min = 0.03
        self.margin = 0.02  # zusätzlicher Abstand zu Rahmen / Stangen [m]
        self.gate_w = 0.106  # Gate-Öffnung im MPCC: halbe Breite für die Drohnenmitte [m]
        self.gate_win = 0.15  # ... gilt in diesem Bereich um den Durchflug (Bogenlänge) [m]
        self.gate_tun = 0.10  # Tunnelradius dort mindestens [m]
        self.q_con = 2250.0  # Gewicht Querfehler
        self.q_con_z = 3000.0  # ... davon in der Höhe (Drohne sackt in engen Kurven durch)
        self.q_lag = 1000.0  # Gewicht Längsfehler (hält s an der Drohne)
        self.q_prog = 40.0  # Gewicht Fortschritt
        self.q_rate = 0.5  # Gewicht Kipprate
        self.q_al = 0.0005  # Gewicht Winkelbeschleunigung
        self.ds_ref = 10.0  # Startwert des Fortschrittsterms (wird pro Stufe überschrieben)
        self.ds_add = 0.29  # Fortschrittsterm: Soll-Tempo = Tempolimit + ds_add [m/s]
        self.dds_max = 30.0  # max. Bahnbeschleunigung im Modell [m/s^2]
        self.k_dist = 0.15  # Störgrößenbeobachter (pro Schritt)
        self.k_kap = 0.1  # Schätzung Schubfaktor (pro Schritt)
        self.dist_par = 0.3  # Anteil des Fehlers in Schubrichtung, der additiv bleibt
        self.dist_max = np.array([1.5, 1.5, 2.0])
        self.takeoff_height = 0.004  # darunter senkrechter Start (Mellinger, fester Schub) [m]
        self.takeoff_f = 0.78  # Schub beim Start [N]
        self.takeoff_dz = 0.04  # Zustand bei der Übergabe an den MPCC: Höhe über Start [m]
        self.takeoff_vz = 0.6  # ... und Steiggeschwindigkeit [m/s] (Startpunkt des Plans)
        self.tun_z = 2000.0  # Strafe fürs Verlassen des Tunnels pro m (linear / quadratisch)
        self.tun_Z = 1e5
        self.z_min = 0.08  # Boden (weiche Grenze im MPCC) [m]
        self.rebuild_iter = 2  # zusätzliche SQP-Iterationen nach einer Bahnänderung

        self._solve_ms = []
        self._speed_sum = 0.0
        self._speed_samples = 0

        w = np.r_[
            self.q_con,
            self.q_con,
            self.q_con_z,
            self.q_lag,
            self.q_prog,
            [self.q_rate] * 2,
            [self.q_al] * 2,
            0.1,
            0.01,
        ]
        w_e = np.r_[self.q_con, self.q_con, self.q_con_z, self.q_lag]
        self._mpc = _mpcc_solver(
            self.mpc_n,
            1.0 / self._freq,
            w,
            w_e,
            self.ds_ref,
            self.dds_max,
            self.tun_z,
            self.tun_Z,
            self.z_min,
        )
        self._mpc.reset()

        # ---- Zustand ------------------------------------------------------------------------
        self._tick = 0
        self._last_obs = obs
        self._seq = np.asarray(obs["gate_sequence"], dtype=int)
        self._dirs = np.asarray(obs["gate_sequence_direction"], dtype=float)
        self._start = np.asarray(obs["pos"], dtype=float).copy()
        # Der Plan wird im Cache für alle Läufe geteilt: deshalb ab der nominalen Startposition
        # planen (sonst bestimmt die zufällige Startposition des ersten Laufs den Plan für alle)
        self.plan_start_fix = None  # fester Startpunkt des Plans (sonst nominal aus der Config)
        try:
            self._plan_start = np.asarray(config.env.track.drones[0]["pos"], dtype=float)
            if self.plan_start_fix is not None:
                self._plan_start = np.asarray(self.plan_start_fix, dtype=float)
        except (AttributeError, IndexError, KeyError, TypeError):
            self._plan_start = self._start.copy()
        self._airborne = False
        self._dist = np.zeros(3)
        self._kap = 1.0
        self._f_est = 0.0
        self._f_cmd = MPC_F_MIN
        self._al = np.zeros(2)
        self._v_pred = None
        self._mpc_warm = False
        self._s = 0.0
        self._ds_pred = np.zeros(self.mpc_n + 1)
        self._mpc_fail = 0

        # ---- Planen: zeitoptimale Bahn (einmal pro Strecke), dann Pfad -----------------------
        self._nominal(obs)
        self._known = np.r_[obs["gates_visited"], obs["obstacles_visited"]].astype(bool)
        self._build_path(obs)

    # ==================================================================================
    # Startlösung: Wegpunkte durch die Gates, um Stangen und Rahmen herum
    # ==================================================================================
    def _guess_waypoints(
        self,
        obs: dict[str, Any],
        start: NDArray[np.float64],
        d: float = 0.3,
        r_pole: float = 0.25,
        r_frame: float = 0.5,
    ) -> NDArray[np.float64]:
        """Erzeuge Hilfspunkte für die initiale Bahn durch die Gates."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        waypoints, fixed = [start], [True]
        for i, (g, sgn) in enumerate(zip(self._seq, self._dirs)):
            normal = gates_rot[g][:, 0] * sgn
            if i in self.plan_via:
                waypoints.append(np.array(self.plan_via[i], dtype=float))
                fixed.append(True)
            waypoints += [gates_pos[g] - d * normal, gates_pos[g], gates_pos[g] + d * normal]
            fixed += [False, True, False]
        waypoints.append(waypoints[-1] + self.run_out * normal)
        fixed.append(False)
        return self._avoid_collisions(np.array(waypoints), np.array(fixed), obs, r_pole, r_frame)

    def _avoid_collisions(
        self,
        waypoints: NDArray[np.float64],
        fixed: NDArray[np.bool_],
        obs: dict[str, Any],
        r_pole: float,
        r_frame: float,
    ) -> NDArray[np.float64]:
        """Passe freie Hilfspunkte zur Umgehung von Hindernissen an."""
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        for _ in range(10):
            seg_len = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            if np.any(seg_len < 1e-3):
                break
            s = np.r_[0.0, np.cumsum(seg_len)]
            ss = np.linspace(0.0, s[-1], max(100, int(s[-1] / 0.03)))
            pts = CubicSpline(s, waypoints)(ss)
            worst, i_worst, via = 0.0, None, None
            # 1) Stangen: Abstand in x-y (Stangen gehen vom Boden bis 1.55 m)
            dist = np.linalg.norm(pts[:, None, :2] - poles[None, :, :2], axis=2)
            if dist.size:
                i, j = np.unravel_index(dist.argmin(), dist.shape)
                if r_pole - dist[i, j] > worst:
                    away = (pts[i, :2] - poles[j, :2]) / max(dist[i, j], 1e-6)
                    worst, i_worst = r_pole - dist[i, j], i
                    via = np.r_[poles[j, :2] + r_pole * away, pts[i, 2]]
            # 2) Gate-Rahmen: Pfad kreuzt eine Gate-Ebene neben der Öffnung
            for g in range(len(gates_pos)):
                loc = (pts - gates_pos[g]) @ gates_rot[g]
                off = np.abs(loc[:, 1:]).max(axis=1)
                bad = (np.abs(loc[:, 0]) < 0.1) & (off > 0.12) & (off < r_frame)
                for i in np.where(bad)[0]:
                    if r_frame - off[i] > worst:
                        y, z = loc[i, 1], loc[i, 2]
                        if z < 0 and abs(z) >= abs(y):  # unten raus -> seitlich (Ständer!)
                            y = np.copysign(r_frame, y)
                        else:
                            y, z = y * r_frame / off[i], z * r_frame / off[i]
                        worst, i_worst = r_frame - off[i], i
                        via = gates_pos[g] + gates_rot[g] @ np.array([loc[i, 0], y, z])
            if i_worst is None:
                break
            k = int(np.clip(np.searchsorted(s, ss[i_worst]), 1, len(waypoints) - 1))
            near = k if abs(s[k] - ss[i_worst]) < abs(s[k - 1] - ss[i_worst]) else k - 1
            if abs(s[near] - ss[i_worst]) < 0.05:
                if fixed[near]:
                    break
                waypoints[near] = via
            else:
                waypoints = np.insert(waypoints, k, via, axis=0)
                fixed = np.insert(fixed, k, False)
        return waypoints

    # ==================================================================================
    # Nominale Bahn (zeitoptimal) + Verformung
    # ==================================================================================
    def _plan_cfg(self) -> dict[str, int | float | tuple[float, ...]]:
        """Stelle die Parameter für die zeitoptimale Planung zusammen."""
        return {
            "n": self.plan_n,
            "t_max": self.plan_t_max,
            "tilt_max": self.plan_tilt,
            "jerk": self.plan_jerk,
            "theta": self.plan_theta,
            "lat": self.plan_lat,
            "r_pole": self.plan_r_pole,
            "r_frame": self.plan_r_frame,
            "z_min": self.plan_z_min,
            "climb": self.plan_climb,
            "unseen_len": self.unseen_len,
            "approach_len": self.approach_len,
            "approach_lat": self.approach_lat,
            "approach_vert": self.approach_vert,
            "bias": self.gate_bias,
            "gate_z": self.gate_z,
            "takeoff_dz": self.takeoff_dz,
            "takeoff_vz": self.takeoff_vz,
        }

    def _guess_segments(
        self, obs: dict[str, Any], d: float, r_pole: float, r_frame: float
    ) -> list[NDArray[np.float64]] | None:
        """Startlösung: Spline durch Hilfspunkte, an den Gate-Ebenen in Segmente geteilt."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        try:
            geo = PathGeo(self._guess_waypoints(obs, self._plan_start, d, r_pole, r_frame))
            bounds, i0 = [0.0], 0
            for g, sgn in zip(self._seq, self._dirs):
                loc = (geo.pts[i0:] - gates_pos[g]) @ (gates_rot[g][:, 0] * sgn)
                near = np.linalg.norm(geo.pts[i0:] - gates_pos[g], axis=1) < 0.3
                idx = int(np.where((loc[:-1] < 0) & (loc[1:] >= 0) & near[:-1])[0][0]) + i0
                bounds.append(geo.ss[idx])
                i0 = idx + 1
        except (ValueError, IndexError):
            return None
        return [
            geo(np.linspace(bounds[i], bounds[i + 1], self.plan_n + 1))
            for i in range(len(self._seq))
        ]

    def _nominal(self, obs: dict[str, Any]) -> None:
        """Zeitoptimale Bahn auf der nominalen Strecke (pro Strecke einmal, dann aus dem Cache)."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        cfg = self._plan_cfg()
        # Gates, die beim Durchflug zum ersten Mal gesehen werden -> Tempolimit davor
        seen = np.asarray(obs["gates_visited"], dtype=bool).copy()
        unseen = []
        for i, g in enumerate(self._seq):
            unseen.append(None if seen[g] or self.v_unseen[i] <= 0 else self.v_unseen[i])
            seen[g] = True
        key = hashlib.sha1(
            np.round(
                np.r_[
                    gates_pos.ravel(),
                    np.ravel(obs["gates_quat"]),
                    poles.ravel(),
                    self._seq,
                    self._dirs,
                    [0.0 if u is None else u for u in unseen],
                ],
                3,
            ).tobytes()
            + repr(
                (
                    sorted(cfg.items()),
                    self.plan_starts,
                    self.plan_via,
                    self._plan_start.round(3).tolist(),
                    "v6",
                )
            ).encode()
        ).hexdigest()
        plan = _PLAN_CACHE.get(key)
        cache_file = os.path.join(tempfile.gettempdir(), f"lsy_my_topt_{key[:16]}.npz")
        if plan is None and os.path.exists(cache_file):
            with np.load(cache_file) as z:
                plan = {k: z[k] for k in z.files}
        if plan is None:
            t0 = time.perf_counter()
            # Mehrere Startlösungen (IPOPT findet nur lokale Optima), die schnellste gewinnt
            plan = None
            for d, r_pole, r_frame in self.plan_starts:
                guess = self._guess_segments(obs, d, r_pole, r_frame)
                if guess is None:
                    continue
                # Plan beginnt dort, wo der senkrechte Start an den MPCC übergibt
                p0 = self._plan_start + np.array([0.0, 0.0, self.takeoff_dz])
                res = _time_optimal(
                    p0,
                    np.array([0.0, 0.0, self.takeoff_vz]),
                    gates_pos,
                    gates_rot,
                    self._seq,
                    self._dirs,
                    poles,
                    guess,
                    unseen,
                    cfg,
                )
                if res is not None and (plan is None or res["t"][-1] < plan["t"][-1]):
                    plan = res
            if plan is None:  # Rückfall: Startlösung mit 2 m/s
                guess = self._guess_segments(obs, *self.plan_starts[0])
                pts = np.concatenate([guess[0]] + [gp[1:] for gp in guess[1:]])
                plan = {
                    "P": pts,
                    "V": np.gradient(pts, axis=0) * 2.0 / 0.05,
                    "cross": np.arange(1, len(self._seq) + 1) * self.plan_n,
                    "t": np.zeros(len(pts)),
                }
                print("Zeitoptimale Planung fehlgeschlagen -> Rückfall auf Hilfspfad")
            else:
                print(
                    f"Zeitoptimale Bahn: {plan['t'][-1]:.2f} s "
                    f"(Rechenzeit {time.perf_counter() - t0:.1f} s)"
                )
                try:
                    np.savez(cache_file, **plan)
                except OSError:
                    pass
            _PLAN_CACHE[key] = plan

        # Auf ein gleichmäßiges Raster in der Bogenlänge bringen, Auslauf anhängen
        P, V = plan["P"], plan["V"]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 1e-6]
        P, V, arc = P[keep], V[keep], arc[keep]
        cross_s = arc[np.searchsorted(np.flatnonzero(keep), plan["cross"])]
        spl = CubicSpline(arc, P)
        n_pts = int(np.ceil(arc[-1] / self.def_ds))
        s_nom = np.linspace(0.0, arc[-1], n_pts + 1)
        q = spl(s_nom)
        speed = np.interp(s_nom, arc, np.linalg.norm(V, axis=1))
        t_end = spl(arc[-1], 1)
        t_end /= np.linalg.norm(t_end)
        n_out = int(np.ceil(self.run_out / self.def_ds))
        s_out = arc[-1] + self.def_ds * np.arange(1, n_out + 1)
        q = np.r_[q, q[-1] + (s_out - arc[-1])[:, None] * t_end]
        speed = np.r_[speed, np.full(n_out, speed[-1])]
        s_nom = np.r_[s_nom, s_out]
        q[0] = P[0]
        self._q_nom = q
        self._v_nom = speed
        self._cross_k = np.clip(np.round(cross_s / self.def_ds).astype(int), 0, len(q) - 1)
        self._unseen_seq = unseen
        self._disp = np.zeros_like(q)
        self._disp[0] = self._start - q[0]
        self._k_now = 0
        self._plan_gp = gates_pos.copy()
        self._plan_rot = gates_rot.copy()

    def _deform(self, obs: dict[str, Any]) -> NDArray[np.float64]:
        """Verschiebung der nominalen Bahn für die aktuell bekannten Positionen.

        Gewichtete kleinste Quadrate pro Koordinate: glatt (2. Differenzen der Verschiebung)
        plus Zielwerte mit Gewicht: Start, Durchflugzonen (starr mit dem Gate), Bahn unter der
        Drohne (bleibt wie bisher) - alles praktisch hart -, dazu weiche Ziele, die die Bahn an
        Stangen und Rahmen vorbeischieben (iterativ, je Hindernis der schlimmste Punkt).
        """
        q = self._q_nom
        m = len(q)
        gp = np.asarray(obs["gates_pos"], dtype=float)
        rot_new = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        hard, soft = 1e8, 1e4
        w = np.zeros((m, 3))
        tgt = np.zeros((m, 3))
        locked = np.zeros(m, dtype=bool)  # Punkte, die nicht weggedrückt werden dürfen

        def pin(
            k: int,
            val: NDArray[np.float64],
            weight: float,
            mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ) -> None:
            """Setze Zielverschiebung und Gewicht eines Bahnpunkts."""
            w[k] = weight * np.asarray(mask)
            tgt[k] = val

        pin(0, self._start - q[0], hard)
        locked[0] = True
        n_r = int(round(self.gate_rigid / self.def_ds))
        for i, (g, k_c) in enumerate(zip(self._seq, self._cross_k)):
            r_rel = rot_new[g] @ self._plan_rot[g].T
            # voll mitbewegter Durchflugpunkt; Verschiebung innerhalb der Toleranz weglassen
            q_full = r_rel @ (q[k_c] - self._plan_gp[g]) + gp[g]
            loc0 = rot_new[g].T @ (q[k_c] - gp[g])  # unverschobener Punkt im neuen Gate
            loc_f = rot_new[g].T @ (q_full - gp[g])
            b = self.gate_bias[i]
            loc_t = np.array(
                [
                    loc_f[0],
                    np.clip(loc0[1], b - self.gate_tol, b + self.gate_tol),
                    np.clip(loc0[2], self.gate_z - self.gate_tol, self.gate_z + self.gate_tol),
                ]
            )
            delta = rot_new[g] @ (loc_t - loc_f)
            for k in range(max(k_c - n_r, 1), min(k_c + n_r, m - 1) + 1):
                pin(k, r_rel @ (q[k] - self._plan_gp[g]) + gp[g] + delta - q[k], hard)
                locked[k] = True
        if self._airborne:  # Bahn unter der Drohne (Lage + Richtung) nicht verändern
            for k in range(max(self._k_now - 1, 1), min(self._k_now + 2, m - 1) + 1):
                if not locked[k]:
                    pin(k, self._disp[k], hard)
            locked[: self._k_now + 3] = True
        caps = [c for g in range(len(gp)) for c in _capsules(gp[g], rot_new[g])]
        disp = self._solve_disp(w, tgt)
        for _ in range(8):
            pts = q + disp
            changed = False
            for o in poles:
                d = np.linalg.norm(pts[:, :2] - o[:2], axis=1)
                viol = np.where(locked, -1.0, self.push_pole - d)
                k = int(np.argmax(viol))
                if viol[k] > 0.005:
                    away = (pts[k, :2] - o[:2]) / max(d[k], 1e-6)
                    target = np.r_[o[:2] + (self.push_pole + 0.01) * away, 0.0]
                    pin(k, target - q[k], soft, (1.0, 1.0, 0.0))
                    changed = True
            for a, b, r in caps:
                d = _capsule_dist(pts, a, b)
                need = r + self.push_frame - 0.08
                viol = np.where(locked, -1.0, need - d)
                k = int(np.argmax(viol))
                if viol[k] > 0.005:
                    ab = b - a
                    c = a + np.clip((pts[k] - a) @ ab / (ab @ ab), 0.0, 1.0) * ab
                    away = (pts[k] - c) / max(d[k], 1e-6)
                    pin(k, c + (need + 0.01) * away - q[k], soft)
                    changed = True
            if not changed:
                break
            disp = self._solve_disp(w, tgt)
        self._disp = disp
        return q + disp

    def _solve_disp(self, w: NDArray[np.float64], tgt: NDArray[np.float64]) -> NDArray[np.float64]:
        """Berechne die glatte Verschiebung aus gewichteten Zielwerten."""
        m = len(w)
        if not hasattr(self, "_lap"):  # L^T L (+ wenig Regularisierung), fünfbandig
            lap = np.zeros((m - 2, m))
            idx = np.arange(m - 2)
            lap[idx, idx], lap[idx, idx + 1], lap[idx, idx + 2] = 1.0, -2.0, 1.0
            h = lap.T @ lap + 1e-6 * np.eye(m)
            self._lap = np.zeros((3, m))
            self._lap[0, 2:], self._lap[1, 1:], self._lap[2] = (
                np.diag(h, 2),
                np.diag(h, 1),
                np.diag(h),
            )
        disp = np.empty((m, 3))
        for c in range(3):
            ab = self._lap.copy()
            ab[2] += w[:, c]
            disp[:, c] = solveh_banded(ab, w[:, c] * tgt[:, c])
        return disp

    # ==================================================================================
    # Pfad + Tunnel + Tempolimit
    # ==================================================================================
    def _build_path(self, obs: dict[str, Any]) -> None:
        """Pfad mit den aktuell bekannten Positionen, dazu Tunnelradius und Tempolimit über s."""
        old = getattr(self, "_path", None)
        p_old = old(self._s) if old is not None else None
        self._s_remap = None
        if old is not None:  # aktuellen Index der Drohne auf der verformten Bahn
            self._k_now = int(
                np.clip(np.searchsorted(old.knot_s, self._s), 0, len(self._q_nom) - 2)
            )
        wps = self._deform(obs)
        self._path = PathGeo(wps)
        if p_old is not None:
            # Bahnparameter übertragen: alter und neuer Pfad haben dieselben Rasterpunkte, also
            # Knoten k <-> Knoten k (gilt auch für die Vorhersage des MPCC)
            self._s_remap = (old.knot_s.copy(), self._path.knot_s.copy())
            self._s = float(np.interp(self._s, *self._s_remap))
            self._extra_iter = self.rebuild_iter

        pts = self._path.pts
        # Abstand zu allen Gate-Rahmen (Kollisionsboxen) und Stangen
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        d_frame = np.full(len(pts), np.inf)
        for gp, rot in zip(gates_pos, gates_rot):
            loc = (pts - gp) @ rot
            for c, h in GATE_BOXES:
                d = np.linalg.norm(np.maximum(np.abs(loc - c) - h, 0.0), axis=1)
                d_frame = np.minimum(d_frame, d)
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        d_pole = (
            np.linalg.norm(pts[:, None, :2] - poles[None, :, :2], axis=2).min(axis=1)
            if len(poles)
            else np.full(len(pts), np.inf)
        )
        r = np.minimum(
            d_frame - DRONE_RADIUS - self.margin, d_pole - POLE_RADIUS - DRONE_RADIUS - self.margin
        )
        self._tunnel = np.clip(r, self.tunnel_min, self.tunnel_max)
        # Durchflüge auf diesem Pfad (Bogenlänge, Gate) für die Gate-Grenze im MPCC
        self._gate_s = np.array([self._path.knot_s[k] for k in self._cross_k])
        self._gate_ids = list(self._seq)
        self._gates_now = (gates_pos.copy(), gates_rot.copy())

        # Tempolimit aus dem zeitoptimalen Profil
        v_nom = self._v_nom
        visited = np.asarray(obs["gates_visited"], dtype=bool)
        # Vor noch nicht gesehenen Gates hart begrenzen (dort muss die Drohne noch korrigieren)
        cap = np.full(len(v_nom), 100.0)
        n_cap = int(round((self.unseen_len + 0.1) / self.def_ds))
        for i, (g, k_c) in enumerate(zip(self._seq, self._cross_k)):
            if self._unseen_seq[i] is not None and not visited[g]:
                cap[max(k_c - n_cap, 0) : k_c + 1] = self._unseen_seq[i]
        v_lim = np.minimum(self.topt_vf * v_nom + 0.3, cap)
        vmax = np.minimum(self.v_max, np.interp(self._path.ss, self._path.knot_s, v_lim))
        self._vmax = vmax

    # ==================================================================================
    # MPCC
    # ==================================================================================
    def _mpcc_control(
        self,
        p: NDArray[np.float64],
        v: NDArray[np.float64],
        quat: NDArray[np.float64],
        ang_vel: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64]] | None:
        """Berechne das MPCC-Kommando oder melde einen fehlgeschlagenen Solverlauf."""
        n, dt = self.mpc_n, 1.0 / self._freq
        path = self._path
        tilt, dtilt = _tilt_state(quat, ang_vel)
        f_cmd0 = float(np.clip(self._f_cmd, MPC_F_MIN, MPC_F_ABS))
        s0 = self._s
        ds0 = float(
            np.clip(
                self._ds_pred[1] if self._mpc_warm else np.dot(v, path.tangent(s0)), 0.0, self.v_max
            )
        )
        x0 = np.r_[p, tilt, v, dtilt, self._al, self._f_est, f_cmd0, s0, ds0]

        sol = self._mpc
        xs = us = None
        if self._mpc_warm:
            xs = np.array([sol.get(j, "x") for j in range(n + 1)])
            us = np.array([sol.get(j, "u") for j in range(n)])
            if getattr(self, "_s_remap", None) is not None:  # Pfad hat sich geändert
                xs[:, 14] = np.interp(xs[:, 14], *self._s_remap)
                self._s_remap = None
            # Lösung und Linearisierungspunkte um einen Schritt verschieben
            s_lin = np.r_[xs[1:, 14], xs[-1, 14] + dt * xs[-1, 15]]
            s_lin += s0 - s_lin[0]
        else:
            ds_guess = np.full(n + 1, max(ds0, 0.5))
            s_lin = s0 + dt * np.r_[0.0, np.cumsum(ds_guess[:n])]
        s_lin = np.clip(s_lin, 0.0, path.length)
        p_lin = path(s_lin)
        t_lin = path.tangent(s_lin)
        # kleineren der beiden benachbarten Werte nehmen (nicht optimistisch interpolieren)
        idx = np.clip(np.searchsorted(path.ss, s_lin, side="right") - 1, 0, len(path.ss) - 1)
        nxt = np.minimum(idx + 1, len(path.ss) - 1)
        r_tun = np.minimum(self._tunnel[idx], self._tunnel[nxt])
        vmax = np.minimum(self._vmax[idx], self._vmax[nxt])
        # Gate-Öffnung statt Tunnel in der Nähe der Durchflüge
        gate_par = np.tile(np.r_[np.zeros(3), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10.0], (n + 1, 1))
        g_pos, g_rot = self._gates_now
        for s_c, g in zip(self._gate_s, self._gate_ids):
            act = np.abs(s_lin - s_c) < self.gate_win
            if np.any(act):
                c = g_pos[g] - np.array([0.0, 0.0, 0.02])  # Kollisionskugel sitzt 2 cm höher
                gate_par[act] = np.r_[c, g_rot[g][:, 1], g_rot[g][:, 2], self.gate_w]
                r_tun[act] = np.maximum(r_tun[act], self.gate_tun)

        sol.set(0, "lbx", x0)
        sol.set(0, "ubx", x0)
        y_ref = np.zeros(4 + 1 + 2 + NU)
        for j in range(n + 1):
            sol.set(
                j,
                "p",
                np.r_[
                    self._dist,
                    p_lin[j],
                    t_lin[j],
                    s_lin[j],
                    r_tun[j],
                    vmax[j],
                    self._kap,
                    gate_par[j],
                ],
            )
            if j < n:
                y_ref[4] = vmax[j] + self.ds_add
                sol.set(j, "yref", y_ref)
        # Schubgrenze folgt dem Schubfaktor: schwere Drohnen bekommen dieselbe Beschleunigung
        f_max = float(np.clip(MPC_F_MAX / self._kap, MPC_F_MAX, MPC_F_ABS))
        for j in range(1, n + 1):
            sol.constraints_set(
                j, "ubx", np.array([2.4, MPC_TILT_MAX, MPC_TILT_MAX, f_max, vmax[j]])
            )
        if self._mpc_warm:  # letzte Lösung um einen Schritt verschieben
            for j in range(n):
                x = xs[min(j + 1, n)].copy()
                x[14] = s_lin[j]
                sol.set(j, "x", x)
                sol.set(j, "u", us[min(j + 1, n - 1)])
            x = xs[n].copy()
            x[14] = s_lin[n]
            sol.set(n, "x", x)
        else:  # Kaltstart entlang des Pfads
            for j in range(n + 1):
                x = np.r_[
                    p_lin[j],
                    tilt,
                    t_lin[j] * ds_guess[j],
                    np.zeros(4),
                    self._f_est,
                    f_cmd0,
                    s_lin[j],
                    ds_guess[j],
                ]
                sol.set(j, "x", x)
                if j < n:
                    sol.set(j, "u", np.zeros(NU))
        sol.set(0, "x", x0)
        solve_start = time.perf_counter()
        status = sol.solve()
        for _ in range(getattr(self, "_extra_iter", 0)):  # nach Bahnänderung weiter iterieren
            if status != 0:
                break
            status = sol.solve()
        self._extra_iter = 0
        self._solve_ms.append(1000.0 * (time.perf_counter() - solve_start))
        x1 = sol.get(1, "x")
        if status != 0 or not np.all(np.isfinite(x1)):
            self._mpc_warm = False
            self._mpc_fail += 1
            return None
        self._mpc_warm = True
        self._v_pred = x1[5:8]
        self._al = x1[10:12]
        self._f_cmd = float(np.clip(x1[13], MPC_F_MIN, MPC_F_ABS))
        self._ds_pred = np.array([sol.get(j, "x")[15] for j in range(n + 1)])
        # Innere Schleife invertieren: Mitte des Regelintervalls ist der Bezugspunkt
        u0 = sol.get(0, "u")
        tilt_m = 0.5 * (tilt + x1[3:5])
        rate_m = 0.5 * (dtilt + x1[8:10])
        al_m = 0.5 * (x0[10:12] + x1[10:12])
        tilt_cmd = np.clip(
            tilt_m + (u0[:2] + MPC_KD * al_m) / MPC_K1, -MPC_CMD_TILT_MAX, MPC_CMD_TILT_MAX
        )
        # Körperraten bei fester Gier aus den Kippraten (R = Ry(pitch) Rx(roll))
        cr, sr = np.cos(tilt_m[0]), np.sin(tilt_m[0])
        rate_b = np.array([rate_m[0], rate_m[1] * cr, -rate_m[1] * sr])
        return tilt_cmd, self._f_cmd, rate_b

    @staticmethod
    def _attitude_to_acc(
        tilt_cmd: NDArray[np.float64], f_cmd: float, quat: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Beschleunigungs-Sollwert, für den der Mellinger genau diese Lage + Schub kommandiert."""
        z_des = _tilt_axis(tilt_cmd)
        z_cur = Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])
        cos = max(float(z_des @ z_cur), 0.5)  # der Mellinger projiziert auf die aktuelle z-Achse
        return f_cmd / (MELLINGER_F_PER_ACC * cos) * z_des - GRAVITY

    def _mellinger_cmd(
        self, p_ref: NDArray[np.float64], v_ref: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        """Rückfall / Start: Sollzustand an den Mellinger, MPC-Schubzustand mitführen."""
        acc_send = GRAVITY / THRUST_SCALE - GRAVITY
        f_cmd = float(np.clip(MELLINGER_F_PER_ACC * (acc_send + GRAVITY)[2], 0.0, MPC_F_MAX))
        self._f_est = f_cmd + (self._f_est - f_cmd) * np.exp(-1.0 / (self._freq * MPC_TAU))
        self._f_cmd = f_cmd
        self._al = np.zeros(2)
        return np.concatenate((p_ref, v_ref, acc_send, [0.0, 0.0, 0.0, 1.0], np.zeros(3))).astype(
            np.float32
        )

    def compute_control(
        self, obs: dict[str, Any], info: dict[str, Any] | None = None
    ) -> NDArray[np.float32]:
        """Berechne den Steuerbefehl aus der aktuellen Beobachtung."""
        p = np.asarray(obs["pos"], dtype=float)
        v = np.asarray(obs["vel"], dtype=float)
        quat = np.asarray(obs["quat"], dtype=float)
        ang_vel = np.asarray(obs["ang_vel"], dtype=float)

        # Bahnparameter nachführen, aber (fast) nur vorwärts: sonst läuft der Bezugspunkt einer
        # zurückfallenden Drohne hinterher und der MPCC jagt sie rückwärts
        self._s = self._path.project(p, self._s, back=0.05, ahead=0.8)
        self._last_ref = self._path(self._s)  # nur fürs Auswerten

        self._airborne |= p[2] > self.takeoff_height
        if not self._airborne:  # senkrecht abheben: Messwert als Sollzustand, fester Schub
            self._v_pred = None
            f_cmd = self.takeoff_f
            acc = np.array([0.0, 0.0, f_cmd / MELLINGER_F_PER_ACC]) - GRAVITY
            self._f_est = f_cmd + (self._f_est - f_cmd) * np.exp(-1.0 / (self._freq * MPC_TAU))
            self._f_cmd = f_cmd
            return np.concatenate((p, v, acc, [0.0, 0.0, 0.0, 1.0], np.zeros(3))).astype(np.float32)

        z_up = float(Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])[2])
        if z_up < np.cos(MPC_RESCUE_TILT):  # Modell gilt hier nicht mehr: erst aufrichten
            self._v_pred = None
            self._mpc_warm = False
            self._al = np.zeros(2)
            self._f_cmd = 0.6
            self._f_est = 0.6 + (self._f_est - 0.6) * np.exp(-1.0 / (self._freq * MPC_TAU))
            # Höchstens 1 rad auf einmal: ab 90 Grad Lagefehler schaltet der Bordregler die
            # Momente ab
            z_cur = Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])
            ang = float(np.arccos(np.clip(z_up, -1.0, 1.0)))
            z_des = (np.sin(ang - 1.0) * z_cur + np.sin(1.0) * np.array([0.0, 0.0, 1.0])) / max(
                np.sin(ang), 0.1
            )
            z_des /= np.linalg.norm(z_des)
            a = self._f_cmd / MELLINGER_F_PER_ACC * z_des - GRAVITY
            return np.concatenate((p, v, a, [0.0, 0.0, 0.0, 1.0], np.zeros(3))).astype(np.float32)

        if self._v_pred is not None:
            # Beobachter: Fehler der Geschwindigkeitsvorhersage -> in Schubrichtung vor allem
            # Schubfaktor (Masse), sonst additive Stör-Beschleunigung
            err = (v - self._v_pred) * self._freq
            z_b = Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])
            e_par = float(err @ z_b)
            a_thrust = MPC_KF * self._f_est / MPC_MASS
            self._kap = float(
                np.clip(self._kap + self.k_kap * e_par / max(a_thrust, 3.0), 0.88, 1.15)
            )
            err = err - (1.0 - self.dist_par) * e_par * z_b
            self._dist = np.clip(self._dist + self.k_dist * err, -self.dist_max, self.dist_max)
        res = self._mpcc_control(p, v, quat, ang_vel)
        if res is None:
            self._v_pred = None
            t = self._path.tangent(self._s)
            return self._mellinger_cmd(self._path(self._s + 0.3), t * 1.0)
        tilt_cmd, f_cmd, rate_b = res
        self._f_est = f_cmd + (self._f_est - f_cmd) * np.exp(-1.0 / (self._freq * MPC_TAU))
        a = self._attitude_to_acc(tilt_cmd, f_cmd, quat)
        # Messwert als Sollzustand: der Mellinger setzt nur noch Lage/Schub um
        return np.concatenate((p, v, a, [0.0, 0.0, 0.0, 1.0], rate_b)).astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.float32],
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> bool:
        """Aktualisiere die Statistik und passe die Bahn bei neuen Sensordaten an."""
        self._tick += 1
        self._speed_sum += float(np.linalg.norm(obs["vel"]))
        self._speed_samples += 1
        self._last_obs = obs

        # Bahn verformen, sobald ein Gate oder eine Stange zum ersten Mal gesehen wird
        known = np.r_[obs["gates_visited"], obs["obstacles_visited"]].astype(bool)
        if np.any(known & ~self._known):
            self._known = known
            self._build_path(obs)
        return False

    def episode_callback(self) -> None:
        """Gib die Diagnose und die Kennzahlen der beendeten Episode aus."""
        o = self._last_obs
        pos = np.asarray(o["pos"])
        poles = np.asarray(o["obstacles_pos"])
        d_pole = np.linalg.norm(poles[:, :2] - pos[:2], axis=1).min() if len(poles) else np.inf
        d_gate = np.linalg.norm(np.asarray(o["gates_pos"]) - pos, axis=1).min()
        cause = "Stange" if d_pole < 0.12 else "Gate-Rahmen" if d_gate < 0.5 else "sonstiges"
        print(
            f"ENDE t={self._tick / self._freq:.2f}s  Gates={int(o['n_gates_passed'])}  "
            f"pos={np.round(pos, 2)}  Stange {d_pole * 100:.0f} cm  Gate {d_gate * 100:.0f} cm  "
            f"-> {cause}  (MPCC-Ausfälle: {self._mpc_fail})"
        )
        summary = {
            "time": self._tick / self._freq,
            "gates": int(o["n_gates_passed"]),
            "expected_gates": len(self._seq),
            "mpc_fail": self._mpc_fail,
            "mean_speed": self._speed_sum / max(self._speed_samples, 1),
            "solve_p95_ms": float(np.percentile(self._solve_ms, 95)) if self._solve_ms else None,
        }
        print("DRONE_METRICS " + json.dumps(summary))

    def render_callback(self, sim: Sim) -> None:
        """Zeichne den aktuellen Pfad in die Simulation."""
        from crazyflow.sim.visualize import draw_line

        draw_line(sim, self._path.pts[::5], rgba=(0.0, 1.0, 0.0, 1.0))

    def episode_reset(self) -> None:
        """Setze die Episodenstatistik und den MPCC-Warmstart zurück."""
        self._tick = 0
        self._solve_ms.clear()
        self._speed_sum = 0.0
        self._speed_samples = 0
        self._mpc_warm = False
        self._v_pred = None
        self._mpc_fail = 0
