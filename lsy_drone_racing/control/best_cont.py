"""Best controller: time-optimal path through the gates + MPCC (maximise progress in a tunnel)."""

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
from acados_template.builders import ocp_get_default_cmake_builder
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


# The onboard controller realises the acceleration ~1.46-fold (identified in sim)
THRUST_SCALE = 1.611 * 0.0393 / 0.0434
GRAVITY = np.array([0.0, 0.0, 9.81])

# ---- Drone model (from my_controller_2.py, identified in the Crazyflow sim) -----------------
# Attitude without yaw via two tilt angles of the body z-axis:
# z = [sin(p) cos(r), -sin(r), cos(r) cos(p)].
# Onboard attitude controller (identified in sim), per tilt axis with angular acceleration al:
#   T = K1 * (tilt_cmd - tilt) + K2 * (rate_cmd - dtilt) - KD * al,  |T| <= AL_MAX
#   dal = (T - al) / TAU_AL
# Without a rate feedforward the damping K2 dominates (slow pole K1 / K2 ~ 7 rad/s). The MPC
# therefore plans T directly and sends attitude + rate such that the inner loop follows exactly
# this motion.
MPC_K1 = 800.0
MPC_K2 = 117.0
MPC_KD = 1.33
MPC_AL_MAX = 170.0  # max. angular acceleration [rad/s^2]
MPC_TAU_AL = 0.051  # rotor lag [s]
MPC_CMD_TILT_MAX = 1.3  # limit of the attitude command sent [rad]
MPC_MASS = 0.0434  # nominal mass [kg] (random +-0.005, handled by the observer)
MPC_KF = 0.978  # realised / commanded collective thrust
MPC_TAU = 0.057  # thrust lag (first order) [s]
MPC_F_MAX = 0.74  # thrust command [N]; the saturation limit keeps reserve for torques
MPC_F_MIN = 0.15
MPC_F_ABS = 0.76  # absolute thrust limit, also for heavy drones [N]; motors end at 0.8 N
MPC_TILT_MAX = 1.14  # max. total tilt [rad]
MPC_TILT_Z = 3000.0  # penalty on the tilt limit (linear, quadratic x 100)
MPC_TILT_LEAD = 0.08  # tilt limit applies to tilt + LEAD * tilt rate (stopping distance) [s]
MPC_RESCUE_TILT = 1.3  # above this total tilt: suspend the MPC and level the drone [rad]
# Motors: thrust f / 4 +- torque share must stay in [THR_MIN, THR_MAX], otherwise the torque is lost
# and the drone keeps tilting. T = 288 rad/s^2 corresponds to 0.049 N per motor.
MPC_THR_MAX = 0.2  # [N per motor]
MPC_THR_MIN = 0.0214
MPC_THR_PER_T = 0.049 / 288.0
MPC_SAT_Z = 300.0  # penalty on the saturation limit (linear, quadratic x 10)
MPC_DF_MAX = 15.0  # max. rate of change of the thrust command [N/s]
DRAG = np.array([0.0216, 0.0216, 0.0248]) / 0.0434  # linear drag [1/s]
# Thrust path in the Mellinger controller:
# F = mass_thrust / pwm_max * 4 * thrust_max * m_ctrl * |a_cmd + g|
MELLINGER_F_PER_ACC = 132000.0 / 65535.0 * 0.8 * 0.0393

# Collision geometry from the simulator
DRONE_RADIUS = 0.10  # collision sphere of the drone (0.086 m) plus reserve [m]
POLE_RADIUS = 0.015
GATE_BOXES = (  # (centre, half extents) in the gate frame, x = passing direction
    (np.array([0.0, 0.0, 0.28]), np.array([0.01, 0.36, 0.08])),  # top
    (np.array([0.0, 0.0, -0.28]), np.array([0.01, 0.36, 0.08])),  # bottom
    (np.array([0.0, -0.28, 0.0]), np.array([0.01, 0.08, 0.36])),  # left
    (np.array([0.0, 0.28, 0.0]), np.array([0.01, 0.08, 0.36])),  # right
    (np.array([0.0, 0.0, -0.86]), np.array([0.05, 0.05, 0.50])),  # stand
)
# The same parts as capsules (axis from a to b, radius) for the planner and path deformation.
# The bar axes run into the corners (+-0.36), otherwise the box corners stick out of the capsules
GATE_CAPSULES = (
    ((0.0, -0.36, 0.28), (0.0, 0.36, 0.28), 0.08),
    ((0.0, -0.36, -0.28), (0.0, 0.36, -0.28), 0.08),
    ((0.0, -0.28, -0.36), (0.0, -0.28, 0.36), 0.08),
    ((0.0, 0.28, -0.36), (0.0, 0.28, 0.36), 0.08),
    ((0.0, 0.0, -0.36), (0.0, 0.0, -1.36), 0.07),
)

NX, NU, NP = 16, 4, 24  # MPCC: states, inputs, parameters per stage


def _tilt_axis(tilt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Body z-axis (..., 3) from the tilt angles (..., 2) = [roll, pitch]."""
    r, p = tilt[..., 0], tilt[..., 1]
    return np.stack((np.sin(p) * np.cos(r), -np.sin(r), np.cos(r) * np.cos(p)), axis=-1)


def _tilt_state(
    quat: NDArray[np.float64], ang_vel: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tilt angles and their rates from attitude and angular velocity."""
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
    """Frame and stand of a gate as capsules in world coordinates: (a, b, radius)."""
    return [
        (gate_pos + gate_rot @ np.array(a), gate_pos + gate_rot @ np.array(b), r)
        for a, b, r in GATE_CAPSULES
    ]


def _capsule_dist(
    pts: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distance of the points (n, 3) to the segment a-b."""
    d = b - a
    t = np.clip((pts - a) @ d / (d @ d), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * d), axis=1)


# ======================================================================================
# Time-optimal point-mass trajectory
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
    """Time-optimal trajectory through the gates in the order seq.

    Point mass with thrust acceleration f as state (jerk as input), |f| <= t_max,
    tilt <= tilt_max, linear drag. One segment with its own duration per gate; the segment end
    lies in the gate opening, the flight direction at most theta away from the gate normal.
    guess: initial guess (points per segment, n+1 each). unseen[i]: speed limit in the last
    unseen_len metres before gate i, because it is only detected there. Starts in start with
    velocity start_v (state at the end of the vertical take-off), thrust free there.
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
    gaps = [
        (poles[j][:2], poles[k][:2])
        for j in range(len(poles))
        for k in range(j + 1, len(poles))
        if np.linalg.norm(poles[j][:2] - poles[k][:2]) < cfg["gap_min"]
    ]
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
            if i == 0 and k <= n // 3:  # start: leave the ground with at least slope climb
                dxy2 = ca.sumsqr(p[0:2] - start[0:2])
                opti.subject_to(p[2] >= start[2])
                opti.subject_to((p[2] - start[2]) ** 2 >= cfg["climb"] ** 2 * dxy2)
            else:
                opti.subject_to(p[2] >= cfg["z_min"])
            # Stay inside the arena (the env terminates outside the safety limits)
            opti.subject_to(opti.bounded(cfg["lo"], p, cfg["hi"]))
            for o, r_o in zip(poles, cfg["r_poles"]):
                if np.min(np.linalg.norm(gp[:, :2] - o[:2], axis=1)) < cfg["act_dist"]:
                    opti.subject_to(ca.sumsqr(p[0:2] - o[0:2]) >= r_o**2)
            # Narrow gaps between two poles are closed: the poles may still move by +-0.15 m
            for a2, b2 in gaps:
                d2 = b2 - a2
                t2 = ca.fmin(ca.fmax(ca.dot(p[0:2] - a2, d2) / float(d2 @ d2), 0.0), 1.0)
                opti.subject_to(ca.sumsqr(p[0:2] - (a2 + t2 * d2)) >= cfg["r_pole"] ** 2)
            for g, (a, b, r) in caps:
                if np.min(np.linalg.norm(gp - gates_pos[g], axis=1)) < cfg["act_dist"]:
                    d = b - a
                    t = ca.fmin(ca.fmax(ca.dot(p - a, d) / float(d @ d), 0.0), 1.0)
                    # Extra distance to unseen gates that are only passed by (not the gate this
                    # segment flies through or just came from): their pose is still uncertain
                    by = seq[i] != g and (i == 0 or seq[i - 1] != g)
                    r_f = cfg["r_frame"] + (cfg["frame_extra"][g] if by else 0.0)
                    opti.subject_to(ca.sumsqr(p - (a + t * d)) >= (r + r_f) ** 2)
            # Plug the openings of all gates except the target of this segment (and the gate
            # just left, until the exit constraint is over): flying through a foreign opening
            # (e.g. back through the last gate to cut a hairpin) leaves no room for errors
            for g in range(len(gates_pos)):
                if cfg["plug_r"] <= 0.0:
                    break
                if g == seq[i] or (i > 0 and g == seq[i - 1] and k <= cfg["cross_nodes"] + 2):
                    continue
                r_p = cfg["plug_r"] + cfg["frame_extra"][g]
                opti.subject_to(ca.sumsqr(p - gates_pos[g]) >= r_p**2)
            if unseen[i] is not None and seg_len[-1] - seg_len[k] < cfg["unseen_len"]:
                opti.subject_to(ca.sumsqr(v) <= unseen[i] ** 2)
            if unseen[i] is not None and seg_len[-1] - seg_len[k] < cfg["approach_len"]:
                # Approach cone: almost level and not too oblique. If the gate moves along its
                # axis, the path then hardly changes
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
            # Leave the previous gate along its normal too (no U-turn right at the gate plane)
            rot_p = gates_rot[seq[i - 1]]
            nrm_p = rot_p[:, 0] * dirs[i - 1]
            c_p = gates_pos[seq[i - 1]] + cfg["gate_z"] * rot_p[:, 2]
            k1 = min(cfg["cross_nodes"], n)
            opti.subject_to(ca.dot(xs[i][0:3, k1] - c_p, nrm_p) >= cfg["cross_dist"])
            for k in range(1, k1 + 1):
                opti.subject_to(ca.dot(xs[i][3:6, k], nrm_p) >= cfg["v_cross"])
        # Segment end = crossing of gate seq[i]
        rot = gates_rot[seq[i]]
        nrm, gy, gz = rot[:, 0] * dirs[i], rot[:, 1], rot[:, 2]
        rel = xs[i][0:3, n] - (gates_pos[seq[i]] + cfg["gate_z"] * gz)
        ve = xs[i][3:6, n]
        opti.subject_to(ca.dot(rel, nrm) == 0.0)
        opti.subject_to(
            opti.bounded(cfg["bias"][i] - cfg["lat"], ca.dot(rel, gy), cfg["bias"][i] + cfg["lat"])
        )
        opti.subject_to(opti.bounded(-0.03, ca.dot(rel, gz), 0.03))
        # Real crossing from the -normal side: without these, the last segment may reach the gate
        # plane from behind and stop its normal velocity at exactly 0 (nothing follows it)
        opti.subject_to(ca.dot(ve, nrm) >= cfg["v_cross"])
        opti.subject_to(ca.dot(ve, nrm) ** 2 >= cos_t2 * ca.sumsqr(ve))
        # ... and over the last cross_nodes nodes: moving along the normal, starting at least
        # cross_dist in front of the plane (no last-moment reversal through the opening)
        c = gates_pos[seq[i]] + cfg["gate_z"] * gz
        k0 = max(n - cfg["cross_nodes"], 0)
        opti.subject_to(ca.dot(xs[i][0:3, k0] - c, nrm) <= -cfg["cross_dist"])
        for k in range(k0, n):
            opti.subject_to(ca.dot(xs[i][3:6, k], nrm) >= cfg["v_cross"])
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
        {
            "max_iter": 1500,
            "max_cpu_time": cfg["cpu_max"],  # hopeless guesses otherwise run for minutes
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
        },
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
    """Define the MPCC model.

    x = [p, tilt, v, dtilt, al, f, f_cmd, s, ds], u = [T (2), thrust command rate, dds].

    Parameters per stage: [disturbance acceleration (3), path point (3), tangent (3), s_lin,
    tunnel radius, speed limit, thrust factor (estimated, absorbs the random mass), gate window
    (centre (3), gate y-axis (3), gate z-axis (3), half width, half height)].
    The path is linearised around s_lin: p_ref(s) ~ path point + tangent * (s - s_lin).
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
    # Gate opening (MPCC++): centre, gate y- and z-axis, half width / height (large = inactive)
    g_c, g_y, g_z, g_w, g_wz = par[13:16], par[16:19], par[19:22], par[22], par[23]

    cr, sr = ca.cos(tilt[0]), ca.sin(tilt[0])
    cp, sp = ca.cos(tilt[1]), ca.sin(tilt[1])
    z_body = ca.vertcat(sp * cr, -sr, cr * cp)
    acc = z_body * MPC_KF * kap * f / MPC_MASS - GRAVITY - DRAG * v + dist

    # Error to the (linearised) path: lag = along, contour = across
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
    # stay in the tunnel; limit the real speed along the path
    model.con_h_expr = ca.vertcat(
        ca.sqrt(ca.dot(e_con, e_con) + 1e-6) - r_tun,
        ca.dot(t_lin, v) - v_lim,
        np.cos(MPC_TILT_MAX)
        - ca.cos(tilt[0] + MPC_TILT_LEAD * dtilt[0]) * ca.cos(tilt[1] + MPC_TILT_LEAD * dtilt[1]),
        ca.sqrt(ca.dot(p - g_c, g_y) ** 2 + 1e-6) - g_w,
        ca.sqrt(ca.dot(p - g_c, g_z) ** 2 + 1e-6) - g_wz,
    )
    model.con_h_expr_e = model.con_h_expr
    # Motor reserve (normalised to 0.05 N): torque and collective thrust share the range
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
    """Build the MPCC (or load it from the temp folder). Never writes into the repo."""
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
                "mpcc_gate_v6",
            )
        ).encode()
    ).hexdigest()[:10]
    name = f"lsy_my_mpcc_{key}"
    root = os.path.join(tempfile.gettempdir(), name)
    json_file = os.path.join(root, "ocp.json")
    if os.name == "nt":  # Windows: acados builds the DLL with CMake/MinGW (as in attitude_mpc.py)
        lib = os.path.join(root, "c_generated_code", f"acados_ocp_solver_{name}.dll")
    else:
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
    # Progress: ds should be as large as possible -> (ds - ds_ref)^2 with a large ds_ref
    ocp.cost.yref = np.r_[np.zeros(4), ds_ref, np.zeros(2 + NU)]
    ocp.cost.yref_e = np.zeros(4)
    ocp.parameter_values = np.zeros(NP)
    ocp.constraints.x0 = np.zeros(NX)
    ocp.constraints.lbu = np.array([-MPC_AL_MAX, -MPC_AL_MAX, -MPC_DF_MAX, -dds_max])
    ocp.constraints.ubu = np.array([MPC_AL_MAX, MPC_AL_MAX, MPC_DF_MAX, dds_max])
    ocp.constraints.idxbu = np.arange(NU)
    # States: height (soft), tilt angles (soft), thrust command, path speed (soft)
    ocp.constraints.idxbx = np.array([2, 3, 4, 13, 15])
    ocp.constraints.lbx = np.array([z_min, -MPC_TILT_MAX, -MPC_TILT_MAX, MPC_F_MIN, 0.0])
    ocp.constraints.ubx = np.array([2.4, MPC_TILT_MAX, MPC_TILT_MAX, MPC_F_MAX, 10.0])
    # Total tilt in addition to the single axes: 0.9 rad per axis would be 68 degrees together
    ocp.constraints.idxsbx = np.array([0, 1, 2, 4])
    ocp.constraints.lh = np.full(7, -100.0)
    ocp.constraints.uh = np.zeros(7)
    ocp.constraints.idxsh = np.arange(7)
    # Penalties of the soft constraints:
    # [ground, tilt, tilt, ds, tunnel, speed, total tilt, gate (2), saturation (2)]
    z_e = np.array([2000.0, 100.0, 100.0, 100.0, tun_z, 1000.0, MPC_TILT_Z, tun_z, tun_z])
    z2_e = np.array([2e5, 1000.0, 1000.0, 1000.0, tun_Z, 1e4, 100 * MPC_TILT_Z, tun_Z, tun_Z])
    ocp.cost.zl = ocp.cost.zu = np.r_[z_e, MPC_SAT_Z, MPC_SAT_Z]
    ocp.cost.Zl = ocp.cost.Zu = np.r_[z2_e, 10 * MPC_SAT_Z, 10 * MPC_SAT_Z]
    # Same constraints in the terminal node
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
    cmake_builder = None
    if os.name == "nt":
        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.build_dir = "build-mingw"
    return AcadosOcpSolver(ocp, json_file=json_file, verbose=False, cmake_builder=cmake_builder)


class PathGeo:
    """Cubic spline through the waypoints, parametrised by arc length s."""

    def __init__(self, waypoints: NDArray[np.float64]) -> None:
        """Initialise the arc-length parametrised path."""
        waypoints = np.asarray(waypoints, dtype=float)
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        if len(seg) == 0 or np.any(seg < 1e-6):
            raise ValueError("Path needs distinct consecutive waypoints")
        knots = np.r_[0.0, np.cumsum(seg)]
        raw = CubicSpline(knots, waypoints)
        # 2.5 mm grid + all knots, then reparametrise by the true arc length
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
        self.knot_s = np.interp(knots, u, arc)  # arc length of the waypoints
        self.ss = np.r_[np.arange(0.0, self.length, 0.01), self.length]
        self.pts = self.spl(self.ss)

    def __call__(self, s: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the path at one or more arc lengths."""
        return self.spl(np.clip(s, 0.0, self.length))

    def tangent(self, s: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the unit tangent along the path."""
        d = self.d1(np.clip(s, 0.0, self.length))
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)

    def project(
        self, p: NDArray[np.float64], s_guess: float, back: float = 0.3, ahead: float = 0.8
    ) -> float:
        """Project a point onto the path within a local search window."""
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
    """Time-optimal path through the gates + MPCC (maximise progress in a tunnel)."""

    def __init__(self, obs: dict[str, Any], info: dict[str, Any], config: ConfigDict) -> None:
        """Initialise planner, MPCC and the controller state."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # ---- Time-optimal trajectory -------------------------------------------------------
        self.plan_n = 30  # nodes per segment
        self.plan_t_max = 15.1  # thrust limit [m/s^2]
        self.plan_tilt = 0.982  # max. tilt [rad]
        self.plan_jerk = 150.0  # max. jerk per axis [m/s^3]
        self.plan_theta = 1.077  # max. angle flight direction <-> gate normal at a crossing [rad]
        self.plan_lat = 0.03  # crossing point: max. lateral offset from the gate centre [m]
        self.plan_r_pole = 0.165  # distance to the pole axis [m]
        # Unseen poles may be up to 0.15 m (per axis) away from their nominal position: keep this
        # much extra distance in the plan (costs a little time, avoids late swerves at 2-3 m/s)
        self.pole_unc = 0.15
        self.plan_r_frame = 0.10  # distance to the frame capsules [m]
        self.frame_unc = 0.18  # extra distance to frames of unseen gates that are passed by [m]
        # Pole / frame constraints are only added where the initial guess comes this close; the
        # solution may move far from the guess, so this must clearly exceed the margins [m]
        self.plan_act_dist = 1.5
        self.plan_cpu_max = 15.0  # CPU time limit per IPOPT solve [s]
        self.plan_budget = 40.0  # no further initial guesses after this once a plan exists [s]
        self.gap_min = 0.5  # gaps between poles narrower than this are closed in the plan [m]
        self.plug_r = 0.30  # sphere that closes the openings of non-target gates [m]
        self.plan_z_min = 0.15  # minimum height after the start [m]
        self.plan_climb = 0.5  # start: height >= climb * horizontal distance
        # speed before gates that have not been seen yet [m/s], per pass (0 = no limit)
        self.v_unseen = (2.2, 3.2, 2.8, 2.7, 0.0)
        self.v_pass = 2.0  # speed limit where the path passes close to unseen objects [m/s]
        self.pass_dist = 0.35  # ... closer than this to an unseen frame / pole surface [m]
        self.unseen_len = 0.7  # ... over the last metres before the gate (sensor range)
        self.approach_len = 0.4  # approach cone before unseen gates: length [m]
        # ... max. lateral angle to the gate axis [rad], per pass
        self.approach_lat = (0.5, 0.5, 0.5, 0.5, 0.5)
        self.approach_vert = 0.1  # ... and vertical [rad]
        # Lateral offset of the crossing point (gate y) [m], per pass. Tuned for the fixed level-2
        # track (gate 0 was approached obliquely); 0 on level 3, where the track is random
        self.gate_bias = (0.078, 0.0, 0.0, 0.0, 0.0)
        self.plan_via = {}  # via points of the initial guess: {pass index: (x, y, z)}
        self.v_cross = 0.5  # min. speed along the gate normal at each crossing [m/s]
        self.cross_nodes = 5  # ... over this many nodes before the crossing
        self.cross_dist = 0.2  # ... which start at least this far in front of the gate [m]
        self.gate_z = 0.05  # crossing height relative to the gate centre [m]
        self.arena_margin = 0.25  # planned path stays this far inside the safety limits [m]
        self.run_out = 1.0  # run-out behind the last gate [m]
        # initial guesses for the planner: (via point distance at the gate, pole, frame distance)
        self.plan_starts = ((0.3, 0.25, 0.5), (0.5, 0.3, 0.5), (0.2, 0.35, 0.6))

        # ---- Path deformation ---------------------------------------------------------------
        self.def_ds = 0.05  # grid of the deformation [m]
        self.gate_rigid = 0.10  # region around the crossing that moves rigidly with the gate [m]
        self.def_z_floor = True  # clamp the deformed path to plan_z_min (after take-off)
        self.def_dz_max = 0.2  # max. vertical displacement of the deformation (0 = off) [m]
        self.gate_tol = 0.04  # gate shifts up to this are not followed [m]
        self.push_pole = 0.165  # path at least this far from poles [m]
        self.push_frame = 0.19  # ... and from the frame capsule axes (incl. frame width) [m]

        # ---- MPCC ---------------------------------------------------------------------------
        self.mpc_n = 40  # horizon (40 x 20 ms = 0.8 s)
        self.v_max = 4.0  # max. path speed [m/s]
        self.topt_vf = 0.7  # speed limit = topt_vf * time-optimal profile + 0.3 m/s
        self.tilt_max = MPC_TILT_MAX  # tilt limit of the MPCC (per axis and total) [rad]
        self.tunnel_max = 0.15  # max. tunnel radius in free space [m]
        self.tunnel_min = 0.03
        self.margin = 0.02  # additional distance to frames / poles [m]
        self.gate_w = 0.09  # gate opening in the MPCC: half width for the drone centre [m]
        # ... and half height: the collision box is only +-0.02 m high (vs. +-0.07 m wide), and the
        # plan crosses up to gate_z + plan_lat above the centre
        self.gate_wz = 0.12
        self.gate_win = 0.15  # ... applies in this range around the crossing (arc length) [m]
        self.gate_xwin = 0.12  # ... and only within this distance to the gate plane [m]
        self.gate_tun = 0.10  # minimum tunnel radius there [m]
        self.q_con = 2250.0  # weight contour error
        self.q_con_z = 3000.0  # ... vertical part (the drone sags in tight turns)
        self.q_lag = 1000.0  # weight lag error (keeps s at the drone)
        self.q_prog = 40.0  # weight progress
        self.q_rate = 0.5  # weight tilt rate
        self.q_al = 0.0005  # weight angular acceleration
        self.ds_ref = 10.0  # initial value of the progress reference (overwritten per stage)
        self.ds_add = 0.29  # progress term: target speed = speed limit + ds_add [m/s]
        self.dds_max = 30.0  # max. path acceleration in the model [m/s^2]
        self.k_dist = 0.15  # disturbance observer gain (per step)
        self.k_kap = 0.1  # thrust factor estimation gain (per step)
        self.dist_par = 0.3  # share of the error along the thrust axis that stays additive
        self.dist_max = np.array([1.5, 1.5, 2.0])
        self.takeoff_height = 0.03  # vertical take-off up to start height + this (fixed thrust) [m]
        self.obs_height = 0.05  # disturbance observer active above start height + this [m]
        self.takeoff_f = 0.78  # thrust during take-off [N]
        self.takeoff_dz = 0.04  # state at the hand-over to the MPCC: height above start [m]
        self.takeoff_vz = 0.6  # ... and climb rate [m/s] (start point of the plan)
        self.tun_z = 2000.0  # penalty for leaving the tunnel per m (linear / quadratic)
        self.tun_Z = 1e5
        self.z_min = 0.08  # ground (soft constraint in the MPCC) [m]
        self.rebuild_iter = 2  # extra SQP iterations after a path change

        # ---- Level-3 preset ------------------------------------------------------------------
        # The values above are tuned for speed on the fixed level-2 track. With a random track
        # (level 3, used for the demo) robustness comes first: no per-pass tuning, slower
        # approach of unseen gates, a gentler plan. See CLAUDE.md, decision D2.
        try:
            random_track = bool(config.env.track.get("randomize", False))
        except AttributeError:
            random_track = False
        if random_track:
            self.v_unseen = (2.2, 2.2, 2.2, 2.2, 2.2)
            self.gate_bias = (0.0, 0.0, 0.0, 0.0, 0.0)
            self.v_max = 3.5
            self.plan_t_max = 13.5
            self.plan_tilt = 0.9
            # More room to frames: hairpins around the gate just passed graze its outer frame
            self.plan_r_frame = 0.13
            # Above ~60 deg the motors saturate in hard manoeuvres (yaw is lost, then the drone)
            self.tilt_max = 1.0
            self.push_frame = 0.22

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

        # ---- State ----------------------------------------------------------------------------
        self._tick = 0
        self._last_obs = obs
        self._seq = np.asarray(obs["gate_sequence"], dtype=int)
        self._dirs = np.asarray(obs["gate_sequence_direction"], dtype=float)
        self._start = np.asarray(obs["pos"], dtype=float).copy()
        # The plan is cached across runs: therefore plan from the nominal start position
        # (otherwise the random start position of the first run would fix the plan for all)
        self.plan_start_fix = None  # fixed start point of the plan (else nominal from the config)
        try:
            self._plan_start = np.asarray(config.env.track.drones[0]["pos"], dtype=float)
            if self.plan_start_fix is not None:
                self._plan_start = np.asarray(self.plan_start_fix, dtype=float)
        except (AttributeError, IndexError, KeyError, TypeError):
            self._plan_start = self._start.copy()
        self._airborne = False
        # Arena for the planner: safety limits of the track minus a margin (x, y) / ceiling (z)
        lo, hi = np.array([-2.5, -1.5, 0.0]), np.array([2.5, 1.5, 2.0])
        try:
            lo = np.asarray(config.env.track.safety_limits.pos_limit_low, dtype=float)
            hi = np.asarray(config.env.track.safety_limits.pos_limit_high, dtype=float)
        except (AttributeError, KeyError, TypeError):
            pass
        m = self.arena_margin
        self._bounds = (np.r_[lo[:2] + m, -1.0], np.r_[hi[:2] - m, hi[2] - m])
        self._plan_fallback = False
        self._plan_relaxed = False  # plan from the relaxed second stage
        self._plan_time = 0.0  # computation time of the time-optimal plan (0 = from cache) [s]
        self._obs_on = False
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

        # ---- Planning: time-optimal trajectory (once per track), then path -------------------
        self._nominal(obs)
        self._known = np.r_[obs["gates_visited"], obs["obstacles_visited"]].astype(bool)
        self._build_path(obs)

    # ==================================================================================
    # Initial guess: waypoints through the gates, around poles and frames
    # ==================================================================================
    def _guess_waypoints(
        self,
        obs: dict[str, Any],
        start: NDArray[np.float64],
        d: float = 0.3,
        r_pole: float = 0.25,
        r_frame: float = 0.5,
    ) -> NDArray[np.float64]:
        """Create via points for the initial path through the gates."""
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
        """Move free via points so that the path avoids obstacles."""
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
            # 1) Poles: distance in x-y (poles reach from the ground to 1.55 m)
            dist = np.linalg.norm(pts[:, None, :2] - poles[None, :, :2], axis=2)
            if dist.size:
                i, j = np.unravel_index(dist.argmin(), dist.shape)
                if r_pole - dist[i, j] > worst:
                    away = (pts[i, :2] - poles[j, :2]) / max(dist[i, j], 1e-6)
                    worst, i_worst = r_pole - dist[i, j], i
                    via = np.r_[poles[j, :2] + r_pole * away, pts[i, 2]]
            # 2) Gate frames: path crosses a gate plane beside the opening
            for g in range(len(gates_pos)):
                loc = (pts - gates_pos[g]) @ gates_rot[g]
                off = np.abs(loc[:, 1:]).max(axis=1)
                bad = (np.abs(loc[:, 0]) < 0.1) & (off > 0.12) & (off < r_frame)
                for i in np.where(bad)[0]:
                    if r_frame - off[i] > worst:
                        y, z = loc[i, 1], loc[i, 2]
                        if z < 0 and abs(z) >= abs(y):  # below -> sideways instead (stand!)
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
    # Nominal trajectory (time-optimal) + deformation
    # ==================================================================================
    def _plan_cfg(self) -> dict[str, int | float | tuple[float, ...]]:
        """Collect the parameters of the time-optimal planner."""
        return {
            "n": self.plan_n,
            "t_max": self.plan_t_max,
            "tilt_max": self.plan_tilt,
            "jerk": self.plan_jerk,
            "theta": self.plan_theta,
            "lat": self.plan_lat,
            "r_pole": self.plan_r_pole,
            "r_poles": tuple(
                self.plan_r_pole if seen else self.plan_r_pole + self.pole_unc
                for seen in np.asarray(self._last_obs["obstacles_visited"], dtype=bool)
            ),
            "r_frame": self.plan_r_frame,
            "frame_extra": tuple(
                0.0 if seen else self.frame_unc
                for seen in np.asarray(self._last_obs["gates_visited"], dtype=bool)
            ),
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
            "v_cross": self.v_cross,
            "act_dist": self.plan_act_dist,
            "plug_r": self.plug_r,
            "gap_min": self.gap_min,
            "cpu_max": self.plan_cpu_max,
            "lo": tuple(self._bounds[0]),
            "hi": tuple(self._bounds[1]),
            "cross_nodes": self.cross_nodes,
            "cross_dist": self.cross_dist,
        }

    def _guess_segments(
        self, obs: dict[str, Any], d: float, r_pole: float, r_frame: float
    ) -> list[NDArray[np.float64]] | None:
        """Initial guess: spline through via points, split into segments at the gate planes."""
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
        """Time-optimal trajectory on the nominal track (once per track, then from the cache)."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        cfg = self._plan_cfg()
        # Gates that are seen for the first time at their pass -> speed limit before them
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
                    "v17",
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
            # Several initial guesses (IPOPT only finds local optima), the fastest wins
            plan = None
            for d, r_pole, r_frame in self.plan_starts:
                if plan is not None and time.perf_counter() - t0 > self.plan_budget:
                    break
                guess = self._guess_segments(obs, d, r_pole, r_frame)
                if guess is None:
                    continue
                # The plan starts where the vertical take-off hands over to the MPCC
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
            if plan is None:
                # Second stage before the crude guess path: drop the extra safety constraints
                # (plugs, uncertainty margins) and allow more CPU time
                relaxed = dict(cfg)
                relaxed.update(
                    plug_r=0.0,
                    frame_extra=(0.0,) * len(gates_pos),
                    r_poles=(self.plan_r_pole,) * len(poles),
                    cpu_max=2.0 * self.plan_cpu_max,
                )
                for d, r_pole, r_frame in self.plan_starts[:2]:
                    guess = self._guess_segments(obs, d, r_pole, r_frame)
                    if guess is None:
                        continue
                    plan = _time_optimal(
                        self._plan_start + np.array([0.0, 0.0, self.takeoff_dz]),
                        np.array([0.0, 0.0, self.takeoff_vz]),
                        gates_pos,
                        gates_rot,
                        self._seq,
                        self._dirs,
                        poles,
                        guess,
                        unseen,
                        relaxed,
                    )
                    if plan is not None:
                        self._plan_relaxed = True
                        break
            self._plan_time = time.perf_counter() - t0
            if plan is None:  # fallback: initial guess at 2 m/s
                guess = self._guess_segments(obs, *self.plan_starts[0])
                pts = np.concatenate([guess[0]] + [gp[1:] for gp in guess[1:]])
                plan = {
                    "P": pts,
                    "V": np.gradient(pts, axis=0) * 2.0 / 0.05,
                    "cross": np.arange(1, len(self._seq) + 1) * self.plan_n,
                    "t": np.zeros(len(pts)),
                }
                print("Time-optimal planning failed -> falling back to the guess path")
                self._plan_fallback = True
            else:
                print(
                    f"Time-optimal trajectory: {plan['t'][-1]:.2f} s "
                    f"(computed in {time.perf_counter() - t0:.1f} s)"
                )
                try:
                    np.savez(cache_file, **plan)
                except OSError:
                    pass
            _PLAN_CACHE[key] = plan

        # Resample on a uniform arc-length grid, append the run-out
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
        """Displacement of the nominal path for the currently known positions.

        Weighted least squares per coordinate: smooth (second differences of the displacement)
        plus weighted targets: start, crossing zones (rigid with the gate), path under the drone
        (stays as it is) - all practically hard -, and soft targets that push the path past poles
        and frames (iteratively, the worst point per obstacle).
        """
        q = self._q_nom
        m = len(q)
        gp = np.asarray(obs["gates_pos"], dtype=float)
        rot_new = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        hard, soft = 1e8, 1e4
        w = np.zeros((m, 3))
        tgt = np.zeros((m, 3))
        locked = np.zeros(m, dtype=bool)  # points that must not be pushed away

        def pin(
            k: int,
            val: NDArray[np.float64],
            weight: float,
            mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ) -> None:
            """Set target displacement and weight of a path point."""
            w[k] = weight * np.asarray(mask)
            tgt[k] = val

        pin(0, self._start - q[0], hard)
        locked[0] = True
        n_r = int(round(self.gate_rigid / self.def_ds))
        for i, (g, k_c) in enumerate(zip(self._seq, self._cross_k)):
            r_rel = rot_new[g] @ self._plan_rot[g].T
            # crossing point moves fully with the gate; ignore shifts within the tolerance
            q_full = r_rel @ (q[k_c] - self._plan_gp[g]) + gp[g]
            loc0 = rot_new[g].T @ (q[k_c] - gp[g])  # unshifted point in the new gate frame
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
        if self._airborne:  # do not change the path under the drone (position + direction)
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
        if self.def_dz_max > 0.0:
            # Gate heights vary by only +-0.1 m; larger vertical shifts come from the smoothing
            # overshooting between pushed regions (dives to the floor were seen) -> cap them
            disp[:, 2] = np.clip(disp[:, 2], -self.def_dz_max, self.def_dz_max)
        self._disp = disp
        return q + disp

    def _solve_disp(self, w: NDArray[np.float64], tgt: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the smooth displacement from weighted targets."""
        m = len(w)
        if not hasattr(self, "_lap"):  # L^T L (+ little regularisation), five bands
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
    # Path + tunnel + speed limit
    # ==================================================================================
    def _build_path(self, obs: dict[str, Any]) -> None:
        """Path for the currently known positions, plus tunnel radius and speed limit over s."""
        old = getattr(self, "_path", None)
        p_old = old(self._s) if old is not None else None
        self._s_remap = None
        if old is not None:  # current index of the drone on the deformed path
            self._k_now = int(
                np.clip(np.searchsorted(old.knot_s, self._s), 0, len(self._q_nom) - 2)
            )
        wps = self._deform(obs)
        if self.def_z_floor:
            # Keep the deformed path above the planner's minimum height (after take-off)
            high = self._q_nom[:, 2] >= self.plan_z_min - 1e-3
            wps[high, 2] = np.maximum(wps[high, 2], self.plan_z_min)
        self._path = PathGeo(wps)
        if p_old is not None:
            # Transfer the path parameter: old and new path share the same grid points, so
            # knot k <-> knot k (also for the MPCC prediction)
            self._s_remap = (old.knot_s.copy(), self._path.knot_s.copy())
            self._s = float(np.interp(self._s, *self._s_remap))
            self._extra_iter = self.rebuild_iter

        pts = self._path.pts
        # Distance to all gate frames (collision boxes) and poles
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
        # Crossings on this path (arc length, gate) for the gate window in the MPCC
        self._gate_s = np.array([self._path.knot_s[k] for k in self._cross_k])
        self._gate_ids = list(self._seq)
        self._gates_now = (gates_pos.copy(), gates_rot.copy())

        # Speed limit from the time-optimal profile
        v_nom = self._v_nom
        visited = np.asarray(obs["gates_visited"], dtype=bool)
        # Hard limit before gates not seen yet (the drone still has to correct there)
        cap = np.full(len(v_nom), 100.0)
        n_cap = int(round((self.unseen_len + 0.1) / self.def_ds))
        for i, (g, k_c) in enumerate(zip(self._seq, self._cross_k)):
            if self._unseen_seq[i] is not None and not visited[g]:
                cap[max(k_c - n_cap, 0) : k_c + 1] = self._unseen_seq[i]
        # Also where the path only passes by objects that have not been seen yet: they may be up
        # to ~0.2 m closer than assumed and are detected late (sensor range 0.7 m)
        if self.v_pass > 0:
            near = np.zeros(len(v_nom), dtype=bool)
            o_vis = np.asarray(obs["obstacles_visited"], dtype=bool)
            for o, seen in zip(poles, o_vis):
                if not seen:
                    d = np.linalg.norm(wps[:, :2] - o[:2], axis=1) - POLE_RADIUS
                    near |= d < self.pass_dist
            for g in range(len(gates_pos)):
                if visited[g]:
                    continue
                d = np.min(
                    [
                        _capsule_dist(wps, a, b) - r
                        for a, b, r in _capsules(gates_pos[g], gates_rot[g])
                    ],
                    axis=0,
                )
                near_g = d < self.pass_dist
                for g2, k_c in zip(self._seq, self._cross_k):  # not when flying through g
                    if g2 == g:
                        near_g[max(k_c - n_cap, 0) : k_c + n_cap + 1] = False
                near |= near_g
            cap[near] = np.minimum(cap[near], self.v_pass)
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
        """Compute the MPCC command, or None if the solver failed."""
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
            if getattr(self, "_s_remap", None) is not None:  # the path has changed
                xs[:, 14] = np.interp(xs[:, 14], *self._s_remap)
                self._s_remap = None
            # shift solution and linearisation points by one step
            s_lin = np.r_[xs[1:, 14], xs[-1, 14] + dt * xs[-1, 15]]
            s_lin += s0 - s_lin[0]
        else:
            ds_guess = np.full(n + 1, max(ds0, 0.5))
            s_lin = s0 + dt * np.r_[0.0, np.cumsum(ds_guess[:n])]
        s_lin = np.clip(s_lin, 0.0, path.length)
        p_lin = path(s_lin)
        t_lin = path.tangent(s_lin)
        # take the smaller of the two neighbouring values (no optimistic interpolation)
        idx = np.clip(np.searchsorted(path.ss, s_lin, side="right") - 1, 0, len(path.ss) - 1)
        nxt = np.minimum(idx + 1, len(path.ss) - 1)
        r_tun = np.minimum(self._tunnel[idx], self._tunnel[nxt])
        vmax = np.minimum(self._vmax[idx], self._vmax[nxt])
        # Gate opening instead of the tunnel near the crossings
        gate_par = np.tile(np.r_[np.zeros(3), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10.0, 10.0], (n + 1, 1))
        g_pos, g_rot = self._gates_now
        for s_c, g in zip(self._gate_s, self._gate_ids):
            # Only where the reference point is near the gate plane: the window is a slab, so far
            # behind a sharp exit turn it would contradict the tunnel and stall the drone
            x_loc = (p_lin - g_pos[g]) @ g_rot[g][:, 0]
            act = (np.abs(s_lin - s_c) < self.gate_win) & (np.abs(x_loc) < self.gate_xwin)
            if np.any(act):
                c = g_pos[g] - np.array([0.0, 0.0, 0.02])  # the collision sphere sits 2 cm higher
                gate_par[act] = np.r_[c, g_rot[g][:, 1], g_rot[g][:, 2], self.gate_w, self.gate_wz]
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
        # Thrust limit follows the thrust factor: heavy drones get the same acceleration
        f_max = float(np.clip(MPC_F_MAX / self._kap, MPC_F_MAX, MPC_F_ABS))
        # Tilt limit at runtime (<= MPC_TILT_MAX, which is built into the model): per axis as a box
        # bound, the total tilt via the upper bound of its (soft) nonlinear constraint
        tm = min(self.tilt_max, MPC_TILT_MAX)
        uh_tilt = np.cos(MPC_TILT_MAX) - np.cos(tm)
        for j in range(1, n + 1):
            sol.constraints_set(j, "lbx", np.array([self.z_min, -tm, -tm, MPC_F_MIN, 0.0]))
            sol.constraints_set(j, "ubx", np.array([2.4, tm, tm, f_max, vmax[j]]))
            n_h = 5 if j == n else 7
            uh = np.zeros(n_h)
            uh[2] = uh_tilt
            sol.constraints_set(j, "uh", uh)
        if self._mpc_warm:  # shift the last solution by one step
            for j in range(n):
                x = xs[min(j + 1, n)].copy()
                x[14] = s_lin[j]
                sol.set(j, "x", x)
                sol.set(j, "u", us[min(j + 1, n - 1)])
            x = xs[n].copy()
            x[14] = s_lin[n]
            sol.set(n, "x", x)
        else:  # cold start along the path
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
        for _ in range(getattr(self, "_extra_iter", 0)):  # iterate more after a path change
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
        # Invert the inner loop: the middle of the control interval is the reference
        u0 = sol.get(0, "u")
        tilt_m = 0.5 * (tilt + x1[3:5])
        rate_m = 0.5 * (dtilt + x1[8:10])
        al_m = 0.5 * (x0[10:12] + x1[10:12])
        tilt_cmd = np.clip(
            tilt_m + (u0[:2] + MPC_KD * al_m) / MPC_K1, -MPC_CMD_TILT_MAX, MPC_CMD_TILT_MAX
        )
        # Body rates at fixed yaw from the tilt rates (R = Ry(pitch) Rx(roll))
        cr, sr = np.cos(tilt_m[0]), np.sin(tilt_m[0])
        rate_b = np.array([rate_m[0], rate_m[1] * cr, -rate_m[1] * sr])
        return tilt_cmd, self._f_cmd, rate_b

    @staticmethod
    def _attitude_to_acc(
        tilt_cmd: NDArray[np.float64], f_cmd: float, quat: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Acceleration set point for which the Mellinger commands this attitude + thrust."""
        z_des = _tilt_axis(tilt_cmd)
        z_cur = Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])
        cos = max(float(z_des @ z_cur), 0.5)  # the Mellinger projects onto the current z-axis
        return f_cmd / (MELLINGER_F_PER_ACC * cos) * z_des - GRAVITY

    def _mellinger_cmd(
        self, p_ref: NDArray[np.float64], v_ref: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        """Fallback / start: state set point to the Mellinger, keep the MPC thrust state updated."""
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
        """Compute the command from the current observation."""
        p = np.asarray(obs["pos"], dtype=float)
        v = np.asarray(obs["vel"], dtype=float)
        quat = np.asarray(obs["quat"], dtype=float)
        ang_vel = np.asarray(obs["ang_vel"], dtype=float)

        # Track the path parameter, but (almost) only forward: otherwise the reference point
        # follows a drone that falls behind and the MPCC chases it backwards
        self._s = self._path.project(p, self._s, back=0.05, ahead=0.8)
        self._last_ref = self._path(self._s)  # only for evaluation

        # Relative to the (random) start height: on the ground the MPCC would already tilt the
        # drone towards the climbing path and it would lift off sideways
        self._airborne |= p[2] > self._start[2] + self.takeoff_height
        if not self._airborne:  # vertical take-off: measurement as set point, fixed thrust
            self._v_pred = None
            f_cmd = self.takeoff_f
            acc = np.array([0.0, 0.0, f_cmd / MELLINGER_F_PER_ACC]) - GRAVITY
            self._f_est = f_cmd + (self._f_est - f_cmd) * np.exp(-1.0 / (self._freq * MPC_TAU))
            self._f_cmd = f_cmd
            return np.concatenate((p, v, acc, [0.0, 0.0, 0.0, 1.0], np.zeros(3))).astype(np.float32)

        z_up = float(Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])[2])
        if z_up < np.cos(MPC_RESCUE_TILT):  # the model is not valid here: level the drone first
            self._v_pred = None
            self._mpc_warm = False
            self._al = np.zeros(2)
            self._f_cmd = 0.6
            self._f_est = 0.6 + (self._f_est - 0.6) * np.exp(-1.0 / (self._freq * MPC_TAU))
            # At most 1 rad at once: above 90 degrees of attitude error the onboard controller
            # switches the torques off
            z_cur = Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])
            ang = float(np.arccos(np.clip(z_up, -1.0, 1.0)))
            z_des = (np.sin(ang - 1.0) * z_cur + np.sin(1.0) * np.array([0.0, 0.0, 1.0])) / max(
                np.sin(ang), 0.1
            )
            z_des /= np.linalg.norm(z_des)
            a = self._f_cmd / MELLINGER_F_PER_ACC * z_des - GRAVITY
            return np.concatenate((p, v, a, [0.0, 0.0, 0.0, 1.0], np.zeros(3))).astype(np.float32)

        # The observer only runs in free flight: on the ground, contact forces would be learned as
        # a disturbance (the start height is random, so the drone may count as airborne at once)
        self._obs_on |= p[2] > self._start[2] + self.obs_height
        if self._v_pred is not None and self._obs_on:
            # Observer: error of the velocity prediction -> along the thrust axis mainly the
            # thrust factor (mass), otherwise an additive disturbance acceleration
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
        # Measurement as set point: the Mellinger only realises attitude / thrust
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
        """Update the statistics and adapt the path to new sensor data."""
        self._tick += 1
        self._speed_sum += float(np.linalg.norm(obs["vel"]))
        self._speed_samples += 1
        self._last_obs = obs

        # Deform the path as soon as a gate or pole is seen for the first time
        known = np.r_[obs["gates_visited"], obs["obstacles_visited"]].astype(bool)
        if np.any(known & ~self._known):
            self._known = known
            self._build_path(obs)
        return False

    def episode_callback(self) -> None:
        """Print the diagnosis and metrics of the finished episode."""
        o = self._last_obs
        pos = np.asarray(o["pos"])
        poles = np.asarray(o["obstacles_pos"])
        d_pole = np.linalg.norm(poles[:, :2] - pos[:2], axis=1).min() if len(poles) else np.inf
        d_gate = np.linalg.norm(np.asarray(o["gates_pos"]) - pos, axis=1).min()
        cause = "pole" if d_pole < 0.12 else "gate frame" if d_gate < 0.5 else "other"
        print(
            f"END t={self._tick / self._freq:.2f}s  Gates={int(o['n_gates_passed'])}  "
            f"pos={np.round(pos, 2)}  pole {d_pole * 100:.0f} cm  Gate {d_gate * 100:.0f} cm  "
            f"-> {cause}  (MPCC failures: {self._mpc_fail})"
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
        """Draw the current path in the simulation."""
        from crazyflow.sim.visualize import draw_line

        draw_line(sim, self._path.pts[::5], rgba=(0.0, 1.0, 0.0, 1.0))

    def episode_reset(self) -> None:
        """Reset the episode statistics and the MPCC warm start."""
        self._tick = 0
        self._solve_ms.clear()
        self._speed_sum = 0.0
        self._speed_samples = 0
        self._mpc_warm = False
        self._v_pred = None
        self._mpc_fail = 0
