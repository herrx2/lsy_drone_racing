from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# Der Mellinger setzt die Soll-Beschleunigung ~1,46-fach um (falsche Masse im Bordregler)
THRUST_SCALE = 1.611 * 0.0393 / 0.0434
GRAVITY = np.array([0.0, 0.0, 9.81])


class MyController(Controller):
    """Our own approach to control the Drone."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Grenzen für die Trajektorie (pro Achse)
        self.v_slow = 0.8  # [m/s] in der Nähe noch nicht gesehener Gates/Stangen
        self.v_fast = 1.8  # [m/s] überall sonst
        self.a_max =  1.8  # [m/s^2]
        self.r_pole = 0.3  # Mindestabstand der Bahn zu den Stangen [m]

        # MPC: Punktmassen-Modell x = [p, v], u = a, Horizont N Schritte
        self.mpc_n, dt = 20, 1.0 / self._freq
        A = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        B = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
        # Vorhersage aller Zustände: X = Phi @ x0 + Gam @ U
        self._Phi = np.vstack([np.linalg.matrix_power(A, k + 1) for k in range(self.mpc_n)])
        self._Gam = np.zeros((6 * self.mpc_n, 3 * self.mpc_n))
        for k in range(self.mpc_n):
            for j in range(k + 1):
                self._Gam[6 * k:6 * k + 6, 3 * j:3 * j + 3] = np.linalg.matrix_power(A, k - j) @ B
        Q = np.kron(np.eye(self.mpc_n), np.diag([100.0] * 3 + [10.0] * 3))  # Position/Geschw.
        R = 0.1 * np.eye(3 * self.mpc_n)  # Beschleunigung
        self._K = np.linalg.solve(self._Gam.T @ Q @ self._Gam + R, self._Gam.T @ Q)

        self._d = self._optimizing(obs)
        self._tick = 0
        self._traj = None
        self._replan(obs)  # erste Bahn, schon um die Hindernisse herum

        self._known = np.r_[obs["gates_visited"], obs["obstacles_visited"]]
        self._finished = False

    def _waypoints(self, obs, d: np.ndarray) -> np.ndarray:
        """Start, dann pro Gate: Punkt davor, Gate-Mitte, Punkt dahinter (Abstand d[k])."""
        start_pos = np.asarray(obs["pos"], dtype=float)
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        seq = np.asarray(obs["gate_sequence"], dtype=int)
        dirs = np.asarray(obs["gate_sequence_direction"], dtype=float)

        waypoints = [start_pos]
        for g, s, d_k in zip(seq, dirs, d):
            center = gates_pos[g]
            normal = gates_rot[g][:, 0] * s
            waypoints += [center - d_k * normal, center, center + d_k * normal]
        waypoints.append(center + 1.0 * normal)  # Auslauf: durch das letzte Gate ohne zu bremsen
        return np.array(waypoints)

    def _plan_trajectory(self, waypoints: np.ndarray, obs, v_start=None) -> CubicSpline:
        """Spline über die Zeit: Abschnittsdauer = Länge / Tempo, wo nötig gestreckt."""
        seg_len = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        v_lim = self._speed_limit(waypoints, obs)
        dt = np.maximum(seg_len / v_lim, 0.02)
        start = np.zeros(3) if v_start is None else np.asarray(v_start, dtype=float)
        bc = ((1, start), (1, np.zeros(3)))  # Start mit aktueller Geschwindigkeit, Ende in Ruhe

        # Abschnitte, in denen Geschwindigkeit oder Beschleunigung zu groß ist, zeitlich strecken
        # (Beschleunigung ~ 1/Dauer^2, Geschwindigkeit ~ 1/Dauer)
        for _ in range(30):
            t = np.r_[0.0, np.cumsum(dt)]
            spline = CubicSpline(t, waypoints, bc_type=bc)
            ts = np.linspace(0.0, t[-1], 20 * len(dt))
            seg = np.clip(np.searchsorted(t, ts, side="right") - 1, 0, len(dt) - 1)
            v_seg, a_seg = np.zeros(len(dt)), np.zeros(len(dt))
            np.maximum.at(v_seg, seg, np.abs(spline(ts, 1)).max(axis=1))
            np.maximum.at(a_seg, seg, np.abs(spline(ts, 2)).max(axis=1))
            stretch = np.maximum(v_seg / v_lim, np.sqrt(a_seg / self.a_max))
            if stretch.max() < 1.02:
                break
            dt *= np.clip(stretch, 1.0, 1.5)
        return spline

    def _speed_limit(self, waypoints: np.ndarray, obs) -> np.ndarray:
        """Tempo pro Abschnitt: langsam nahe Gates/Stangen, deren echte Lage noch unbekannt ist."""
        unseen = np.r_[
            np.asarray(obs["gates_pos"])[~np.asarray(obs["gates_visited"], dtype=bool)],
            np.asarray(obs["obstacles_pos"])[~np.asarray(obs["obstacles_visited"], dtype=bool)],
        ][:, :2]
        mid = 0.5 * (waypoints[:-1, :2] + waypoints[1:, :2])  # Mitte jedes Abschnitts
        if len(unseen) == 0:
            return np.full(len(mid), self.v_fast)
        near = np.linalg.norm(mid[:, None] - unseen[None], axis=2).min(axis=1) < 0.8
        return np.where(near, self.v_slow, self.v_fast)

    def _optimizing(self, obs) -> np.ndarray:
        """Abstand der Hilfspunkte pro Gate so wählen, dass die Trajektorie am kürzesten dauert."""
        n = len(obs["gate_sequence"])

        def duration(d):
            try:
                return self._plan_trajectory(self._waypoints(obs, d), obs).x[-1]
            except ValueError:
                return 1e3  # ungültige Wegpunkte bestrafen

        res = minimize(duration, np.full(n, 0.3), method="Powell", bounds=[(0.25, 0.5)] * n,
                       options={"maxfev": 200})
        return res.x

    def _replan(self, obs):
        """Bahn ab dem aktuellen Sollpunkt neu planen, um Stangen und Gate-Rahmen herum."""
        first = int(obs["n_gates_passed"])
        if first < 0 or first >= len(obs["gate_sequence"]):
            return  # alle Gates durch
        if self._traj is None:  # Start
            pos, vel = np.asarray(obs["pos"], dtype=float), np.zeros(3)
        else:
            t = min(self._tick / self._freq - self._t0, self._traj.x[-1])
            pos, vel = self._traj(t), self._traj(t, 1)

        # 1. Wegpunkte der noch offenen Gates (Vor-Punkt nur, wenn er noch vor uns liegt)
        wps = self._waypoints(obs, self._d)
        wps = np.vstack([pos, wps[1 + 3 * first:]])
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = Rotation.from_quat(obs["gates_quat"]).as_matrix()
        g = obs["gate_sequence"][first]
        if np.dot(wps[1] - pos, gates_rot[g][:, 0] * obs["gate_sequence_direction"][first]) < 0.05:
            wps = np.delete(wps, 1, axis=0)

        # 2. Hilfspunkte (nicht Gate-Mitten) aus der Nähe von Stangen wegschieben
        poles = np.asarray(obs["obstacles_pos"], dtype=float)[:, :2]
        for i in range(1, len(wps)):
            dp = wps[i, :2] - poles
            dd = np.linalg.norm(dp, axis=1)
            j = dd.argmin()
            if dd[j] < self.r_pole + 0.1 and np.linalg.norm(gates_pos - wps[i], axis=1).min() > 0.01:
                wps[i, :2] = poles[j] + (self.r_pole + 0.1) * dp[j] / dd[j]

        # 3. Jede Strecke prüfen und bei Bedarf einen Umweg-Punkt einfügen
        k = 0
        while k < len(wps) - 1 and len(wps) < 60:
            a, b = wps[k], wps[k + 1]
            via = None

            # Stange zu nah -> Punkt neben die Stange, auf der Außenseite der Kurve
            ab = b[:2] - a[:2]
            u = np.clip((poles - a[:2]) @ ab / max(ab @ ab, 1e-9), 0.0, 1.0)
            dist = np.linalg.norm(a[:2] + u[:, None] * ab - poles, axis=1)
            j = dist.argmin()
            if dist[j] < self.r_pole and 0.0 < u[j] < 1.0:
                q = a + u[j] * (b - a)
                side = (q[:2] - poles[j]) / max(dist[j], 1e-6)
                if k > 0:
                    prev = a[:2] - wps[k - 1][:2]
                    right = np.array([ab[1], -ab[0]]) / np.linalg.norm(ab)
                    side = right if prev[0] * ab[1] - prev[1] * ab[0] > 0 else -right
                via = [np.r_[poles[j] + (self.r_pole + 0.05) * side, q[2]]]

            # Strecke kreuzt eine Gate-Ebene im Rahmen -> erst seitlich neben den Rahmen, dann durch
            for g in range(len(gates_pos)):
                la, lb = (a - gates_pos[g]) @ gates_rot[g], (b - gates_pos[g]) @ gates_rot[g]
                if la[0] * lb[0] < 0:
                    lq = la + la[0] / (la[0] - lb[0]) * (lb - la)
                    lq[2] = max(lq[2], 0.0)  # nie unten herum (Ständer)
                    off = np.abs(lq[1:]).max()
                    if 0.1 < off < 0.6:
                        lq[1:] *= 0.7 / off
                        side_pt = np.r_[0.25 * np.sign(la[0]), lq[1:]]
                        via = [gates_pos[g] + gates_rot[g] @ p for p in (side_pt, lq * [0, 1, 1])]

            if via is None:
                k += 1
            else:
                wps = np.vstack([wps[: k + 1], via, wps[k + 1 :]])

        # 4. Lange Strecken unterteilen, damit die Spline nicht weit ausschwingt
        dense = [wps[0]]
        for a, b in zip(wps[:-1], wps[1:]):
            n = int(np.ceil(np.linalg.norm(b - a) / 0.3))
            dense += [a + (b - a) * i / n for i in range(1, n + 1)]

        # 5. Trajektorie: startet mit der aktuellen Geschwindigkeit
        try:
            self._traj = self._plan_trajectory(np.array(dense), obs, vel)
            self._t0 = self._tick / self._freq
        except ValueError:
            pass  # (z.B. doppelte Punkte) -> alte Bahn weiterfliegen

    def _mpc(self, pos, vel, t):
        """Beschleunigung, mit der die Drohne in den nächsten 0.4 s der Spline am besten folgt."""
        ts = np.minimum(t + np.arange(1, self.mpc_n + 1) / self._freq, self._traj.x[-1])
        x_ref = np.hstack([self._traj(ts), self._traj(ts, 1)]).ravel()
        x0 = np.r_[pos, vel]
        u = self._K @ (x_ref - self._Phi @ x0)
        return np.clip(u[:3], -self.a_max, self.a_max)  # nur der erste Schritt wird verwendet

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Sollzustand aus der Trajektorie zur aktuellen Zeit."""
        t = self._tick / self._freq - self._t0
        self._finished = t > self._traj.x[-1] + 1.0  # 1 s Reserve, falls die Drohne nachhängt
        t = min(t, self._traj.x[-1])

        pos = self._traj(t)
        vel = self._traj(t, 1)
        acc = self._mpc(np.asarray(obs["pos"]), np.asarray(obs["vel"]), t)
        acc = (acc + GRAVITY) / THRUST_SCALE - GRAVITY  # Schubskalierung ausgleichen
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Yaw = 0
        return np.concatenate((pos, vel, acc, quat, np.zeros(3)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Zeitschritt weiterzählen; neu planen, sobald ein Gate oder eine Stange sichtbar wird."""
        self._tick += 1
        known = np.r_[obs["gates_visited"], obs["obstacles_visited"]]
        if np.any(known != self._known):
            self._known = known
            self._replan(obs)
        return self._finished

    def episode_callback(self):
        """Zähler für die nächste Episode zurücksetzen."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Geplante Trajektorie grün einzeichnen."""
        t = np.linspace(0.0, self._traj.x[-1], 100)
        draw_line(sim, self._traj(t), rgba=(0.0, 1.0, 0.0, 1.0))
