# src/sbc/gui/dashboard.py

import tkinter as tk
from tkinter import ttk
import queue
import numpy as np
from typing import Callable, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle


class TelemetryDashboard(tk.Tk):
    """
    High-fidelity real-time telemetry cockpit for the Stewart-Ball Control (SBC v30) system.
    Provides 4 tabs: 2D Map, Tracking Error, Raw vs Filtered Perception, and Platform Dynamics
    (including full 3D platform origin translation and linear velocity).
    """

    def __init__(
        self,
        telemetry_queue: queue.Queue,
        command_callback: Callable[[dict], None],
        plate_radius: float = 0.25,
        max_history: int = 350
    ) -> None:
        super().__init__()
        self._ui_initialized: bool = False
        self.queue = telemetry_queue
        self.cmd_callback = command_callback
        self.R_plate = plate_radius
        self.max_len = max_history

        self.title("SBC v30 - Stewart Platform Telemetry Cockpit")
        self.geometry("1480x920")
        self.minsize(1180, 740)

        self._setup_styles()

        # Rolling history buffers
        self.t_hist = []
        self.bx_hist, self.by_hist = [], []
        self.rx_hist, self.ry_hist = [], []
        self.bx_raw_hist, self.by_raw_hist = [], []
        self.vx_hist, self.vy_hist = [], []
        self.err_hist = []
        self.roll_hist = []
        self.pitch_hist = []
        self.yaw_hist = []
        self.force_hist = []
        
        # Platform origin translation and velocity history
        self.tx_hist, self.ty_hist, self.tz_hist = [], [], []
        self.tvx_hist, self.tvy_hist, self.tvz_hist = [], [], []

        # Build UI
        self._build_sidebar()
        self._build_central_notebook()

        self._ui_initialized = True
        self._on_mode_switched()
        self._poll_telemetry()

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        self.bg_color = "#1E1E24"
        self.card_bg = "#2B2D42"
        self.text_color = "#EDF2F4"
        self.accent_red = "#EF233C"
        self.accent_blue = "#4895EF"
        self.accent_green = "#06D6A0"
        self.accent_yellow = "#FFD166"

        self.configure(bg=self.bg_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="flat")
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"), foreground=self.accent_blue, background=self.card_bg)
        style.configure("Metric.TLabel", font=("Consolas", 10, "bold"), background=self.card_bg, foreground=self.text_color)
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

    def _build_sidebar(self) -> None:
        sidebar = ttk.Frame(self, width=350, padding=12)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="STEWART-BALL CONTROL", font=("Segoe UI", 14, "bold"), foreground=self.text_color).pack(anchor=tk.W)
        ttk.Label(sidebar, text="SBC v30 - High-Speed Robust Tracking", font=("Segoe UI", 8), foreground="#8D99AE").pack(anchor=tk.W, pady=(0, 6))

        # Card 1: Safety & System State
        card_status = ttk.Frame(sidebar, style="Card.TFrame", padding=10)
        card_status.pack(fill=tk.X, pady=4)
        ttk.Label(card_status, text="SYSTEM STATUS & SAFETY", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.lbl_safety = ttk.Label(card_status, text="Safety Filter: SOCP [OK]", font=("Segoe UI", 10, "bold"), foreground=self.accent_green, background=self.card_bg)
        self.lbl_safety.pack(anchor=tk.W, pady=1)

        self.lbl_contact = ttk.Label(card_status, text="Normal Force: 0.642 N [TOUCH]", style="Card.TLabel")
        self.lbl_contact.pack(anchor=tk.W, pady=1)

        # Card 2: Numeric Metrics
        card_metrics = ttk.Frame(sidebar, style="Card.TFrame", padding=10)
        card_metrics.pack(fill=tk.X, pady=4)
        ttk.Label(card_metrics, text="LIVE KINEMATICS", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.lbl_ball = ttk.Label(card_metrics, text="Ball (x, y): [+0.000, +0.000] m", style="Metric.TLabel")
        self.lbl_ball.pack(anchor=tk.W, pady=1)

        self.lbl_ref = ttk.Label(card_metrics, text="Ref  (x, y): [+0.000, +0.000] m", style="Metric.TLabel")
        self.lbl_ref.pack(anchor=tk.W, pady=1)

        self.lbl_err = ttk.Label(card_metrics, text="Tracking Err:  0.00 mm", font=("Consolas", 12, "bold"), foreground=self.accent_green, background=self.card_bg)
        self.lbl_err.pack(anchor=tk.W, pady=2)

        self.lbl_tilt = ttk.Label(card_metrics, text="Tilt (R/P/Y): [ +0.0°, +0.0°, +0.0°]", style="Metric.TLabel")
        self.lbl_tilt.pack(anchor=tk.W, pady=1)

        self.lbl_plat_pos = ttk.Label(card_metrics, text="Plat Pos:   [ +0.0,  +0.0,  +0.0] mm", style="Metric.TLabel")
        self.lbl_plat_pos.pack(anchor=tk.W, pady=1)

        self.lbl_plat_vel = ttk.Label(card_metrics, text="Plat Vel:   [ +0.0,  +0.0,  +0.0] mm/s", style="Metric.TLabel")
        self.lbl_plat_vel.pack(anchor=tk.W, pady=1)

        # Card 3: Interactive Trajectory Generator
        card_traj = ttk.Frame(sidebar, style="Card.TFrame", padding=10)
        card_traj.pack(fill=tk.X, pady=4)
        ttk.Label(card_traj, text="TRAJECTORY GENERATOR", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(card_traj, text="Mode:", style="Card.TLabel").pack(anchor=tk.W)
        self.cmb_mode = ttk.Combobox(
            card_traj,
            values=["Circle Orbit", "Static Point", "Lissajous 1:2 (Figure 8)", "Lissajous 2:3"],
            state="readonly"
        )
        self.cmb_mode.set("Lissajous 2:3")
        self.cmb_mode.pack(fill=tk.X, pady=2)
        self.cmb_mode.bind("<<ComboboxSelected>>", self._on_mode_switched)

        self.frame_dynamic_ctrls = ttk.Frame(card_traj, style="Card.TFrame")
        self.frame_dynamic_ctrls.pack(fill=tk.X, pady=2)

        self.lbl_p1 = ttk.Label(self.frame_dynamic_ctrls, text="Param 1:", style="Card.TLabel")
        self.scale_p1 = ttk.Scale(self.frame_dynamic_ctrls, from_=0.02, to=0.16, orient=tk.HORIZONTAL, command=self._on_param_changed)
        self.scale_p1.set(0.08)
        self.lbl_val_p1 = ttk.Label(self.frame_dynamic_ctrls, text="", style="Card.TLabel")

        self.lbl_p2 = ttk.Label(self.frame_dynamic_ctrls, text="Param 2:", style="Card.TLabel")
        self.scale_p2 = ttk.Scale(self.frame_dynamic_ctrls, from_=0.2, to=2.5, orient=tk.HORIZONTAL, command=self._on_param_changed)
        self.scale_p2.set(0.75)
        self.lbl_val_p2 = ttk.Label(self.frame_dynamic_ctrls, text="", style="Card.TLabel")

        btn_box = ttk.Frame(sidebar)
        btn_box.pack(side=tk.BOTTOM, fill=tk.X, pady=8)
        ttk.Button(btn_box, text="Center Ball to Origin [0, 0]", command=self._on_cmd_center).pack(fill=tk.X, pady=2)
        ttk.Button(btn_box, text="Start / Resume Trajectory", command=self._on_cmd_orbit).pack(fill=tk.X, pady=2)

    def _render_trajectory_controls(self) -> None:
        mode = self.cmb_mode.get()
        self.lbl_p1.pack_forget()
        self.scale_p1.pack_forget()
        self.lbl_val_p1.pack_forget()
        self.lbl_p2.pack_forget()
        self.scale_p2.pack_forget()
        self.lbl_val_p2.pack_forget()

        if mode == "Static Point":
            self.lbl_p1.config(text="Target X [m]:")
            self.scale_p1.config(from_=-0.15, to=0.15)
            self.lbl_p2.config(text="Target Y [m]:")
            self.scale_p2.config(from_=-0.15, to=0.15)
        elif "Lissajous" in mode:
            self.lbl_p1.config(text="Lissajous Amplitude R [m]:")
            self.scale_p1.config(from_=0.03, to=0.15)
            self.lbl_p2.config(text="Base Frequency w [rad/s]:")
            self.scale_p2.config(from_=0.2, to=2.0)
        else:
            self.lbl_p1.config(text="Orbit Radius R [m]:")
            self.scale_p1.config(from_=0.02, to=0.18)
            self.lbl_p2.config(text="Orbit Speed w [rad/s]:")
            self.scale_p2.config(from_=0.4, to=3.0)

        self.lbl_p1.pack(anchor=tk.W)
        self.scale_p1.pack(fill=tk.X)
        self.lbl_val_p1.pack(anchor=tk.E)
        self.lbl_p2.pack(anchor=tk.W, pady=(4, 0))
        self.scale_p2.pack(fill=tk.X)
        self.lbl_val_p2.pack(anchor=tk.E)

    def _build_central_notebook(self) -> None:
        container = ttk.Frame(self, padding=6)
        container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_map = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_map, text="  Plate 2D Overhead Map  ")

        self.tab_tracking = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tracking, text="  Tracking Performance  ")

        self.tab_perception = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_perception, text="  Perception: Raw vs Filtered  ")

        self.tab_platform = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_platform, text="  Platform: Tilts, Origin & Velocity  ")

        self._setup_map_plot()
        self._setup_tracking_plot()
        self._setup_perception_plot()
        self._setup_platform_plot()

    def _setup_map_plot(self) -> None:
        self.fig_map = plt.Figure(figsize=(6, 6), facecolor="#242633", dpi=100)
        self.ax_map = self.fig_map.add_subplot(1, 1, 1)
        self.ax_map.set_facecolor("#1A1B24")
        self.ax_map.set_aspect("equal")

        lim = self.R_plate + 0.04
        self.ax_map.set_xlim(-lim, lim)
        self.ax_map.set_ylim(-lim, lim)
        self.ax_map.grid(True, color="#383A4D", linestyle="--", alpha=0.6)
        self.ax_map.set_xlabel("Platform X [m]", color="#A0A5B5", fontsize=9)
        self.ax_map.set_ylabel("Platform Y [m]", color="#A0A5B5", fontsize=9)
        self.ax_map.tick_params(colors="#A0A5B5", labelsize=8)

        plate_circle = Circle((0, 0), self.R_plate, color=self.accent_red, fill=False, linewidth=2.0, linestyle="-", label=f"Boundary (R={self.R_plate}m)")
        self.ax_map.add_patch(plate_circle)

        self.line_trail, = self.ax_map.plot([], [], color=self.accent_blue, linewidth=1.4, alpha=0.75, label="Ball Trail")
        self.line_ref_orbit, = self.ax_map.plot([], [], color=self.accent_green, linestyle="--", linewidth=1.5, label="Reference Path")
        self.marker_ball, = self.ax_map.plot([], [], "o", color=self.accent_red, markersize=13, markeredgecolor="#FFFFFF", markeredgewidth=1.5, label="Ball Position")
        self.marker_raw_cop, = self.ax_map.plot([], [], "x", color=self.accent_yellow, markersize=8, alpha=0.6, label="Raw CoP")
        self.marker_ref, = self.ax_map.plot([], [], "X", color=self.accent_green, markersize=11, markeredgecolor="#FFFFFF", label="Target Point")

        self.ax_map.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=8)
        self.fig_map.tight_layout()

        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.tab_map)
        self.canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_tracking_plot(self) -> None:
        self.fig_track = plt.Figure(figsize=(7, 6), facecolor="#242633", dpi=100)
        self.fig_track.subplots_adjust(hspace=0.32, left=0.10, right=0.95, top=0.95, bottom=0.08)

        self.ax_err = self.fig_track.add_subplot(2, 1, 1)
        self.ax_err.set_facecolor("#1A1B24")
        self.ax_err.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_err, = self.ax_err.plot([], [], color=self.accent_red, linewidth=1.8, label="Tracking Error ||e||")
        self.ax_err.set_ylabel("Error [mm]", color="#A0A5B5", fontsize=9)
        self.ax_err.tick_params(colors="#A0A5B5", labelsize=8)
        self.ax_err.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7)

        self.ax_pos = self.fig_track.add_subplot(2, 1, 2)
        self.ax_pos.set_facecolor("#1A1B24")
        self.ax_pos.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_bx, = self.ax_pos.plot([], [], color=self.accent_blue, linewidth=1.4, label="Ball x")
        self.line_rx, = self.ax_pos.plot([], [], color=self.accent_blue, linestyle="--", linewidth=1.2, label="Ref x")
        self.line_by, = self.ax_pos.plot([], [], color="#F72585", linewidth=1.4, label="Ball y")
        self.line_ry, = self.ax_pos.plot([], [], color="#F72585", linestyle="--", linewidth=1.2, label="Ref y")
        self.ax_pos.set_ylabel("Coordinates [m]", color="#A0A5B5", fontsize=9)
        self.ax_pos.set_xlabel("Time [s]", color="#A0A5B5", fontsize=9)
        self.ax_pos.tick_params(colors="#A0A5B5", labelsize=8)
        self.ax_pos.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7, ncol=2)

        self.canvas_track = FigureCanvasTkAgg(self.fig_track, master=self.tab_tracking)
        self.canvas_track.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_perception_plot(self) -> None:
        self.fig_perc = plt.Figure(figsize=(7, 6), facecolor="#242633", dpi=100)
        self.fig_perc.subplots_adjust(hspace=0.32, left=0.10, right=0.95, top=0.95, bottom=0.08)

        self.ax_cop = self.fig_perc.add_subplot(2, 1, 1)
        self.ax_cop.set_facecolor("#1A1B24")
        self.ax_cop.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_raw_x, = self.ax_cop.plot([], [], color=self.accent_yellow, alpha=0.6, label="Raw CoP x")
        self.line_clean_x, = self.ax_cop.plot([], [], color=self.accent_blue, linewidth=1.4, label="Comp Rho x")
        self.line_raw_y, = self.ax_cop.plot([], [], color="#FFAA00", alpha=0.6, label="Raw CoP y")
        self.line_clean_y, = self.ax_cop.plot([], [], color="#F72585", linewidth=1.4, label="Comp Rho y")
        self.ax_cop.set_ylabel("Position [m]", color="#A0A5B5", fontsize=9)
        self.ax_cop.tick_params(colors="#A0A5B5", labelsize=8)
        self.ax_cop.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7, ncol=2)

        self.ax_vel = self.fig_perc.add_subplot(2, 1, 2)
        self.ax_vel.set_facecolor("#1A1B24")
        self.ax_vel.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_vx, = self.ax_vel.plot([], [], color=self.accent_green, linewidth=1.4, label="Est Vel vx (Hermite-GP)")
        self.line_vy, = self.ax_vel.plot([], [], color="#4CC9F0", linewidth=1.4, label="Est Vel vy (Hermite-GP)")
        self.ax_vel.set_ylabel("Velocity [m/s]", color="#A0A5B5", fontsize=9)
        self.ax_vel.set_xlabel("Time [s]", color="#A0A5B5", fontsize=9)
        self.ax_vel.tick_params(colors="#A0A5B5", labelsize=8)
        self.ax_vel.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7)

        self.canvas_perc = FigureCanvasTkAgg(self.fig_perc, master=self.tab_perception)
        self.canvas_perc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_platform_plot(self) -> None:
        """Configures Tab 4 with Tilts, Platform Origin Displacement, and Linear Velocity."""
        self.fig_plat = plt.Figure(figsize=(7, 6), facecolor="#242633", dpi=100)
        self.fig_plat.subplots_adjust(hspace=0.38, left=0.10, right=0.95, top=0.95, bottom=0.08)

        # 1. Tilts
        self.ax_tilt = self.fig_plat.add_subplot(3, 1, 1)
        self.ax_tilt.set_facecolor("#1A1B24")
        self.ax_tilt.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_roll, = self.ax_tilt.plot([], [], color=self.accent_blue, linewidth=1.5, label="Roll (phi)")
        self.line_pitch, = self.ax_tilt.plot([], [], color="#F72585", linewidth=1.5, label="Pitch (theta)")
        self.line_yaw, = self.ax_tilt.plot([], [], color=self.accent_yellow, linewidth=1.2, label="Yaw (psi)")
        self.ax_tilt.set_ylabel("Tilt [deg]", color="#A0A5B5", fontsize=8)
        self.ax_tilt.tick_params(colors="#A0A5B5", labelsize=7)
        self.ax_tilt.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7, ncol=3)

        # 2. Platform Origin Translation [mm]
        self.ax_trans = self.fig_plat.add_subplot(3, 1, 2)
        self.ax_trans.set_facecolor("#1A1B24")
        self.ax_trans.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_tx, = self.ax_trans.plot([], [], color="#4CC9F0", linewidth=1.4, label="dx (X_p)")
        self.line_ty, = self.ax_trans.plot([], [], color="#06D6A0", linewidth=1.4, label="dy (Y_p)")
        self.line_tz, = self.ax_trans.plot([], [], color="#FFAA00", linewidth=1.4, label="dz (Z_p)")
        self.ax_trans.set_ylabel("Trans [mm]", color="#A0A5B5", fontsize=8)
        self.ax_trans.tick_params(colors="#A0A5B5", labelsize=7)
        self.ax_trans.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7, ncol=3)

        # 3. Platform Linear Velocity [mm/s]
        self.ax_tvel = self.fig_plat.add_subplot(3, 1, 3)
        self.ax_tvel.set_facecolor("#1A1B24")
        self.ax_tvel.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_tvx, = self.ax_tvel.plot([], [], color="#4CC9F0", linewidth=1.2, label="vx_p")
        self.line_tvy, = self.ax_tvel.plot([], [], color="#06D6A0", linewidth=1.2, label="vy_p")
        self.line_tvz, = self.ax_tvel.plot([], [], color="#FFAA00", linewidth=1.2, label="vz_p")
        self.ax_tvel.set_ylabel("Vel [mm/s]", color="#A0A5B5", fontsize=8)
        self.ax_tvel.set_xlabel("Time [s]", color="#A0A5B5", fontsize=8)
        self.ax_tvel.tick_params(colors="#A0A5B5", labelsize=7)
        self.ax_tvel.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7, ncol=3)

        self.canvas_plat = FigureCanvasTkAgg(self.fig_plat, master=self.tab_platform)
        self.canvas_plat.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _poll_telemetry(self) -> None:
        new_packet = None
        try:
            while not self.queue.empty():
                new_packet = self.queue.get_nowait()
        except queue.Empty:
            pass

        if new_packet is not None:
            self._update_views(new_packet)

        self.after(33, self._poll_telemetry)

    def _update_views(self, data: dict) -> None:
        t = data["t"]
        bp = data["ball_pos"]
        rp = data["ref_pos"]
        bp_raw = data.get("ball_pos_raw", bp)
        bv = data.get("ball_vel", np.zeros(2))
        tilt = data["tilt"]
        n_force = data["normal_force"]
        is_safe = data["safety_ok"]
        
        # Extract platform origin translation and velocity
        p_trans = data.get("platform_trans", [0.0, 0.0, 0.0])
        p_vel = data.get("platform_vel", [0.0, 0.0, 0.0])

        err_mm = float(np.linalg.norm(bp - rp) * 1000.0)

        # Update labels
        self.lbl_ball.config(text=f"Ball (x, y): [{bp[0]:+6.3f}, {bp[1]:+6.3f}] m")
        self.lbl_ref.config(text=f"Ref  (x, y): [{rp[0]:+6.3f}, {rp[1]:+6.3f}] m")
        
        err_color = self.accent_green if err_mm < 15.0 else (self.accent_yellow if err_mm < 35.0 else self.accent_red)
        self.lbl_err.config(text=f"Tracking Err: {err_mm:6.2f} mm", foreground=err_color)

        yaw_deg = tilt[2] if len(tilt) > 2 else 0.0
        self.lbl_tilt.config(text=f"Tilt (R/P/Y): [{tilt[0]:+5.1f}°, {tilt[1]:+5.1f}°, {yaw_deg:+5.1f}°]")

        self.lbl_plat_pos.config(text=f"Plat Pos:   [{p_trans[0]:+5.1f}, {p_trans[1]:+5.1f}, {p_trans[2]:+5.1f}] mm")
        self.lbl_plat_vel.config(text=f"Plat Vel:   [{p_vel[0]:+5.1f}, {p_vel[1]:+5.1f}, {p_vel[2]:+5.1f}] mm/s")

        if n_force > 0.10:
            self.lbl_contact.config(text=f"Normal Force: {n_force:.3f} N [TOUCH]", foreground=self.text_color)
        else:
            self.lbl_contact.config(text=f"Normal Force: {n_force:.3f} N [DETACHED]", foreground=self.accent_red)

        if is_safe:
            self.lbl_safety.config(text="Safety Filter: SOCP [OK]", foreground=self.accent_green)
        else:
            self.lbl_safety.config(text="Safety Filter: [FALLBACK]", foreground=self.accent_red)

        # Buffers
        self.t_hist.append(t)
        self.bx_hist.append(bp[0])
        self.by_hist.append(bp[1])
        self.rx_hist.append(rp[0])
        self.ry_hist.append(rp[1])
        self.bx_raw_hist.append(bp_raw[0])
        self.by_raw_hist.append(bp_raw[1])
        self.vx_hist.append(bv[0])
        self.vy_hist.append(bv[1])
        self.err_hist.append(err_mm)
        self.roll_hist.append(tilt[0])
        self.pitch_hist.append(tilt[1])
        self.yaw_hist.append(yaw_deg)
        self.force_hist.append(n_force)
        
        self.tx_hist.append(p_trans[0])
        self.ty_hist.append(p_trans[1])
        self.tz_hist.append(p_trans[2])
        self.tvx_hist.append(p_vel[0])
        self.tvy_hist.append(p_vel[1])
        self.tvz_hist.append(p_vel[2])

        if len(self.t_hist) > self.max_len:
            self.t_hist.pop(0)
            self.bx_hist.pop(0)
            self.by_hist.pop(0)
            self.rx_hist.pop(0)
            self.ry_hist.pop(0)
            self.bx_raw_hist.pop(0)
            self.by_raw_hist.pop(0)
            self.vx_hist.pop(0)
            self.vy_hist.pop(0)
            self.err_hist.pop(0)
            self.roll_hist.pop(0)
            self.pitch_hist.pop(0)
            self.yaw_hist.pop(0)
            self.force_hist.pop(0)
            self.tx_hist.pop(0)
            self.ty_hist.pop(0)
            self.tz_hist.pop(0)
            self.tvx_hist.pop(0)
            self.tvy_hist.pop(0)
            self.tvz_hist.pop(0)

        # Render Active Tab Only
        active_tab = self.notebook.index(self.notebook.select())

        if active_tab == 0:
            self.marker_ball.set_data([bp[0]], [bp[1]])
            self.marker_ref.set_data([rp[0]], [rp[1]])
            self.marker_raw_cop.set_data([bp_raw[0]], [bp_raw[1]])
            self.line_trail.set_data(self.bx_hist[-75:], self.by_hist[-75:])
            self.canvas_map.draw_idle()

        elif active_tab == 1:
            t_arr = np.array(self.t_hist)
            self.line_err.set_data(t_arr, self.err_hist)
            self.ax_err.relim()
            self.ax_err.autoscale_view()

            self.line_bx.set_data(t_arr, self.bx_hist)
            self.line_rx.set_data(t_arr, self.rx_hist)
            self.line_by.set_data(t_arr, self.by_hist)
            self.line_ry.set_data(t_arr, self.ry_hist)
            self.ax_pos.relim()
            self.ax_pos.autoscale_view()
            self.canvas_track.draw_idle()

        elif active_tab == 2:
            t_arr = np.array(self.t_hist)
            self.line_raw_x.set_data(t_arr, self.bx_raw_hist)
            self.line_clean_x.set_data(t_arr, self.bx_hist)
            self.line_raw_y.set_data(t_arr, self.by_raw_hist)
            self.line_clean_y.set_data(t_arr, self.by_hist)
            self.ax_cop.relim()
            self.ax_cop.autoscale_view()

            self.line_vx.set_data(t_arr, self.vx_hist)
            self.line_vy.set_data(t_arr, self.vy_hist)
            self.ax_vel.relim()
            self.ax_vel.autoscale_view()
            self.canvas_perc.draw_idle()

        elif active_tab == 3:
            t_arr = np.array(self.t_hist)
            self.line_roll.set_data(t_arr, self.roll_hist)
            self.line_pitch.set_data(t_arr, self.pitch_hist)
            self.line_yaw.set_data(t_arr, self.yaw_hist)
            self.ax_tilt.relim()
            self.ax_tilt.autoscale_view()

            self.line_tx.set_data(t_arr, self.tx_hist)
            self.line_ty.set_data(t_arr, self.ty_hist)
            self.line_tz.set_data(t_arr, self.tz_hist)
            self.ax_trans.relim()
            self.ax_trans.autoscale_view()

            self.line_tvx.set_data(t_arr, self.tvx_hist)
            self.line_tvy.set_data(t_arr, self.tvy_hist)
            self.line_tvz.set_data(t_arr, self.tvz_hist)
            self.ax_tvel.relim()
            self.ax_tvel.autoscale_view()

            self.canvas_plat.draw_idle()

    def _on_mode_switched(self, event=None) -> None:
        self._render_trajectory_controls()
        self._on_param_changed()

    def _on_param_changed(self, event=None) -> None:
        if not getattr(self, "_ui_initialized", False):
            return

        mode = self.cmb_mode.get()
        p1 = float(self.scale_p1.get())
        p2 = float(self.scale_p2.get())

        if mode == "Static Point":
            self.lbl_val_p1.config(text=f"X = {p1:+.3f} m")
            self.lbl_val_p2.config(text=f"Y = {p2:+.3f} m")
            self.line_ref_orbit.set_data([p1], [p2])
        elif "Lissajous" in mode:
            self.lbl_val_p1.config(text=f"Amp = {p1:.2f} m ({p1*100:.1f} cm)")
            self.lbl_val_p2.config(text=f"w = {p2:.2f} rad/s")
            theta = np.linspace(0, 2 * np.pi, 140)
            if "1:2" in mode:
                self.line_ref_orbit.set_data(p1 * np.sin(theta), p1 * np.sin(2.0 * theta))
            else:
                self.line_ref_orbit.set_data(p1 * np.sin(2.0 * theta), p1 * np.cos(3.0 * theta))
        else:
            self.lbl_val_p1.config(text=f"R = {p1:.2f} m ({p1*100:.1f} cm)")
            self.lbl_val_p2.config(text=f"w = {p2:.2f} rad/s ({np.degrees(p2):.1f} °/s)")
            theta = np.linspace(0, 2 * np.pi, 80)
            self.line_ref_orbit.set_data(p1 * np.cos(theta), p1 * np.sin(theta))

        self.canvas_map.draw_idle()

        self.cmd_callback({
            "action": "update_params",
            "mode": mode,
            "p1": p1,
            "p2": p2
        })

    def _on_cmd_center(self) -> None:
        self.cmd_callback({"action": "center"})

    def _on_cmd_orbit(self) -> None:
        self.cmd_callback({"action": "orbit"})