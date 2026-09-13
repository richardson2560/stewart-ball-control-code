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
    Professional real-time telemetry dashboard for the Stewart-Ball Control (SBC v30) system.
    Decoupled 30 Hz Tkinter/Matplotlib interface consuming thread-safe telemetry queues.
    """

    def __init__(
        self,
        telemetry_queue: queue.Queue,
        command_callback: Callable[[dict], None],
        plate_radius: float = 0.45,
        max_history: int = 300
    ) -> None:
        super().__init__()
        self.queue = telemetry_queue
        self.cmd_callback = command_callback
        self.R_plate = plate_radius
        self.max_len = max_history

        self.title("SBC v30 - High-Speed Stewart Platform Telemetry Cockpit")
        self.geometry("1400x880")
        self.minsize(1100, 700)

        # Style configuration
        self._setup_styles()

        # Rolling history buffers
        self.t_hist = []
        self.bx_hist, self.by_hist = [], []
        self.rx_hist, self.ry_hist = [], []
        self.err_hist = []
        self.roll_hist, self.pitch_hist, self.yaw_hist = [], [], []
        self.force_hist = []

        # Build UI layout
        self._build_sidebar()
        self._build_central_notebook()

        # Start periodic non-blocking consumer
        self._poll_telemetry()

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        
        # Color palette
        self.bg_color = "#1E1E24"
        self.card_bg = "#2B2D42"
        self.text_color = "#EDF2F4"
        self.accent_color = "#EF233C"
        self.accent_blue = "#4895EF"
        self.accent_green = "#4BB543"

        self.configure(bg=self.bg_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="flat")
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground=self.accent_blue)
        style.configure("Metric.TLabel", font=("Consolas", 11, "bold"), background=self.card_bg)
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

    def _build_sidebar(self) -> None:
        sidebar = ttk.Frame(self, width=340, padding=12)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="STEWART-BALL CONTROL", font=("Segoe UI", 14, "bold"), foreground=self.text_color).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(sidebar, text="Model-Based Robust Tracking & SOCP Safety", font=("Segoe UI", 8), foreground="#8D99AE").pack(anchor=tk.W, pady=(0, 10))

        # --- Card 1: Live Status & Safety Indicators ---
        card_status = ttk.Frame(sidebar, style="Card.TFrame", padding=10)
        card_status.pack(fill=tk.X, pady=6)
        ttk.Label(card_status, text="SYSTEM STATUS & SAFETY", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 6))

        self.lbl_status = ttk.Label(card_status, text="State: INITIALIZING", style="Card.TLabel")
        self.lbl_status.pack(anchor=tk.W, pady=2)
        
        self.lbl_safety = ttk.Label(card_status, text="Safety Filter: SOCP [OK]", font=("Segoe UI", 10, "bold"), foreground=self.accent_green, background=self.card_bg)
        self.lbl_safety.pack(anchor=tk.W, pady=2)

        self.lbl_contact = ttk.Label(card_status, text="Normal Force: 0.642 N [TOUCH]", style="Card.TLabel")
        self.lbl_contact.pack(anchor=tk.W, pady=2)

        # --- Card 2: Numeric Readouts ---
        card_metrics = ttk.Frame(sidebar, style="Card.TFrame", padding=10)
        card_metrics.pack(fill=tk.X, pady=6)
        ttk.Label(card_metrics, text="LIVE KINEMATICS", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 6))

        self.lbl_ball = ttk.Label(card_metrics, text="Ball (x, y): [+0.000, +0.000] m", style="Metric.TLabel")
        self.lbl_ball.pack(anchor=tk.W, pady=2)

        self.lbl_ref = ttk.Label(card_metrics, text="Ref  (x, y): [+0.000, +0.000] m", style="Metric.TLabel")
        self.lbl_ref.pack(anchor=tk.W, pady=2)

        self.lbl_err = ttk.Label(card_metrics, text="Tracking Err: 0.00 mm", font=("Consolas", 12, "bold"), foreground=self.accent_green, background=self.card_bg)
        self.lbl_err.pack(anchor=tk.W, pady=3)

        self.lbl_tilt = ttk.Label(card_metrics, text="Tilt (R/P):  [ +0.0°,  +0.0°]", style="Metric.TLabel")
        self.lbl_tilt.pack(anchor=tk.W, pady=2)

        # --- Card 3: Interactive Trajectory Generator ---
        card_traj = ttk.Frame(sidebar, style="Card.TFrame", padding=10)
        card_traj.pack(fill=tk.X, pady=6)
        ttk.Label(card_traj, text="TRAJECTORY GENERATOR", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 6))

        ttk.Label(card_traj, text="Mode:", style="Card.TLabel").pack(anchor=tk.W)
        self.cmb_mode = ttk.Combobox(card_traj, values=["Circle Orbit", "Static Point", "Lissajous 8"], state="readonly")
        self.cmb_mode.set("Circle Orbit")
        self.cmb_mode.pack(fill=tk.X, pady=3)
        self.cmb_mode.bind("<<ComboboxSelected>>", self._on_param_changed)

        ttk.Label(card_traj, text="Orbit Radius R [m]:", style="Card.TLabel").pack(anchor=tk.W, pady=(4, 0))
        self.scale_r = ttk.Scale(card_traj, from_=0.02, to=0.25, orient=tk.HORIZONTAL, command=self._on_param_changed)
        self.scale_r.set(0.08)
        self.scale_r.pack(fill=tk.X)
        self.lbl_scale_r = ttk.Label(card_traj, text="R = 0.08 m (8.0 cm)", style="Card.TLabel")
        self.lbl_scale_r.pack(anchor=tk.E)

        ttk.Label(card_traj, text="Speed w [rad/s]:", style="Card.TLabel").pack(anchor=tk.W, pady=(4, 0))
        self.scale_w = ttk.Scale(card_traj, from_=0.5, to=3.0, orient=tk.HORIZONTAL, command=self._on_param_changed)
        self.scale_w.set(1.2)
        self.scale_w.pack(fill=tk.X)
        self.lbl_scale_w = ttk.Label(card_traj, text="w = 1.20 rad/s (68.8 °/s)", style="Card.TLabel")
        self.lbl_scale_w.pack(anchor=tk.E)

        # --- Buttons ---
        btn_box = ttk.Frame(sidebar)
        btn_box.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        ttk.Button(btn_box, text="Center Ball [0, 0]", command=self._on_cmd_center).pack(fill=tk.X, pady=3)
        ttk.Button(btn_box, text="Start / Resume Orbit", command=self._on_cmd_orbit).pack(fill=tk.X, pady=3)

    def _build_central_notebook(self) -> None:
        container = ttk.Frame(self, padding=8)
        container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: 2D Spatial Plate Map
        self.tab_map = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_map, text="  Plate 2D Overhead Map  ")

        # Tab 2: Time Series Error & Dynamics
        self.tab_series = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_series, text="  Tracking Dynamics & Error  ")

        self._setup_map_plot()
        self._setup_series_plot()

    def _setup_map_plot(self) -> None:
        self.fig_map = plt.Figure(figsize=(6, 6), facecolor="#242633", dpi=100)
        self.ax_map = self.fig_map.add_subplot(1, 1, 1)
        self.ax_map.set_facecolor("#1A1B24")
        self.ax_map.set_aspect("equal")

        # Draw physical boundary of Stewart plate
        lim = self.R_plate + 0.05
        self.ax_map.set_xlim(-lim, lim)
        self.ax_map.set_ylim(-lim, lim)
        self.ax_map.grid(True, color="#383A4D", linestyle="--", alpha=0.6)
        self.ax_map.set_xlabel("Platform X [m]", color="#A0A5B5", fontsize=9)
        self.ax_map.set_ylabel("Platform Y [m]", color="#A0A5B5", fontsize=9)
        self.ax_map.tick_params(colors="#A0A5B5", labelsize=8)

        # Plate boundary circle
        plate_circle = Circle((0, 0), self.R_plate, color="#EF233C", fill=False, linewidth=2.0, linestyle="-", label=f"Boundary (R={self.R_plate}m)")
        self.ax_map.add_patch(plate_circle)

        # Elements: Trail, Ball, Reference
        self.line_trail, = self.ax_map.plot([], [], color="#4895EF", linewidth=1.2, alpha=0.7, label="Ball Trail")
        self.line_ref_orbit, = self.ax_map.plot([], [], color="#06D6A0", linestyle="--", linewidth=1.5, label="Ref Orbit")
        self.marker_ball, = self.ax_map.plot([], [], "o", color="#E63946", markersize=14, markeredgecolor="#FFFFFF", markeredgewidth=1.5, label="Ball (Contact)")
        self.marker_ref, = self.ax_map.plot([], [], "X", color="#06D6A0", markersize=12, markeredgecolor="#FFFFFF", label="Ref Target")

        self.ax_map.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=8)
        self.fig_map.tight_layout()

        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.tab_map)
        self.canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_series_plot(self) -> None:
        self.fig_series = plt.Figure(figsize=(7, 6), facecolor="#242633", dpi=100)
        self.fig_series.subplots_adjust(hspace=0.35, left=0.10, right=0.95, top=0.95, bottom=0.08)

        # Subplot 1: Tracking Error [mm]
        self.ax_err = self.fig_series.add_subplot(3, 1, 1)
        self.ax_err.set_facecolor("#1A1B24")
        self.ax_err.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_err, = self.ax_err.plot([], [], color="#EF233C", linewidth=1.8, label="Tracking Error ||e||")
        self.ax_err.set_ylabel("Error [mm]", color="#A0A5B5", fontsize=9)
        self.ax_err.tick_params(colors="#A0A5B5", labelsize=8)
        self.ax_err.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7)

        # Subplot 2: Tilt Angles Roll/Pitch [deg]
        self.ax_tilt = self.fig_series.add_subplot(3, 1, 2)
        self.ax_tilt.set_facecolor("#1A1B24")
        self.ax_tilt.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_roll, = self.ax_tilt.plot([], [], color="#4895EF", linewidth=1.5, label="Roll (phi)")
        self.line_pitch, = self.ax_tilt.plot([], [], color="#F72585", linewidth=1.5, label="Pitch (theta)")
        self.ax_tilt.set_ylabel("Tilt [deg]", color="#A0A5B5", fontsize=9)
        self.ax_tilt.tick_params(colors="#A0A5B5", labelsize=8)
        self.ax_tilt.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7)

        # Subplot 3: Normal Load N [N]
        self.ax_norm = self.fig_series.add_subplot(3, 1, 3)
        self.ax_norm.set_facecolor("#1A1B24")
        self.ax_norm.grid(True, color="#383A4D", linestyle="--", alpha=0.5)
        self.line_norm, = self.ax_norm.plot([], [], color="#06D6A0", linewidth=1.5, label="Normal Force N")
        self.ax_norm.axhline(0.10, color="#E63946", linestyle=":", linewidth=1.2, label="N_min floor")
        self.ax_norm.set_ylabel("Force [N]", color="#A0A5B5", fontsize=9)
        self.ax_norm.set_xlabel("Time [s]", color="#A0A5B5", fontsize=9)
        self.ax_norm.tick_params(colors="#A0A5B5", labelsize=8)
        self.ax_norm.legend(loc="upper right", facecolor="#242633", edgecolor="#383A4D", labelcolor="#EDF2F4", fontsize=7)

        self.canvas_series = FigureCanvasTkAgg(self.fig_series, master=self.tab_series)
        self.canvas_series.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _poll_telemetry(self) -> None:
        """Polls queue non-blockingly and drains to newest state."""
        new_packet = None
        try:
            while not self.queue.empty():
                new_packet = self.queue.get_nowait()
        except queue.Empty:
            pass

        if new_packet is not None:
            self._update_views(new_packet)

        # Schedule next tick at ~30 FPS (33 ms)
        self.after(33, self._poll_telemetry)

    def _update_views(self, data: dict) -> None:
        t = data["t"]
        bp = data["ball_pos"]
        rp = data["ref_pos"]
        tilt = data["tilt"]
        n_force = data["normal_force"]
        is_safe = data["safety_ok"]

        err_mm = float(np.linalg.norm(bp - rp) * 1000.0)

        # Update text metrics
        self.lbl_ball.config(text=f"Ball (x, y): [{bp[0]:+6.3f}, {bp[1]:+6.3f}] m")
        self.lbl_ref.config(text=f"Ref  (x, y): [{rp[0]:+6.3f}, {rp[1]:+6.3f}] m")
        
        err_color = self.accent_green if err_mm < 15.0 else ("#FFD166" if err_mm < 35.0 else self.accent_color)
        self.lbl_err.config(text=f"Tracking Err: {err_mm:6.2f} mm", foreground=err_color)

        self.lbl_tilt.config(text=f"Tilt (R/P):  [{tilt[0]:+5.1f}°, {tilt[1]:+5.1f}°]")
        
        if n_force > 0.10:
            self.lbl_contact.config(text=f"Normal Force: {n_force:.3f} N [TOUCH]", foreground=self.text_color)
        else:
            self.lbl_contact.config(text=f"Normal Force: {n_force:.3f} N [DETACHED]", foreground=self.accent_color)

        if is_safe:
            self.lbl_safety.config(text="Safety Filter: SOCP [OK]", foreground=self.accent_green)
        else:
            self.lbl_safety.config(text="Safety Filter: [FALLBACK]", foreground=self.accent_color)

        # Append rolling history
        self.t_hist.append(t)
        self.bx_hist.append(bp[0])
        self.by_hist.append(bp[1])
        self.rx_hist.append(rp[0])
        self.ry_hist.append(rp[1])
        self.err_hist.append(err_mm)
        self.roll_hist.append(tilt[0])
        self.pitch_hist.append(tilt[1])
        self.force_hist.append(n_force)

        if len(self.t_hist) > self.max_len:
            self.t_hist.pop(0)
            self.bx_hist.pop(0)
            self.by_hist.pop(0)
            self.rx_hist.pop(0)
            self.ry_hist.pop(0)
            self.err_hist.pop(0)
            self.roll_hist.pop(0)
            self.pitch_hist.pop(0)
            self.force_hist.pop(0)

        # Update 2D Map Tab
        self.marker_ball.set_data([bp[0]], [bp[1]])
        self.marker_ref.set_data([rp[0]], [rp[1]])
        self.line_trail.set_data(self.bx_hist[-80:], self.by_hist[-80:])

        # Render preview of nominal orbit
        r_val = self.scale_r.get()
        theta_cir = np.linspace(0, 2 * np.pi, 60)
        self.line_ref_orbit.set_data(r_val * np.cos(theta_cir), r_val * np.sin(theta_cir))

        active_tab = self.notebook.index(self.notebook.select())
        if active_tab == 0:
            self.canvas_map.draw_idle()
        elif active_tab == 1:
            # Update Series Tab
            t_arr = np.array(self.t_hist)
            self.line_err.set_data(t_arr, self.err_hist)
            self.ax_err.relim()
            self.ax_err.autoscale_view()

            self.line_roll.set_data(t_arr, self.roll_hist)
            self.line_pitch.set_data(t_arr, self.pitch_hist)
            self.ax_tilt.relim()
            self.ax_tilt.autoscale_view()

            self.line_norm.set_data(t_arr, self.force_hist)
            self.ax_norm.relim()
            self.ax_norm.autoscale_view()

            self.canvas_series.draw_idle()

    def _on_param_changed(self, event=None) -> None:
        r = float(self.scale_r.get())
        w = float(self.scale_w.get())
        self.lbl_scale_r.config(text=f"R = {r:.2f} m ({r*100:.1f} cm)")
        self.lbl_scale_w.config(text=f"w = {w:.2f} rad/s ({np.degrees(w):.1f} °/s)")

        self.cmd_callback({
            "action": "update_params",
            "mode": self.cmb_mode.get(),
            "radius": r,
            "speed": w
        })

    def _on_cmd_center(self) -> None:
        self.cmd_callback({"action": "center"})

    def _on_cmd_orbit(self) -> None:
        self.cmd_callback({"action": "orbit"})