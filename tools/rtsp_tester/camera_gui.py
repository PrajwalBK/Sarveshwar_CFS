"""
Low-Latency CP Plus PoE IP Camera RTSP Tester & Recorder GUI
100% Offline | Zero Cloud Dependency | Windows Laptop & NVIDIA Jetson Ready
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# Import decoupled streaming engine & auto-scanner
from rtsp_streamer import RTSPStreamer, StreamStatus
from scanner import scan_network_for_cameras


class CameraTesterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CP Plus PoE IP Camera - Low-Latency RTSP Tester")
        self.root.geometry("1240x820")
        self.root.minsize(980, 680)

        # Apply Modern Dark Palette
        self.bg_color = "#0f172a"        # Dark slate background
        self.panel_bg = "#1e293b"        # Card background
        self.panel_border = "#334155"    # Card borders
        self.text_primary = "#f8fafc"    # Bright text
        self.text_secondary = "#94a3b8"  # Muted text
        self.accent_blue = "#38bdf8"     # Sky blue
        self.accent_green = "#22c55e"    # Green
        self.accent_red = "#ef4444"      # Red
        self.accent_yellow = "#eab308"   # Amber

        self.root.configure(bg=self.bg_color)

        # State Variables
        self.streamer: Optional[RTSPStreamer] = None
        self.is_connected = False
        self.current_img_tk = None
        self.output_dir = Path(__file__).parent / "output"

        self._setup_styles()
        self._build_ui()
        self._start_gui_update_loop()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.bg_color)
        style.configure("Panel.TFrame", background=self.panel_bg)

        style.configure("TLabel", background=self.panel_bg, foreground=self.text_primary, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.bg_color, foreground=self.text_primary, font=("Segoe UI", 15, "bold"))
        style.configure("SubHeader.TLabel", background=self.panel_bg, foreground=self.accent_blue, font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", background=self.panel_bg, foreground=self.text_secondary, font=("Segoe UI", 9))

        style.configure("TEntry", fieldbackground="#0f172a", foreground="#ffffff", insertcolor="#ffffff")
        style.configure("TCombobox", fieldbackground="#0f172a", foreground="#ffffff")

    def _build_ui(self):
        # 1. Top Title Header
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=10, padx=16)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="🎥 CP Plus PoE IP Camera - Low-Latency RTSP Stream Tester",
            font=("Segoe UI", 15, "bold"),
            bg=self.bg_color,
            fg=self.text_primary,
        )
        title_lbl.pack(side=tk.LEFT)

        self.status_badge = tk.Label(
            header_frame,
            text="● DISCONNECTED",
            font=("Segoe UI", 10, "bold"),
            bg="#334155",
            fg=self.text_secondary,
            padx=12,
            pady=4,
        )
        self.status_badge.pack(side=tk.RIGHT)

        # 2. Main Content Split (Left: Video Canvas, Right: Controls & Metrics)
        main_content = tk.Frame(self.root, bg=self.bg_color, padx=14, pady=6)
        main_content.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Viewport
        video_panel = tk.Frame(main_content, bg=self.panel_bg, highlightbackground=self.panel_border, highlightthickness=1)
        video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(video_panel, bg="#050811", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_text = self.canvas.create_text(
            400, 300,
            text="No Video Feed\nConfigure RTSP settings on the right and click [Connect]",
            fill=self.text_secondary,
            font=("Segoe UI", 12),
            justify=tk.CENTER,
        )

        # Right Column: Configuration & Live Stats
        right_panel = tk.Frame(main_content, bg=self.panel_bg, width=380, highlightbackground=self.panel_border, highlightthickness=1, padx=14, pady=14)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # --- Section A: Connection Parameters ---
        header_row = tk.Frame(right_panel, bg=self.panel_bg)
        header_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(header_row, text="Connection Setup", font=("Segoe UI", 11, "bold"), bg=self.panel_bg, fg=self.accent_blue).pack(side=tk.LEFT)

        self.scan_btn = tk.Button(
            header_row,
            text="🔍 Auto-Scan IP",
            bg="#334155",
            fg=self.accent_blue,
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=2,
            command=self.auto_scan_camera,
            cursor="hand2",
        )
        self.scan_btn.pack(side=tk.RIGHT)

        # Camera IP
        ip_row = tk.Frame(right_panel, bg=self.panel_bg)
        ip_row.pack(fill=tk.X, pady=3)
        tk.Label(ip_row, text="Camera IP:", width=12, anchor=tk.W, bg=self.panel_bg, fg=self.text_primary).pack(side=tk.LEFT)
        self.ip_entry = tk.Entry(ip_row, bg="#0f172a", fg="#ffffff", insertbackground="#ffffff", relief=tk.FLAT)
        self.ip_entry.insert(0, "192.168.1.250")
        self.ip_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.ip_entry.bind("<KeyRelease>", self._on_param_change)

        # RTSP Port
        port_row = tk.Frame(right_panel, bg=self.panel_bg)
        port_row.pack(fill=tk.X, pady=3)
        tk.Label(port_row, text="RTSP Port:", width=12, anchor=tk.W, bg=self.panel_bg, fg=self.text_primary).pack(side=tk.LEFT)
        self.port_entry = tk.Entry(port_row, bg="#0f172a", fg="#ffffff", insertbackground="#ffffff", relief=tk.FLAT)
        self.port_entry.insert(0, "554")
        self.port_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.port_entry.bind("<KeyRelease>", self._on_param_change)

        # Username
        user_row = tk.Frame(right_panel, bg=self.panel_bg)
        user_row.pack(fill=tk.X, pady=3)
        tk.Label(user_row, text="Username:", width=12, anchor=tk.W, bg=self.panel_bg, fg=self.text_primary).pack(side=tk.LEFT)
        self.user_entry = tk.Entry(user_row, bg="#0f172a", fg="#ffffff", insertbackground="#ffffff", relief=tk.FLAT)
        self.user_entry.insert(0, "admin")
        self.user_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.user_entry.bind("<KeyRelease>", self._on_param_change)

        # Password
        pass_row = tk.Frame(right_panel, bg=self.panel_bg)
        pass_row.pack(fill=tk.X, pady=3)
        tk.Label(pass_row, text="Password:", width=12, anchor=tk.W, bg=self.panel_bg, fg=self.text_primary).pack(side=tk.LEFT)
        self.pass_entry = tk.Entry(pass_row, bg="#0f172a", fg="#ffffff", insertbackground="#ffffff", relief=tk.FLAT, show="*")
        self.pass_entry.insert(0, "admin")
        self.pass_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.pass_entry.bind("<KeyRelease>", self._on_param_change)

        # Stream Channel / Subtype
        stream_row = tk.Frame(right_panel, bg=self.panel_bg)
        stream_row.pack(fill=tk.X, pady=3)
        tk.Label(stream_row, text="Stream Type:", width=12, anchor=tk.W, bg=self.panel_bg, fg=self.text_primary).pack(side=tk.LEFT)
        self.stream_type_var = tk.StringVar(value="Main Stream (subtype=0)")
        self.stream_type_combo = ttk.Combobox(
            stream_row,
            textvariable=self.stream_type_var,
            values=["Main Stream (subtype=0)", "Sub Stream (subtype=1)"],
            state="readonly"
        )
        self.stream_type_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.stream_type_combo.bind("<<ComboboxSelected>>", self._on_param_change)

        # Full RTSP URL Field
        url_lbl = tk.Label(right_panel, text="Resolved RTSP URL:", anchor=tk.W, bg=self.panel_bg, fg=self.text_secondary, font=("Segoe UI", 9))
        url_lbl.pack(fill=tk.X, pady=(6, 2))
        self.url_entry = tk.Entry(right_panel, bg="#0f172a", fg="#38bdf8", insertbackground="#ffffff", relief=tk.FLAT)
        self.url_entry.pack(fill=tk.X, pady=(0, 6))

        # Backend Engine Selection
        backend_row = tk.Frame(right_panel, bg=self.panel_bg)
        backend_row.pack(fill=tk.X, pady=3)
        tk.Label(backend_row, text="Backend Engine:", width=12, anchor=tk.W, bg=self.panel_bg, fg=self.text_primary).pack(side=tk.LEFT)
        self.backend_var = tk.StringVar(value="OpenCV TCP (Low-Latency)")
        self.backend_combo = ttk.Combobox(
            backend_row,
            textvariable=self.backend_var,
            values=[
                "OpenCV TCP (Low-Latency)",
                "GStreamer H.264 (Pipeline)",
                "GStreamer H.265 / HEVC",
                "NVIDIA Jetson (HW-Dec)",
            ],
            state="readonly"
        )
        self.backend_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self._update_rtsp_url_field()

        # --- Section B: Action Buttons ---
        btn_frame = tk.Frame(right_panel, bg=self.panel_bg, pady=8)
        btn_frame.pack(fill=tk.X)

        self.connect_btn = tk.Button(
            btn_frame,
            text="▶ Connect",
            bg=self.accent_green,
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self.connect_camera,
            cursor="hand2",
        )
        self.connect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.disconnect_btn = tk.Button(
            btn_frame,
            text="⏹ Disconnect",
            bg="#475569",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self.disconnect_camera,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.disconnect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Action Buttons Row 2
        act_frame = tk.Frame(right_panel, bg=self.panel_bg, pady=4)
        act_frame.pack(fill=tk.X)

        self.snap_btn = tk.Button(
            act_frame,
            text="📸 Snapshot",
            bg="#0284c7",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=4,
            command=self.take_snapshot,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.snap_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.rec_btn = tk.Button(
            act_frame,
            text="⏺ Start Recording",
            bg="#b91c1c",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=4,
            command=self.toggle_recording,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.rec_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Open Output Folder Button
        open_folder_btn = tk.Button(
            right_panel,
            text="📁 Open Captures / Recordings Folder",
            bg="#1e293b",
            fg=self.text_secondary,
            font=("Segoe UI", 9),
            relief=tk.GROOVE,
            command=self.open_output_folder,
            cursor="hand2",
        )
        open_folder_btn.pack(fill=tk.X, pady=(6, 12))

        # --- Section C: Live Statistics HUD ---
        tk.Label(right_panel, text="Live Telemetry & Metrics", font=("Segoe UI", 11, "bold"), bg=self.panel_bg, fg=self.accent_blue).pack(anchor=tk.W, pady=(4, 6))

        stats_frame = tk.Frame(right_panel, bg="#0f172a", padx=10, pady=10, highlightbackground=self.panel_border, highlightthickness=1)
        stats_frame.pack(fill=tk.X)

        self.stat_labels = {}
        metric_keys = [
            ("Status", "status", "Disconnected"),
            ("Resolution", "resolution", "0x0"),
            ("Camera Nominal FPS", "camera_fps", "0.0 FPS"),
            ("Actual Received FPS", "rx_fps", "0.0 FPS"),
            ("Display Render FPS", "render_fps", "0.0 FPS"),
            ("Ingestion Latency", "latency_ms", "0.0 ms"),
            ("Frames Received", "frames_received", "0"),
            ("Dropped Frames", "frames_dropped", "0"),
            ("Reconnection Count", "reconnect_count", "0"),
            ("Recording Status", "recording_status", "Idle"),
        ]

        for i, (label_text, key, initial_val) in enumerate(metric_keys):
            row = tk.Frame(stats_frame, bg="#0f172a")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label_text + ":", bg="#0f172a", fg=self.text_secondary, font=("Segoe UI", 9)).pack(side=tk.LEFT)
            val_lbl = tk.Label(row, text=initial_val, bg="#0f172a", fg=self.text_primary, font=("Segoe UI", 9, "bold"))
            val_lbl.pack(side=tk.RIGHT)
            self.stat_labels[key] = val_lbl

        # Log Messages area
        self.log_lbl = tk.Label(right_panel, text="", bg=self.panel_bg, fg=self.accent_yellow, font=("Segoe UI", 8), wraplength=350, justify=tk.LEFT)
        self.log_lbl.pack(fill=tk.X, pady=(8, 0))

    def auto_scan_camera(self):
        """Run non-blocking discovery scan across Ethernet adapters to find camera IP."""
        self.scan_btn.config(state=tk.DISABLED, text="⏳ Scanning...")
        self.log_lbl.config(text="Scanning Ethernet network for CP Plus / ONVIF IP camera...", fg=self.accent_blue)

        def _scan_worker():
            try:
                cams = scan_network_for_cameras()
                
                def _update_ui():
                    self.scan_btn.config(state=tk.NORMAL, text="🔍 Auto-Scan IP")
                    if cams:
                        best_cam = cams[0]
                        cam_ip = best_cam["ip"]
                        proto = best_cam.get("protocol", "IP Camera")
                        self.ip_entry.delete(0, tk.END)
                        self.ip_entry.insert(0, cam_ip)
                        self._update_rtsp_url_field()
                        self.log_lbl.config(
                            text=f"✓ Found camera at {cam_ip} via {proto}!",
                            fg=self.accent_green
                        )
                        messagebox.showinfo(
                            "Camera Detected",
                            f"Successfully detected IP Camera!\n\nIP Address: {cam_ip}\nProtocol: {proto}\n\nClick [Connect] to begin streaming."
                        )
                    else:
                        self.log_lbl.config(
                            text="No camera found automatically. Default 192.168.1.250 kept.",
                            fg=self.accent_yellow
                        )
                        messagebox.showwarning(
                            "Scan Completed",
                            "Could not auto-detect camera IP.\n\nTips:\n1. Ensure PoE injector is powered on and LAN cable is firmly plugged in.\n2. Ensure your laptop's Ethernet IPv4 is set to 192.168.1.100.\n3. Default IP 192.168.1.250 will be used."
                        )

                self.root.after(0, _update_ui)
            except Exception as e:
                def _error_ui():
                    self.scan_btn.config(state=tk.NORMAL, text="🔍 Auto-Scan IP")
                    self.log_lbl.config(text=f"Scan error: {e}", fg=self.accent_red)
                self.root.after(0, _error_ui)

        threading.Thread(target=_scan_worker, daemon=True).start()

    def _on_param_change(self, event=None):
        self._update_rtsp_url_field()

    def _update_rtsp_url_field(self):
        ip = self.ip_entry.get().strip()
        port = self.port_entry.get().strip() or "554"
        user = self.user_entry.get().strip()
        pwd = self.pass_entry.get().strip()
        subtype = "0" if "subtype=0" in self.stream_type_var.get() else "1"

        # Standard CP Plus / Dahua RTSP URL pattern:
        # rtsp://username:password@IP:PORT/cam/realmonitor?channel=1&subtype=0
        if user and pwd:
            url = f"rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel=1&subtype={subtype}"
        else:
            url = f"rtsp://{ip}:{port}/cam/realmonitor?channel=1&subtype={subtype}"

        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, url)

    def connect_camera(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please provide a valid RTSP URL.")
            return

        backend_map = {
            "OpenCV TCP (Low-Latency)": "opencv_tcp",
            "GStreamer H.264 (Pipeline)": "gstreamer",
            "GStreamer H.265 / HEVC": "gstreamer_h265",
            "NVIDIA Jetson (HW-Dec)": "jetson",
        }
        selected_backend = backend_map.get(self.backend_var.get(), "opencv_tcp")

        self.streamer = RTSPStreamer(
            rtsp_url=url,
            backend=selected_backend,
            output_dir=self.output_dir,
        )
        self.streamer.start()
        self.is_connected = True

        self.connect_btn.config(state=tk.DISABLED, bg="#475569")
        self.disconnect_btn.config(state=tk.NORMAL, bg=self.accent_red)
        self.snap_btn.config(state=tk.NORMAL)
        self.rec_btn.config(state=tk.NORMAL)
        self.log_lbl.config(text="Connecting to camera RTSP stream...", fg=self.accent_blue)

    def disconnect_camera(self):
        if self.streamer:
            self.streamer.stop()
            self.streamer = None

        self.is_connected = False
        self.connect_btn.config(state=tk.NORMAL, bg=self.accent_green)
        self.disconnect_btn.config(state=tk.DISABLED, bg="#475569")
        self.snap_btn.config(state=tk.DISABLED)
        self.rec_btn.config(state=tk.DISABLED, text="⏺ Start Recording", bg="#b91c1c")

        self.status_badge.config(text="● DISCONNECTED", bg="#334155", fg=self.text_secondary)
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 400
        h = self.canvas.winfo_height() or 300
        self.canvas_text = self.canvas.create_text(
            w // 2, h // 2,
            text="Camera Disconnected",
            fill=self.text_secondary,
            font=("Segoe UI", 12),
            justify=tk.CENTER,
        )
        self.log_lbl.config(text="Disconnected.", fg=self.text_secondary)

    def take_snapshot(self):
        if not self.streamer:
            return
        success, msg = self.streamer.capture_snapshot()
        if success:
            self.log_lbl.config(text=f"✓ Snapshot saved: {Path(msg).name}", fg=self.accent_green)
        else:
            self.log_lbl.config(text=f"✗ Snapshot failed: {msg}", fg=self.accent_red)

    def toggle_recording(self):
        if not self.streamer:
            return

        if not self.streamer._is_recording:
            success, msg = self.streamer.start_recording()
            if success:
                self.rec_btn.config(text="⏹ Stop Recording", bg=self.accent_yellow, fg="#000000")
                self.log_lbl.config(text=f"⏺ Recording: {Path(msg).name}", fg=self.accent_green)
            else:
                self.log_lbl.config(text=f"Recording error: {msg}", fg=self.accent_red)
        else:
            success, msg = self.streamer.stop_recording()
            self.rec_btn.config(text="⏺ Start Recording", bg="#b91c1c", fg="#ffffff")
            self.log_lbl.config(text=f"✓ {msg}", fg=self.accent_green)

    def open_output_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(self.output_dir))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(self.output_dir)])
        else:
            subprocess.run(["xdg-open", str(self.output_dir)])

    def _start_gui_update_loop(self):
        """Runs periodic frame rendering and metrics updates at ~30 FPS."""
        if self.streamer and self.is_connected:
            has_frame, frame_bgr, metrics = self.streamer.get_latest_frame()

            # 1. Update Telemetry Labels
            status = metrics["status"]
            self.stat_labels["status"].config(text=status)
            if status == StreamStatus.CONNECTED:
                self.status_badge.config(text="● LIVE STREAM", bg=self.accent_green, fg="#ffffff")
            elif status == StreamStatus.RECONNECTING:
                self.status_badge.config(text="● RECONNECTING", bg=self.accent_yellow, fg="#000000")
            elif status == StreamStatus.ERROR:
                self.status_badge.config(text="● ERROR", bg=self.accent_red, fg="#ffffff")

            self.stat_labels["resolution"].config(text=metrics["resolution"])
            self.stat_labels["camera_fps"].config(text=f"{metrics['camera_fps']:.1f} FPS")
            self.stat_labels["rx_fps"].config(text=f"{metrics['rx_fps']:.1f} FPS")
            self.stat_labels["render_fps"].config(text=f"{metrics['render_fps']:.1f} FPS")
            self.stat_labels["latency_ms"].config(text=f"{metrics['latency_ms']:.1f} ms")
            self.stat_labels["frames_received"].config(text=str(metrics["frames_received"]))
            self.stat_labels["frames_dropped"].config(text=str(metrics["frames_dropped"]))
            self.stat_labels["reconnect_count"].config(text=str(metrics["reconnect_count"]))

            if metrics["is_recording"]:
                self.stat_labels["recording_status"].config(
                    text=f"Recording ({metrics['recording_frames']} frames)", fg=self.accent_red
                )
            else:
                self.stat_labels["recording_status"].config(text="Idle", fg=self.text_primary)

            if metrics.get("error"):
                self.log_lbl.config(text=metrics["error"], fg=self.accent_red)

            # 2. Render Frame on Canvas
            if has_frame and frame_bgr is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Scale to fit canvas maintaining aspect ratio
                canvas_w = max(self.canvas.winfo_width(), 100)
                canvas_h = max(self.canvas.winfo_height(), 100)

                img_h, img_w = rgb_frame.shape[:2]
                scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w = max(int(img_w * scale), 1)
                new_h = max(int(img_h * scale), 1)

                resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img_pil = Image.fromarray(resized)
                self.current_img_tk = ImageTk.PhotoImage(image=img_pil)

                # Center on canvas
                x_pos = (canvas_w - new_w) // 2
                y_pos = (canvas_h - new_h) // 2

                self.canvas.delete("all")
                self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.current_img_tk)

        # Schedule next update in ~30ms (~33 FPS)
        self.root.after(30, self._start_gui_update_loop)


def main():
    root = tk.Tk()
    app = CameraTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
