#!/usr/bin/env python3
"""pixpack VM sender - display a file as a stream of dense colour frames.

Run this INSIDE the VM. Run host_recv.py on the host, then press SPACE here.

    python vm_send.py payload.zip
    python vm_send.py payload.zip --profile turbo --fps 8

Needs only numpy + pillow (tkinter ships with Python). No OpenCV in the guest.
The profile must match host_recv.py exactly.
"""

from __future__ import annotations

import argparse
import struct
import sys
import threading
import queue
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MAGIC = b"PXV2"
VERSION = 2
HEADER_FMT = "<4sBBIIIQII"
HEADER_BODY = struct.calcsize(HEADER_FMT)      # 34
HEADER_SIZE = HEADER_BODY + 2                  # + crc16
RS_BLOCK = 255
CAL_ROWS = 1
PRIM = 0x11D

# 6x6 ArUco DICT_4X4_50 bitmaps (ids 0-3), row-major MSB first, bit set = white.
MARKER_BITS = (0x016286100, 0x000792500, 0x006184680, 0x012488300)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    name: str
    width: int
    height: int
    block: int
    bpc: int
    ecc_percent: int
    marker_cells: int
    margin: int = 8
    note: str = ""

    @property
    def levels(self) -> int:
        return 1 << self.bpc

    @property
    def step(self) -> int:
        return 255 // (self.levels - 1)

    @property
    def bits_per_cell(self) -> int:
        return 3 * self.bpc

    @property
    def grid_w(self) -> int:
        return (self.width - 2 * self.margin) // self.block

    @property
    def grid_h(self) -> int:
        return (self.height - 2 * self.margin) // self.block

    @property
    def data_cells(self) -> int:
        mc = self.marker_cells
        return (self.grid_w * self.grid_h - 4 * mc * mc
                - CAL_ROWS * max(0, self.grid_w - 2 * mc))

    @property
    def nsym(self) -> int:
        return max(2, min(200, round(RS_BLOCK * self.ecc_percent / 100)))

    @property
    def rs_k(self) -> int:
        return RS_BLOCK - self.nsym

    @property
    def rs_blocks(self) -> int:
        return (self.data_cells * self.bits_per_cell // 8) // RS_BLOCK

    @property
    def payload_bytes(self) -> int:
        return self.rs_blocks * self.rs_k - HEADER_SIZE

    @property
    def marker_px(self) -> int:
        """Marker side in pixels: whole modules, leaving a 1-module quiet zone."""
        room = self.marker_cells * self.block
        return 6 * max(1, int(room * 0.75) // 6)

    @property
    def marker_region(self) -> int:
        return self.marker_cells * self.block

    def frames_for(self, n: int) -> int:
        return max(1, -(-n // self.payload_bytes))


PROFILES = {
    # full-screen 1080p guests
    "fast": Profile("fast", 1840, 1000, 2, 2, 12, 20,
                    note="default; 2x2 cells, 64 colours. tolerant and quick."),
    "turbo": Profile("turbo", 1840, 1000, 1, 2, 12, 38,
                     note="1px cells. needs a pixel-exact VM display path."),
    "max": Profile("max", 1840, 1000, 1, 4, 12, 38,
                   note="1px cells, 4096 colours. fastest, least forgiving."),
    "safe": Profile("safe", 1840, 1000, 4, 2, 25, 11,
                    note="4x4 cells, heavy parity. use if others fail."),
    # smaller guest resolutions - the frame must fit or the markers get clipped
    "fast-md": Profile("fast-md", 1360, 760, 2, 2, 12, 20,
                       note="for guests around 1440x900."),
    "safe-md": Profile("safe-md", 1360, 760, 4, 2, 25, 11,
                       note="for guests around 1440x900, extra robust."),
    "fast-sm": Profile("fast-sm", 1000, 560, 2, 2, 12, 20,
                       note="for guests around 1024x768."),
    "safe-sm": Profile("safe-sm", 1000, 560, 4, 2, 25, 11,
                       note="for guests around 1024x768, extra robust."),
    # last resort: 8px cells, 8 colours, heavy parity. slow but survives almost
    # anything, including a compressed remote-desktop console.
    "tough": Profile("tough", 1840, 1000, 8, 1, 30, 6,
                     note="maximum robustness, low capacity. small files only."),
    "tough-md": Profile("tough-md", 1360, 760, 8, 1, 30, 6,
                        note="maximum robustness for smaller guests."),
}
DEFAULT_PROFILE = "fast"

# widest frame first, so fitting picks the most capacity that will actually show
AUTOFIT_ORDER = ("fast", "fast-md", "fast-sm")
ROBUST_ORDER = ("safe", "safe-md", "safe-sm")
TOUGH_ORDER = ("tough", "tough-md", "tough-md")


def fits(p: Profile, screen_w: int, screen_h: int) -> bool:
    return p.width <= screen_w and p.height <= screen_h


def fit_profile(screen_w: int, screen_h: int, prefer: str = DEFAULT_PROFILE) -> Profile:
    """Largest frame that fits the guest screen, keeping the requested density."""
    wanted = PROFILES[prefer]
    if fits(wanted, screen_w, screen_h):
        return wanted
    if prefer.startswith("tough"):
        order = TOUGH_ORDER
    elif prefer.startswith("safe"):
        order = ROBUST_ORDER
    else:
        order = AUTOFIT_ORDER
    for name in order:
        if fits(PROFILES[name], screen_w, screen_h):
            return PROFILES[name]
    return PROFILES[order[-1]]


# ---------------------------------------------------------------------------
# GF(2^8) Reed-Solomon, batched across every block in the file at once
# ---------------------------------------------------------------------------


def _tables() -> tuple[np.ndarray, np.ndarray]:
    exp = np.zeros(512, dtype=np.uint8)
    log = np.zeros(256, dtype=np.uint8)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM
    exp[255:] = exp[np.arange(255, 512) - 255]
    return exp, log


EXP, LOG = _tables()


def _poly_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.zeros(len(p) + len(q) - 1, dtype=np.uint8)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                if b:
                    out[i + j] ^= EXP[int(LOG[a]) + int(LOG[b])]
    return out


@lru_cache(maxsize=8)
def generator_poly(nsym: int) -> np.ndarray:
    g = np.array([1], dtype=np.uint8)
    for i in range(nsym):
        g = _poly_mul(g, np.array([1, EXP[i]], dtype=np.uint8))
    return g


def rs_encode(blocks: np.ndarray, nsym: int) -> np.ndarray:
    """(n, k) -> (n, k+nsym). One pass over k for every block simultaneously."""
    n, k = blocks.shape
    tail = generator_poly(nsym)[1:]
    log_tail = LOG[tail].astype(np.uint16)

    rem = np.zeros((n, nsym), dtype=np.uint8)
    scratch = np.empty_like(rem)
    for j in range(k):
        feedback = blocks[:, j] ^ rem[:, 0]
        scratch[:, :-1] = rem[:, 1:]
        scratch[:, -1] = 0
        rem, scratch = scratch, rem

        live = feedback != 0
        if live.any():
            logs = LOG[feedback[live]].astype(np.uint16)
            rem[live] ^= EXP[logs[:, None] + log_tail[None, :]]
    return np.concatenate([blocks, rem], axis=1)


# ---------------------------------------------------------------------------
# frame layout and rendering
# ---------------------------------------------------------------------------


def pack_header(idx, total, payload_len, total_len, file_crc, chunk_crc, flags=0):
    body = struct.pack(HEADER_FMT, MAGIC, VERSION, flags, idx, total,
                       payload_len, total_len, file_crc, chunk_crc)
    return body + struct.pack("<H", zlib.crc32(body) & 0xFFFF)


@lru_cache(maxsize=8)
def data_mask(p: Profile) -> np.ndarray:
    mask = np.ones((p.grid_h, p.grid_w), dtype=bool)
    mc = p.marker_cells
    mask[:mc, :mc] = mask[:mc, -mc:] = False
    mask[-mc:, :mc] = mask[-mc:, -mc:] = False
    mask[:CAL_ROWS, mc:-mc] = False
    return mask


@lru_cache(maxsize=8)
def calibration_ramp(p: Profile) -> np.ndarray:
    n = CAL_ROWS * max(0, p.grid_w - 2 * p.marker_cells)
    return (np.arange(n) % p.levels).astype(np.uint8)


def marker_rect(p: Profile, corner: int) -> tuple[int, int]:
    """Top-left pixel of the reserved corner region (0=TL,1=TR,2=BR,3=BL)."""
    mc = p.marker_cells
    row0 = 0 if corner in (0, 1) else p.grid_h - mc
    col0 = 0 if corner in (0, 3) else p.grid_w - mc
    return p.margin + col0 * p.block, p.margin + row0 * p.block


@lru_cache(maxsize=8)
def marker_image(corner: int, side: int) -> np.ndarray:
    bits = np.array([(MARKER_BITS[corner] >> (35 - i)) & 1 for i in range(36)],
                    dtype=np.uint8).reshape(6, 6)
    return np.kron(bits * 255, np.ones((side // 6, side // 6), dtype=np.uint8))


def render(payload: bytes, p: Profile) -> np.ndarray:
    mask = data_mask(p)
    n_cells = int(mask.sum())

    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    need = n_cells * p.bits_per_cell
    if bits.size < need:
        bits = np.pad(bits, (0, need - bits.size))
    bits = bits[:need].reshape(n_cells, 3, p.bpc)
    weights = (1 << np.arange(p.bpc - 1, -1, -1)).astype(np.uint32)
    idx = (bits * weights).sum(axis=2).astype(np.uint16)

    grid = np.zeros((p.grid_h, p.grid_w, 3), dtype=np.uint8)
    grid[mask] = (idx * p.step).astype(np.uint8)

    mc = p.marker_cells
    ramp = calibration_ramp(p)
    if ramp.size:
        grid[:CAL_ROWS, mc:-mc] = (ramp.astype(np.uint16) * p.step
                                   ).astype(np.uint8).reshape(CAL_ROWS, -1)[:, :, None]

    canvas = np.full((p.height, p.width, 3), 255, dtype=np.uint8)
    body = np.repeat(np.repeat(grid, p.block, axis=0), p.block, axis=1)
    canvas[p.margin:p.margin + body.shape[0], p.margin:p.margin + body.shape[1]] = body

    side, region = p.marker_px, p.marker_region
    slack = (region - side) // 2
    for corner in range(4):
        rx, ry = marker_rect(p, corner)
        # whiten only the reserved cells - any wider and it eats real data
        canvas[ry:ry + region, rx:rx + region] = 255
        canvas[ry + slack:ry + slack + side, rx + slack:rx + slack + side] = \
            marker_image(corner, side)[:, :, None]
    return canvas


# ---------------------------------------------------------------------------
# encoding the whole file up front
# ---------------------------------------------------------------------------


def encode_file(data: bytes, p: Profile, progress=None) -> list[bytes]:
    """Post-FEC bytes for every frame. Rendering happens later, on demand."""
    cap, k, nblocks = p.payload_bytes, p.rs_k, p.rs_blocks
    total = p.frames_for(len(data))
    file_crc = zlib.crc32(data) & 0xFFFFFFFF

    frames: list[bytes] = []
    batch = max(1, 24 * 1024 * 1024 // (nblocks * k))  # ~24 MB of blocks per pass

    for start in range(0, total, batch):
        stop = min(start + batch, total)
        raw = np.zeros(((stop - start) * nblocks, k), dtype=np.uint8)

        for i in range(start, stop):
            chunk = data[i * cap:(i + 1) * cap]
            blob = pack_header(i, total, len(chunk), len(data), file_crc,
                               zlib.crc32(chunk) & 0xFFFFFFFF) + chunk
            flat = np.frombuffer(blob.ljust(nblocks * k, b"\0"), dtype=np.uint8)
            raw[(i - start) * nblocks:(i - start + 1) * nblocks] = flat.reshape(nblocks, k)

        coded = rs_encode(raw, p.nsym)
        for i in range(stop - start):
            frames.append(coded[i * nblocks:(i + 1) * nblocks].T.tobytes())
        if progress:
            progress(len(frames), total)
    return frames


# ---------------------------------------------------------------------------
# full-screen player
# ---------------------------------------------------------------------------


class Player:
    def __init__(self, frames: list[bytes], p: Profile, fps: float, source: str,
                 autostart: bool = False, parent=None, status=None) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.frames, self.p, self.source = frames, p, source
        self.delay = max(10, int(1000 / fps))
        self.index = 0
        self.started = False
        self.alive = True
        self.passes = 0
        self.status = status or (lambda *_: None)
        self.standalone = parent is None
        self._photo = None

        self.ready: queue.Queue = queue.Queue(maxsize=4)
        self.wanted: queue.Queue = queue.Queue()

        self.root = tk.Tk() if self.standalone else tk.Toplevel(parent)
        self.root.title("pixpack sender")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)  # nothing may cover the frames
        self.root.config(cursor="none")
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.root.bind("<Escape>", lambda _: self.close())
        self.root.bind("<space>", self.toggle)
        self.root.bind("<Up>", lambda _: self.nudge(1.25))
        self.root.bind("<Down>", lambda _: self.nudge(0.8))
        self.root.bind("<Left>", lambda _: self.step(-1))
        self.root.bind("<Right>", lambda _: self.step(1))
        self.root.bind("h", lambda _: self.splash())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.focus_force()
        self.screen = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())

        threading.Thread(target=self._render_worker, daemon=True).start()
        self.splash()
        if autostart:
            self.root.after(500, self.toggle)

    def close(self) -> None:
        self.started = False
        self.alive = False
        self.status("stopped", 0, 0)
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def nudge(self, factor: float) -> None:
        self.delay = max(20, min(2000, int(self.delay / factor)))
        self.status("playing", self.index, self.passes)

    def _render_worker(self) -> None:
        while True:
            index = self.wanted.get()
            if index is None:
                return
            self.ready.put((index, render(self.frames[index], self.p)))

    def splash(self) -> None:
        self.canvas.delete("all")
        w, h = self.screen
        lines = [
            ("pixpack sender", 34, "#5aa9e6", -150),
            (self.source, 16, "#8b93a7", -95),
            (f"{len(self.frames):,} frames   profile {self.p.name}   "
             f"{1000 / self.delay:.1f} fps", 15, "#e8eaf0", -40),
            (f"frame {self.p.width}x{self.p.height} on a {w}x{h} screen", 13, "#8b93a7", 0),
            ("1.  start the receiver on the host", 16, "#e8eaf0", 70),
            ("2.  press SPACE here to begin", 16, "#e8eaf0", 105),
            ("SPACE play/hold   LEFT/RIGHT step   UP/DOWN speed   H this screen   ESC quit",
             12, "#8b93a7", 165),
        ]
        for text, size, colour, dy in lines:
            self.canvas.create_text(w // 2, h // 2 + dy, text=text, fill=colour,
                                    font=("Segoe UI", size))
        if self.p.width > w or self.p.height > h:
            self.canvas.create_text(
                w // 2, h // 2 + 210,
                text=f"WARNING: frame is {self.p.width}x{self.p.height} but this screen "
                     f"is only {w}x{h} - the markers will be cut off and nothing "
                     f"will decode. Pick a smaller profile.",
                fill="#ff6b6b", font=("Segoe UI", 14), width=w - 200, justify="center")

    def toggle(self, _=None) -> None:
        self.started = not self.started
        if self.started:
            for _ in range(2):
                self.wanted.put(self.index)
                self.index = (self.index + 1) % len(self.frames)
            self.root.after(0, self.tick)
        # pausing deliberately leaves the current frame up: a completely static
        # frame is the only way to tell a torn capture from a real decode fault
        self.status("playing" if self.started else "paused", self.index, self.passes)

    def step(self, delta: int) -> None:
        """Move one frame while paused, so a single frame can be held still."""
        if self.started:
            return
        self.index = (self.index + delta) % len(self.frames)
        self.show_now(self.index)
        self.status("paused", self.index, self.passes)

    def show_now(self, index: int) -> None:
        image = self.ImageTk.PhotoImage(self.Image.fromarray(render(self.frames[index], self.p)))
        self.canvas.delete("all")
        self._photo = image
        self.canvas.create_image(self.screen[0] // 2, self.screen[1] // 2,
                                 image=image, anchor="center")

    def tick(self) -> None:
        if not self.started or not self.alive:
            return
        try:
            _, array = self.ready.get_nowait()
        except queue.Empty:
            self.root.after(5, self.tick)
            return

        try:
            image = self.ImageTk.PhotoImage(self.Image.fromarray(array))
            self.canvas.delete("all")
            self._photo = image  # keep a reference or Tk drops it mid-draw
            self.canvas.create_image(self.screen[0] // 2, self.screen[1] // 2,
                                     image=image, anchor="center")
        except self.tk.TclError:  # window closed between scheduling and drawing
            self.alive = False
            return

        if self.index == 0:
            self.passes += 1
        self.status("playing", self.index, self.passes)
        self.wanted.put(self.index)
        self.index = (self.index + 1) % len(self.frames)
        self.root.after(self.delay, self.tick)

    def run(self) -> None:
        if self.standalone:
            self.root.mainloop()


def set_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    for call in (lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),
                 lambda: ctypes.windll.user32.SetProcessDPIAware()):
        try:
            call()
            return
        except (AttributeError, OSError):
            continue


# ---------------------------------------------------------------------------
# control panel
# ---------------------------------------------------------------------------

BG = "#F3F0E8"
CARD = "#FFFFFF"
EDGE = "#D6D1C2"
TEXT = "#171615"
MUTED = "#5F5C55"
ACCENT = "#C15F3C"
ACCENT_DARK = "#A44A2B"
INK = "#2E2B27"
INK_HOVER = "#46423C"
OK = "#1E6B45"
BAD = "#A3341F"

# app icon, embedded so the file stays self-contained
ICON_PNG = (
    "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"
)


def apply_icon(window) -> None:
    import tkinter as tk

    try:
        window._pix_icon = tk.PhotoImage(data=ICON_PNG)  # keep a reference alive
        window.iconphoto(True, window._pix_icon)
    except Exception:
        pass  # icons are cosmetic; never let one stop the app


class SenderGUI:
    def __init__(self, preload: Path | None = None) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        self.tk, self.filedialog = tk, filedialog
        self.source: Path | None = None
        self.frames: list[bytes] = []
        self.player: Player | None = None
        self.busy = False
        self.events: queue.Queue = queue.Queue()

        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD

            self.root = TkinterDnD.Tk()
            self.dnd = DND_FILES
        except ImportError:
            self.root = tk.Tk()
            self.dnd = None

        self.root.title("pixpack sender")
        self.root.geometry("560x600")
        self.root.configure(bg=BG)
        self.root.minsize(520, 560)
        apply_icon(self.root)
        self.screen = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("pix.Horizontal.TProgressbar", troughcolor=EDGE,
                        background=ACCENT, bordercolor=EDGE, lightcolor=ACCENT,
                        darkcolor=ACCENT, thickness=8)

        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=26, pady=(20, 12))
        tk.Label(head, text="pixpack sender", bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 19)).pack(anchor="w")
        tk.Label(head, text="run this on the machine holding the file", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        drop_card = self._card()
        self.drop = tk.Frame(drop_card, bg=CARD)
        self.drop.pack(fill="x", ipady=20)
        self.drop_label = tk.Label(
            self.drop, bg=CARD, fg=TEXT, font=("Segoe UI", 11), justify="center",
            text="Drop a file here" if self.dnd else "Click to choose a file")
        self.drop_label.pack()
        self.drop_hint = tk.Label(self.drop, text="or click to browse", bg=CARD,
                                  fg=MUTED, font=("Segoe UI", 8))
        self.drop_hint.pack()

        for widget in (self.drop, self.drop_label, self.drop_hint):
            widget.bind("<Button-1>", lambda _: self.browse())
            if self.dnd:
                # tkinterdnd2 patches these on at runtime
                getattr(widget, "drop_target_register")(self.dnd)
                getattr(widget, "dnd_bind")("<<Drop>>", self.on_drop)

        settings = self._card()
        row = tk.Frame(settings, bg=CARD)
        row.pack(fill="x", padx=14, pady=(12, 10))
        tk.Label(row, text="Profile", bg=CARD, fg=MUTED, width=7, anchor="w",
                 font=("Segoe UI", 9)).pack(side="left")
        self.profile = tk.StringVar(value=fit_profile(*self.screen, DEFAULT_PROFILE).name)
        ttk.Combobox(row, textvariable=self.profile, values=sorted(PROFILES),
                     state="readonly", width=11).pack(side="left")
        tk.Label(row, text="Speed", bg=CARD, fg=MUTED, font=("Segoe UI", 9)
                 ).pack(side="left", padx=(18, 6))
        self.fps = tk.StringVar(value="8")
        ttk.Spinbox(row, textvariable=self.fps, from_=1, to=30, width=5,
                    command=self.refresh).pack(side="left")
        tk.Label(row, text="fps", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(4, 0))
        self.profile.trace_add("write", lambda *_: self.refresh())
        self.fps.trace_add("write", lambda *_: self.refresh())

        self.info = tk.Label(settings, text="", bg=CARD, fg=TEXT, justify="left",
                             font=("Segoe UI", 9), anchor="w")
        self.info.pack(fill="x", padx=14)
        self.fit = tk.Label(settings, text="", bg=CARD, fg=MUTED, justify="left",
                            font=("Segoe UI", 9), anchor="w", wraplength=460)
        self.fit.pack(fill="x", padx=14, pady=(4, 12))

        progress = self._card()
        self.bar = ttk.Progressbar(progress, style="pix.Horizontal.TProgressbar")
        self.bar.pack(fill="x", padx=14, pady=(14, 8))
        self.state = tk.Label(progress, text="waiting for a file", bg=CARD, fg=TEXT,
                              font=("Segoe UI Semibold", 11))
        self.state.pack(pady=(0, 14))

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=26, pady=(4, 8))
        self.play_button = self._button(bar, "Play full screen", self.play, kind="primary")
        self.play_button.pack(side="left")
        self.play_button.configure(state="disabled")
        self.stop_button = self._button(bar, "Stop", self.stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.stop_button.configure(state="disabled")

        tk.Label(self.root, bg=BG, fg=MUTED, font=("Segoe UI", 8), justify="left",
                 text="while playing:   SPACE hold a frame    LEFT / RIGHT step    "
                      "UP / DOWN speed    ESC stop"
                 ).pack(anchor="w", padx=26, pady=(0, 16))

        self.refresh()
        self.root.after(80, self._drain)
        if preload and preload.is_file():
            self.load(preload)

    def _card(self, expand: bool = False):
        outer = self.tk.Frame(self.root, bg=EDGE)
        outer.pack(fill="both" if expand else "x", expand=expand, padx=26, pady=(0, 12))
        inner = self.tk.Frame(outer, bg=CARD)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return inner

    def _button(self, parent, text, command, kind="normal"):
        colours = {
            "primary": (ACCENT, "#FFFFFF", ACCENT_DARK),
            "normal": (INK, "#FFFFFF", INK_HOVER),
        }[kind]
        return self.tk.Button(
            parent, text=text, command=command, bg=colours[0], fg=colours[1],
            activebackground=colours[2], activeforeground="#FFFFFF",
            disabledforeground="#C9C4BA",
            font=("Segoe UI Semibold", 9), bd=0, relief="flat",
            padx=16, pady=8, cursor="hand2", highlightthickness=0)

    # -- input ------------------------------------------------------------

    def browse(self) -> None:
        if self.busy:
            return
        chosen = self.filedialog.askopenfilename(title="Pick a file to send")
        if chosen:
            self.load(Path(chosen))

    def on_drop(self, event) -> None:
        if self.busy:
            return
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self.load(Path(paths[0]))

    def load(self, path: Path) -> None:
        if not path.is_file():
            self.state.configure(text=f"not a file: {path.name}")
            return
        self.source = path
        self.frames = []
        self.play_button.configure(state="disabled")
        self.drop_label.configure(text=path.name)
        self.refresh()
        self.encode()

    def refresh(self) -> None:
        p = PROFILES[self.profile.get()]
        try:
            fps = max(1.0, float(self.fps.get()))
        except (TypeError, ValueError):
            fps = 8.0

        sw, sh = self.screen
        if fits(p, sw, sh):
            self.fit.configure(
                text=f"screen {sw}x{sh} \u00b7 frame {p.width}x{p.height} fits", fg=OK)
        else:
            suggestion = fit_profile(sw, sh, p.name).name
            self.fit.configure(
                text=f"this screen is only {sw}x{sh} but the frame is "
                     f"{p.width}x{p.height}. It will be cut off and nothing will "
                     f"decode - switch to '{suggestion}'.", fg=BAD)

        if self.source and self.source.is_file():
            size = self.source.stat().st_size
            n = p.frames_for(size)
            self.info.configure(
                text=f"{size / 1048576:.1f} MiB  \u2192  {n:,} frames of "
                     f"{p.payload_bytes / 1024:.0f} KiB\n"
                     f"one pass takes {n / fps / 60:.1f} min at {fps:g} fps")
        else:
            self.info.configure(
                text=f"{p.payload_bytes / 1024:.0f} KiB per frame\n"
                     f"200 MB would be {p.frames_for(200_000_000):,} frames")

    # -- encoding ---------------------------------------------------------

    def encode(self) -> None:
        if not self.source:
            return
        self.busy = True
        self.state.configure(text="encoding ...")
        profile = PROFILES[self.profile.get()]
        threading.Thread(target=self._encode_worker, args=(self.source, profile),
                         daemon=True).start()

    def _encode_worker(self, source: Path, profile: Profile) -> None:
        try:
            data = source.read_bytes()
            frames = encode_file(
                data, profile,
                lambda done, of: self.events.put(("progress", (done, of))))
            self.events.put(("encoded", frames))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _drain(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    done, of = value
                    self.bar.configure(maximum=of, value=done)
                    self.state.configure(text=f"encoding {done:,}/{of:,}")
                elif kind == "encoded":
                    self.busy = False
                    self.frames = value
                    self.bar.configure(value=self.bar["maximum"])
                    self.state.configure(text=f"ready - {len(value):,} frames encoded")
                    self.play_button.configure(state="normal")
                elif kind == "error":
                    self.busy = False
                    self.state.configure(text=value)
                elif kind == "player":
                    label, index, passes = value
                    if label == "playing":
                        self.state.configure(
                            text=f"playing  frame {index + 1:,}/{len(self.frames):,}"
                                 f"   pass {passes}")
                    else:
                        self.state.configure(text=label)
                    if label == "stopped":
                        self.player = None
                        if self.root.winfo_exists():
                            self.root.deiconify()
                        self.stop_button.configure(state="disabled")
                        self.play_button.configure(state="normal")
        except queue.Empty:
            pass
        except self.tk.TclError:  # main window closed mid-update
            return
        self.root.after(80, self._drain)

    # -- playback ---------------------------------------------------------

    def play(self) -> None:
        if not self.frames or self.player is not None:
            return
        try:
            fps = max(1.0, float(self.fps.get()))
        except (TypeError, ValueError):
            fps = 8.0
        self.player = Player(
            self.frames, PROFILES[self.profile.get()], fps,
            self.source.name if self.source else "", parent=self.root,
            status=lambda *a: self.events.put(("player", a)))
        self.root.withdraw()  # keep the panel from covering the frames
        self.play_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def stop(self) -> None:
        if self.player is not None:
            self.player.close()
        if self.root.winfo_exists():
            self.root.deiconify()

    def run(self) -> None:
        self.root.mainloop()


def export_frames(source: Path, profile: Profile, out_dir: Path) -> int:
    """Write the frames to disk as PNGs instead of showing them."""
    from PIL import Image

    data = source.read_bytes()
    total = profile.frames_for(len(data))
    print(f"file     {source.name}  ({len(data) / 1048576:.1f} MiB)")
    print(f"profile  {profile.name}  ->  {total:,} frames")
    print(f"folder   {out_dir}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.png"))
    if existing:
        print(f"error: {out_dir} already holds {len(existing)} frame_*.png files.",
              file=sys.stderr)
        print("       pick an empty folder so old frames cannot mix in.", file=sys.stderr)
        return 2

    print("encoding ...")
    frames = encode_file(data, profile,
                         lambda d, o: print(f"\r  {d:,}/{o:,}", end="", flush=True))

    print("\nwriting ...")
    written = 0
    for i, payload in enumerate(frames):
        path = out_dir / f"frame_{i:05d}.png"
        Image.fromarray(render(payload, profile)).save(path, compress_level=1)
        written += path.stat().st_size
        if (i + 1) % 10 == 0 or i + 1 == len(frames):
            print(f"\r  {i + 1:,}/{len(frames):,}", end="", flush=True)

    print(f"\n\nwrote {len(frames):,} PNGs to {out_dir}  "
          f"({written / 1048576:.1f} MiB on disk)")
    print(f"rebuild with:  python host_recv.py --from-frames {out_dir} "
          f"-o {source.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", type=Path, help="file to send")
    ap.add_argument("-p", "--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    ap.add_argument("--fps", type=float, default=8.0, help="frames per second")
    ap.add_argument("--cli", action="store_true", help="no window, straight to full screen")
    ap.add_argument("--export", type=Path, metavar="FOLDER",
                    help="write the frames to a folder as PNGs instead of playing them")
    ap.add_argument("--autostart", action="store_true",
                    help="skip the splash and start playing immediately")
    ap.add_argument("--list-profiles", action="store_true")
    args = ap.parse_args()

    if args.list_profiles:
        for prof in PROFILES.values():
            mb = prof.payload_bytes / 1048576
            print(f"  {prof.name:<6} {prof.width}x{prof.height} block={prof.block} "
                  f"{prof.levels ** 3:>5} colours  {prof.payload_bytes:>9,} B/frame "
                  f"({mb:.2f} MiB)  200MB -> {prof.frames_for(200_000_000):,} frames")
            print(f"         {prof.note}")
        return 0

    set_dpi_aware()

    if args.export is not None:
        if not args.source or not args.source.is_file():
            print("error: --export needs a file to encode", file=sys.stderr)
            return 2
        return export_frames(args.source, PROFILES[args.profile], args.export)

    if not args.cli:
        SenderGUI(args.source).run()
        return 0

    if not args.source or not args.source.is_file():
        print("error: --cli needs a file to send (see --help)", file=sys.stderr)
        return 2

    profile = PROFILES[args.profile]
    data = args.source.read_bytes()
    total = profile.frames_for(len(data))

    print(f"file     {args.source.name}  ({len(data) / 1048576:.1f} MiB)")
    print(f"profile  {args.profile}  ->  {total:,} frames "
          f"({profile.payload_bytes:,} B each)")
    print(f"at {args.fps:g} fps one pass takes {total / args.fps / 60:.1f} min\n")
    print("encoding ...")

    frames = encode_file(data, profile,
                         lambda d, o: print(f"\r  {d:,}/{o:,} frames", end="", flush=True))
    print("\n\nready - press SPACE in the window once the host is listening\n")

    Player(frames, profile, args.fps, args.source.name, args.autostart).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
