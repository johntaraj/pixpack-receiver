#!/usr/bin/env python3
"""pixpack host receiver - rebuild a file from pixpack frames shown on screen.

Run this on the HOST while vm_send.py plays inside the VM.

    python host_recv.py -o payload.zip
    python host_recv.py -o payload.zip --profile turbo --workers 6
    python host_recv.py -o payload.zip --select

By default it watches every display and finds the frame wherever the VM window
sits. The profile must match vm_send.py exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue

import cv2
import mss
import numpy as np

try:
    from reedsolo import RSCodec, ReedSolomonError

    HAVE_RS = True
except ImportError:  # retries cover most damage; repair is a bonus
    HAVE_RS = False

MAGIC = b"PXV2"
VERSION = 2
HEADER_FMT = "<4sBBIIIQII"
HEADER_BODY = struct.calcsize(HEADER_FMT)
HEADER_SIZE = HEADER_BODY + 2
RS_BLOCK = 255
CAL_ROWS = 1
MSS_CLASS = getattr(mss, "MSS", mss.mss)


class FrameError(Exception):
    pass


# ---------------------------------------------------------------------------
# profile - must stay identical to vm_send.py
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
        room = self.marker_cells * self.block
        return 6 * max(1, int(room * 0.75) // 6)

    @property
    def marker_region(self) -> int:
        return self.marker_cells * self.block


PROFILES = {
    "fast": Profile("fast", 1840, 1000, 2, 2, 12, 20),
    "turbo": Profile("turbo", 1840, 1000, 1, 2, 12, 38),
    "max": Profile("max", 1840, 1000, 1, 4, 12, 38),
    "safe": Profile("safe", 1840, 1000, 4, 2, 25, 11),
    "fast-md": Profile("fast-md", 1360, 760, 2, 2, 12, 20),
    "safe-md": Profile("safe-md", 1360, 760, 4, 2, 25, 11),
    "fast-sm": Profile("fast-sm", 1000, 560, 2, 2, 12, 20),
    "safe-sm": Profile("safe-sm", 1000, 560, 4, 2, 25, 11),
    "tough": Profile("tough", 1840, 1000, 8, 1, 30, 6),
    "tough-md": Profile("tough-md", 1360, 760, 8, 1, 30, 6),
}
DEFAULT_PROFILE = "auto"

# most likely first; auto-detect stops at the first one that decodes
DETECT_ORDER = ("fast", "fast-md", "fast-sm", "safe", "safe-md", "safe-sm",
                "tough", "tough-md", "turbo", "max")
CHOICES = ("auto",) + tuple(sorted(PROFILES))

MARKER_BITS = (0x016286100, 0x000792500, 0x006184680, 0x012488300)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


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


def marker_origin(p: Profile, corner: int) -> tuple[int, int]:
    mc = p.marker_cells
    row0 = 0 if corner in (0, 1) else p.grid_h - mc
    col0 = 0 if corner in (0, 3) else p.grid_w - mc
    slack = (p.marker_region - p.marker_px) // 2
    return p.margin + col0 * p.block + slack, p.margin + row0 * p.block + slack


@lru_cache(maxsize=8)
def marker_image(corner: int, side: int) -> np.ndarray:
    bits = np.array([(MARKER_BITS[corner] >> (35 - i)) & 1 for i in range(36)],
                    dtype=np.uint8).reshape(6, 6)
    return np.kron(bits * 255, np.ones((side // 6, side // 6), dtype=np.uint8))


@lru_cache(maxsize=1)
def detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 10
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)


# ---------------------------------------------------------------------------
# locating the frame in a screen grab
# ---------------------------------------------------------------------------


def refine_origin(gray: np.ndarray, p: Profile, x0: int, y0: int,
                  radius: int = 8) -> tuple[int, int] | None:
    """Snap to the exact integer origin by matching the top-left marker.

    ArUco's sub-pixel corners carry a half-pixel convention worth about a pixel
    of scale error across the frame, which is fatal for 1-2px cells.
    """
    mx, my = marker_origin(p, 0)
    template = marker_image(0, p.marker_px)
    ex, ey = x0 + mx, y0 + my

    sx0, sy0 = max(0, ex - radius), max(0, ey - radius)
    sx1 = min(gray.shape[1], ex + template.shape[1] + radius)
    sy1 = min(gray.shape[0], ey + template.shape[0] + radius)
    window = gray[sy0:sy1, sx0:sx1]
    if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
        return None

    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    if score < 0.5:
        return None
    return sx0 + loc[0] - mx, sy0 + loc[1] - my


def detect_markers(image: np.ndarray):
    """One ArUco pass over a grab. Reused across every candidate profile."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector().detectMarkers(gray)
    return gray, corners, ids


def locate_from(image, gray, corners, ids, p: Profile):
    """Return (canonical frame, integer origin if it was a clean 1:1 crop)."""
    if ids is None:
        return None

    src, dst = [], []
    for quad, marker_id in zip(corners, ids.flatten()):
        if marker_id > 3:
            continue
        mx, my = marker_origin(p, int(marker_id))
        side = p.marker_px
        src.append(quad.reshape(4, 2))
        dst.append(np.array([[mx, my], [mx + side, my],
                             [mx + side, my + side], [mx, my + side]], dtype=np.float32))
    if len(src) < 2:
        return None

    matrix, _ = cv2.findHomography(np.concatenate(src).astype(np.float32),
                                   np.concatenate(dst).astype(np.float32),
                                   cv2.RANSAC, 3.0)
    if matrix is None:
        return None

    quad = np.array([[0, 0], [p.width, 0], [p.width, p.height], [0, p.height]],
                    dtype=np.float32)
    try:
        back = cv2.perspectiveTransform(quad.reshape(1, 4, 2),
                                        np.linalg.inv(matrix)).reshape(4, 2)
    except np.linalg.LinAlgError:
        return None

    axis_aligned = (
        abs(back[1][0] - back[0][0] - p.width) < 2.0
        and abs(back[2][0] - back[3][0] - p.width) < 2.0
        and abs(back[3][1] - back[0][1] - p.height) < 2.0
        and abs(back[2][1] - back[1][1] - p.height) < 2.0
        and max(abs(back[0][1] - back[1][1]), abs(back[3][1] - back[2][1]),
                abs(back[0][0] - back[3][0]), abs(back[1][0] - back[2][0])) < 1.5)

    if axis_aligned:
        spot = refine_origin(gray, p, round(float(back[0][0])), round(float(back[0][1])))
        if spot is not None:
            x0, y0 = spot
            h, w = image.shape[:2]
            if 0 <= x0 <= w - p.width and 0 <= y0 <= h - p.height:
                return image[y0:y0 + p.height, x0:x0 + p.width], (x0, y0)

    return cv2.warpPerspective(image, matrix, (p.width, p.height),
                               flags=cv2.INTER_LINEAR), None


def locate(image: np.ndarray, p: Profile):
    gray, corners, ids = detect_markers(image)
    return locate_from(image, gray, corners, ids, p)


def expected_span(p: Profile) -> tuple[int, int]:
    """Outer marker-to-marker distance, which identifies a profile geometrically."""
    return ((p.grid_w - p.marker_cells) * p.block + p.marker_px,
            (p.grid_h - p.marker_cells) * p.block + p.marker_px)


def marker_span(corners) -> tuple[float, float, float, float]:
    pts = np.concatenate([q.reshape(4, 2) for q in corners])
    return (pts[:, 0].min(), pts[:, 1].min(),
            pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())


def rank_profiles(corners, limit: int = 3, tolerance: float = 0.06) -> list[Profile]:
    """Plausible profiles, closest geometric match first.

    Marker span pins the geometry down, so anything far off is not worth the
    cost of a trial decode. The best match is always kept, even if the frame is
    scaled and nothing lands inside the tolerance.
    """
    _, _, sw, sh = marker_span(corners)
    scored = []
    for name in DETECT_ORDER:
        ex, ey = expected_span(PROFILES[name])
        relative = abs(ex - sw) / max(sw, 1.0) + abs(ey - sh) / max(sh, 1.0)
        scored.append((relative, name))
    scored.sort()

    keep = [PROFILES[n] for r, n in scored if r <= tolerance][:limit]
    return keep or [PROFILES[scored[0][1]]]


def identify(image, gray, corners, ids, allow_repair: bool = False) -> Profile | None:
    """Pick the profile for an already-detected set of markers.

    The strict pass is cheap. Error correction is pure Python and costs seconds
    per profile, so it is only ever used on the two best geometric matches, and
    only when the caller says it can afford it.
    """
    candidates = rank_profiles(corners)
    for p in candidates:
        found = locate_from(image, gray, corners, ids, p)
        if found is None:
            continue
        try:
            decode(found[0], p, repair_ok=False)
        except (FrameError, ValueError):
            continue
        return p

    if not allow_repair:
        return None
    for p in candidates[:2]:
        found = locate_from(image, gray, corners, ids, p)
        if found is None:
            continue
        try:
            decode(found[0], p, repair_ok=True)
        except (FrameError, ValueError):
            continue
        return p
    return None


def detect_profile(image: np.ndarray, allow_repair: bool = True) -> Profile | None:
    """Work out which profile is on screen, using a single marker pass."""
    gray, corners, ids = detect_markers(image)
    if ids is None or len(corners) == 0:
        return None
    return identify(image, gray, corners, ids, allow_repair)


def _describe_attempt(image, gray, corners, ids, p: Profile) -> tuple[list[str], bool]:
    """Returns (report lines, geometry_and_colour_were_perfect)."""
    out = [f"  trying '{p.name}' ({p.width}x{p.height}, "
           f"expects span {expected_span(p)[0]}x{expected_span(p)[1]})"]
    found = locate_from(image, gray, corners, ids, p)
    if found is None:
        out.append("    could not lift the frame out")
        return out, False

    canonical, origin = found
    exact = origin is not None
    out.append(f"    crop: {'exact 1:1 at ' + str(origin) if exact else 'WARPED (scaled)'}")

    raw = frame_bytes(canonical, p)
    try:
        parse(deinterleave(raw, p)[:, :p.rs_k].tobytes())
        out.append("    decodes CLEANLY - this profile works")
        return out, True
    except FrameError as exc:
        out.append(f"    clean read failed: {exc}")

    try:
        parse(repair(raw, p))
        out.append("    decodes AFTER error correction - usable but lossy")
        return out, True
    except (FrameError, ValueError) as exc:
        out.append(f"    still fails after error correction: {exc}")

    values = sample_cells(canonical, p)
    seen = [round(float(v)) for v in centroids(values, p)]
    ideal = [i * p.step for i in range(p.levels)]
    out.append(f"    colour levels seen  {seen[:8]}")
    out.append(f"    colour levels wanted {ideal[:8]}")
    return out, exact and seen == ideal


def diagnose(image: np.ndarray) -> str:
    """Plain-language description of what the camera can actually see."""
    gray, corners, ids = detect_markers(image)
    h, w = image.shape[:2]
    lines = [f"capture area: {w}x{h}"]

    if ids is None or len(ids) == 0:
        lines.append("no corner markers found.")
        lines.append("- is the sender actually playing (not on the splash screen)?")
        lines.append("- is the whole frame inside the capture area?")
        lines.append("- is anything covering the VM window?")
        return "\n".join(lines)

    flat = sorted(int(i) for i in ids.flatten())
    lines.append(f"markers found: {flat}  (need 0,1,2,3)")

    x0, y0, sw, sh = marker_span(corners)
    lines.append(f"marker span: {sw:.0f}x{sh:.0f} px at ({x0:.0f}, {y0:.0f})")
    if len([i for i in flat if i <= 3]) < 4:
        lines.append("fewer than 4 corners visible - the frame is probably clipped.")

    pristine = False
    for p in rank_profiles(corners, limit=2):
        report, perfect = _describe_attempt(image, gray, corners, ids, p)
        lines.extend(report)
        pristine = pristine or perfect

    found = detect_profile(image)
    if found:
        lines.append(f"VERDICT: decodes as {found.name}")
    elif pristine:
        lines.append("VERDICT: geometry and colour are PERFECT but the checksum fails.")
        lines.append("  That means the grab caught the screen mid-update, so the top and")
        lines.append("  bottom of the image come from two different frames.")
        lines.append("  - drop the sender's FPS right down (try 2) and retry")
        lines.append("  - or press SPACE in the sender to hold one frame still: if that")
        lines.append("    decodes, the display simply cannot keep up with the frame rate")
    else:
        lines.append("VERDICT: nothing decodes")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# frame decoding
# ---------------------------------------------------------------------------


def sample_cells(img: np.ndarray, p: Profile) -> np.ndarray:
    h, w = p.grid_h * p.block, p.grid_w * p.block
    crop = img[p.margin:p.margin + h, p.margin:p.margin + w]
    if p.block == 1:
        return crop.astype(np.float32)

    cells = crop.astype(np.float32).reshape(
        p.grid_h, p.block, p.grid_w, p.block, 3).transpose(0, 2, 1, 3, 4)
    inset = p.block // 4
    if inset:
        cells = cells[:, :, inset:p.block - inset, inset:p.block - inset, :]
    return cells.reshape(p.grid_h, p.grid_w, -1, 3).mean(axis=2)


def centroids(values: np.ndarray, p: Profile) -> np.ndarray:
    ideal = np.arange(p.levels, dtype=np.float32) * p.step
    ramp = calibration_ramp(p)
    if ramp.size == 0:
        return ideal

    mc = p.marker_cells
    seen = values[:CAL_ROWS, mc:-mc].reshape(-1, 3).mean(axis=1)
    if seen.size < ramp.size:
        return ideal

    out = ideal.copy()
    for level in range(p.levels):
        hit = seen[:ramp.size][ramp == level]
        if hit.size:
            out[level] = float(hit.mean())
    return np.sort(out)


def frame_bytes(img: np.ndarray, p: Profile) -> bytes:
    values = sample_cells(img, p)
    bounds = centroids(values, p)
    edges = (bounds[:-1] + bounds[1:]) * 0.5

    flat = values[data_mask(p)]
    idx = np.searchsorted(edges, flat).astype(np.uint32)
    shifts = np.arange(p.bpc - 1, -1, -1, dtype=np.uint32)
    bits = ((idx[:, :, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits[:bits.size - (bits.size % 8)]).tobytes()


def parse(blob: bytes) -> tuple[int, int, int, int, bytes]:
    body = blob[:HEADER_BODY]
    if len(blob) < HEADER_SIZE:
        raise FrameError("short frame")
    if (zlib.crc32(body) & 0xFFFF) != struct.unpack("<H", blob[HEADER_BODY:HEADER_SIZE])[0]:
        raise FrameError("header crc")

    magic, version, _flags, idx, total, plen, tlen, fcrc, ccrc = struct.unpack(
        HEADER_FMT, body)
    if magic != MAGIC or version != VERSION:
        raise FrameError("not a pixpack v2 frame")
    if plen > len(blob) - HEADER_SIZE:
        raise FrameError("payload overruns frame")

    chunk = blob[HEADER_SIZE:HEADER_SIZE + plen]
    if (zlib.crc32(chunk) & 0xFFFFFFFF) != ccrc:
        raise FrameError("chunk crc")
    return idx, total, tlen, fcrc, chunk


def deinterleave(raw: bytes, p: Profile) -> np.ndarray:
    need = p.rs_blocks * RS_BLOCK
    if len(raw) < need:
        raise FrameError("frame too short")
    return np.frombuffer(raw[:need], dtype=np.uint8).reshape(RS_BLOCK, p.rs_blocks).T


def repair(raw: bytes, p: Profile) -> bytes:
    if not HAVE_RS:
        raise FrameError("damaged (install reedsolo to enable repair)")
    grid = deinterleave(raw, p)
    rsc = RSCodec(p.nsym)  # type: ignore[possibly-unbound]
    out = []
    for i in range(p.rs_blocks):
        try:
            out.append(bytes(rsc.decode(grid[i].tobytes())[0]))
        except (ReedSolomonError, ValueError):  # type: ignore[possibly-unbound]
            out.append(grid[i][:p.rs_k].tobytes())
    return b"".join(out)


def decode(image: np.ndarray, p: Profile, repair_ok: bool = True):
    raw = frame_bytes(image, p)
    try:
        # systematic code: undamaged frames need no error correction at all
        return parse(deinterleave(raw, p)[:, :p.rs_k].tobytes())
    except FrameError:
        if not repair_ok:
            raise
        return parse(repair(raw, p))


# ---------------------------------------------------------------------------
# receiver
# ---------------------------------------------------------------------------


class Receiver:
    def __init__(self, p: Profile, workers: int) -> None:
        self.p = p
        self.jobs: Queue = Queue(maxsize=workers * 2)
        self.lock = threading.Lock()
        self.chunks: dict[int, bytes] = {}
        self.total = 0
        self.file_len = 0
        self.file_crc = 0
        self.origin: tuple[int, int] | None = None
        self.stop = threading.Event()
        self.duplicates = 0
        self.failures = 0
        self.decoded = 0
        self.miss_streak = 0
        self.first_at: float | None = None
        self.threads = [threading.Thread(target=self._work, daemon=True)
                        for _ in range(workers)]

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def _canonical(self, grab: np.ndarray) -> np.ndarray | None:
        """Try the locked position first; only hunt for markers when it fails."""
        p, spot = self.p, self.origin
        if spot is not None:
            x0, y0 = spot
            h, w = grab.shape[:2]
            if 0 <= x0 <= w - p.width and 0 <= y0 <= h - p.height:
                return grab[y0:y0 + p.height, x0:x0 + p.width]
        found = locate(grab, p)
        if found is None:
            return None
        canonical, origin = found
        if origin is not None:
            self.origin = origin
        return canonical

    def _work(self) -> None:
        while not self.stop.is_set():
            try:
                grab = self.jobs.get(timeout=0.2)
            except Empty:
                continue
            if grab is None:
                return

            canonical = self._canonical(grab)
            if canonical is None:
                self.failures += 1
                continue
            try:
                idx, total, tlen, fcrc, chunk = decode(canonical, self.p)
            except (FrameError, ValueError):
                if self.origin is not None:
                    self.origin = None  # window probably moved; re-hunt next time
                self.failures += 1
                continue

            with self.lock:
                if idx in self.chunks:
                    self.duplicates += 1
                    continue
                if self.first_at is None:
                    self.first_at = time.time()
                self.chunks[idx] = chunk
                self.total, self.file_len, self.file_crc = total, tlen, fcrc
                self.decoded += 1

    @property
    def complete(self) -> bool:
        return self.total > 0 and len(self.chunks) == self.total

    def missing(self) -> list[int]:
        with self.lock:
            return [i for i in range(self.total) if i not in self.chunks]

    def assemble(self) -> bytes:
        data = b"".join(self.chunks[i] for i in range(self.total))[:self.file_len]
        if (zlib.crc32(data) & 0xFFFFFFFF) != self.file_crc:
            raise FrameError("reassembled file failed its checksum")
        return data

    def shutdown(self) -> None:
        self.stop.set()
        for _ in self.threads:
            self.jobs.put(None)


def grab_rgb(sct, region) -> np.ndarray:
    return np.asarray(sct.grab(region))[:, :, 2::-1]


def pick_region(sct, monitor) -> dict | None:
    frame = grab_rgb(sct, monitor)
    preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    scale = min(1.0, 1500 / preview.shape[1])
    if scale < 1.0:
        preview = cv2.resize(preview, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    box = cv2.selectROI("drag the capture region, then press Enter", preview,
                        showCrosshair=False)
    cv2.destroyAllWindows()
    if not box[2] or not box[3]:
        return None
    x, y, w, h = (int(round(v / scale)) for v in box)
    pad = 32
    return {"left": monitor["left"] + max(0, x - pad),
            "top": monitor["top"] + max(0, y - pad),
            "width": w + 2 * pad, "height": h + 2 * pad}


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


def default_workers() -> int:
    import os

    return max(2, min(8, (os.cpu_count() or 4) - 1))


class CaptureSession:
    """Grab loop plus decoder pool. Drives both the CLI and the GUI.

    Pausing keeps everything decoded so far, so a stop/start cycle resumes
    rather than starting over.
    """

    def __init__(self, profile: Profile | None, region: dict, workers: int) -> None:
        self.profile = profile          # None means auto-detect from the screen
        self.region = dict(region)
        self.workers = workers
        self.rx: Receiver | None = None
        self.running = threading.Event()
        self.grabs = 0
        self.preview: np.ndarray | None = None
        self.last_grab: np.ndarray | None = None
        self.started_at = 0.0
        self.error: str | None = None
        self.saw_markers = False
        self.probe_fails = 0
        self._last_deep_probe = 0.0
        self._thread: threading.Thread | None = None

        if profile is not None:
            self._ensure_receiver(profile)

    def _ensure_receiver(self, profile: Profile) -> None:
        if self.rx is None:
            self.profile = profile
            self.rx = Receiver(profile, self.workers)
            self.rx.start()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.running.set()
        if not self.started_at:
            self.started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Stop grabbing but keep every frame already decoded."""
        self.running.clear()
        self._thread = None

    def close(self) -> None:
        self.running.clear()
        if self.rx is not None:
            self.rx.shutdown()
        self._thread = None

    def set_region(self, region: dict) -> None:
        self.region = dict(region)

    def _loop(self) -> None:
        # only wide enough to skip back-to-back grabs of a still screen. Anything
        # longer would also swallow the sender's next pass, so a frame that
        # failed to decode could never be retried.
        recent: deque = deque(maxlen=8)
        last_preview = 0.0
        try:
            with MSS_CLASS() as sct:  # mss handles are per-thread
                while self.running.is_set():
                    if self.rx is not None and self.rx.complete:
                        break
                    grab = grab_rgb(sct, self.region)
                    self.grabs += 1
                    self.last_grab = grab

                    now = time.time()
                    if now - last_preview > 0.12:
                        last_preview = now
                        step = max(1, grab.shape[1] // 420)
                        self.preview = np.ascontiguousarray(grab[::step, ::step])

                    if self.rx is None:
                        gray, corners, ids = detect_markers(grab)
                        if ids is None or len(corners) == 0:
                            self.saw_markers = False
                            continue
                        self.saw_markers = True
                        # repair costs seconds per profile, so ration it hard or
                        # the grab loop looks frozen
                        deep = now - self._last_deep_probe > 4.0
                        if deep:
                            self._last_deep_probe = now
                        found = identify(grab, gray, corners, ids, allow_repair=deep)
                        if found is None:
                            self.probe_fails += 1
                            continue
                        self._ensure_receiver(found)
                    rx = self.rx
                    assert rx is not None

                    # cheap duplicate test so a static screen costs almost nothing
                    signature = hashlib.blake2b(
                        np.ascontiguousarray(grab[::16, ::16]).tobytes(), digest_size=8
                    ).digest()
                    if signature in recent:
                        rx.duplicates += 1
                        time.sleep(0.002)
                        continue
                    recent.append(signature)

                    if not rx.jobs.full():
                        rx.jobs.put(grab)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.running.clear()

    @property
    def complete(self) -> bool:
        return self.rx is not None and self.rx.complete

    def stats(self) -> dict:
        rx = self.rx
        if rx is None:
            return {"have": 0, "total": 0, "grabs": self.grabs, "mbps": 0.0,
                    "eta": 0.0, "duplicates": 0, "failures": 0, "locked": False,
                    "profile": None, "saw_markers": self.saw_markers,
                    "probe_fails": self.probe_fails}
        have, total = len(rx.chunks), rx.total
        elapsed = max(time.time() - (rx.first_at or self.started_at), 1e-3)
        rate = have / elapsed
        payload = self.profile.payload_bytes if self.profile else 0
        return {
            "have": have, "total": total, "grabs": self.grabs,
            "mbps": rate * payload / 1048576,
            "eta": (total - have) / rate if rate > 0.05 else 0.0,
            "duplicates": rx.duplicates, "failures": rx.failures,
            "locked": rx.origin is not None,
            "profile": self.profile.name if self.profile else None,
            "saw_markers": self.saw_markers, "probe_fails": self.probe_fails,
        }


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

BG = "#F3F0E8"        # warm paper
CARD = "#FFFFFF"
EDGE = "#D6D1C2"
TEXT = "#171615"
MUTED = "#5F5C55"
ACCENT = "#C15F3C"    # deep coral
ACCENT_DARK = "#A44A2B"
INK = "#2E2B27"       # dark buttons
INK_HOVER = "#46423C"
OK = "#1E6B45"
WARN = "#8F5A0C"

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


def next_free(path: Path) -> Path:
    """Nearest unused filename, so one capture never overwrites the last."""
    if not path.exists():
        return path
    for n in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


class ReceiverGUI:
    def __init__(self, out: Path | None, profile_name: str) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        self.tk, self.filedialog = tk, filedialog
        self.session: CaptureSession | None = None
        self.out = out
        self._photo = None
        self.saved: list[Path] = []

        self.root = tk.Tk()
        self.root.title("pixpack receiver")
        self.root.geometry("620x780")
        self.root.configure(bg=BG)
        self.root.minsize(580, 720)
        apply_icon(self.root)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("pix.Horizontal.TProgressbar", troughcolor=EDGE,
                        background=ACCENT, bordercolor=EDGE, lightcolor=ACCENT,
                        darkcolor=ACCENT, thickness=8)
        style.configure("pix.TCombobox", fieldbackground=CARD, background=CARD)

        # header
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=26, pady=(20, 12))
        tk.Label(head, text="pixpack receiver", bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 19)).pack(anchor="w")
        tk.Label(head, text="capturing this computer's screens", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        # live view
        view_card = self._card()
        self.view = tk.Label(view_card, bg="#EFEDE4", bd=0)
        self.view.pack(padx=10, pady=(10, 6))
        self.lock = tk.Label(view_card, text="idle", bg=CARD, fg=MUTED,
                             font=("Segoe UI Semibold", 9))
        self.lock.pack(pady=(0, 10))

        # settings
        settings = self._card()
        row = tk.Frame(settings, bg=CARD)
        row.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(row, text="Profile", bg=CARD, fg=MUTED, width=8, anchor="w",
                 font=("Segoe UI", 9)).pack(side="left")
        self.profile = tk.StringVar(value=profile_name)
        ttk.Combobox(row, textvariable=self.profile, values=list(CHOICES),
                     state="readonly", width=12).pack(side="left")
        tk.Label(row, text="'auto' matches the sender by itself", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(10, 0))
        self.profile.trace_add("write", lambda *_: self.profile_changed())

        row2 = tk.Frame(settings, bg=CARD)
        row2.pack(fill="x", padx=14, pady=(0, 12))
        tk.Label(row2, text="Save to", bg=CARD, fg=MUTED, width=8, anchor="w",
                 font=("Segoe UI", 9)).pack(side="left")
        self.out_label = tk.Label(row2, bg=CARD, fg=TEXT, font=("Segoe UI", 9),
                                  anchor="w", text=str(out) if out else "(not chosen)")
        self.out_label.pack(side="left", fill="x", expand=True)
        self._button(row2, "Choose", self.choose_out, kind="quiet").pack(side="right")

        # progress
        progress = self._card()
        self.bar = ttk.Progressbar(progress, style="pix.Horizontal.TProgressbar")
        self.bar.pack(fill="x", padx=14, pady=(14, 8))
        self.state = tk.Label(progress, text="ready", bg=CARD, fg=TEXT,
                              font=("Segoe UI Semibold", 11))
        self.state.pack()
        self.detail = tk.Label(progress, text="", bg=CARD, fg=MUTED,
                               font=("Consolas", 9))
        self.detail.pack(pady=(2, 14))

        # actions
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=26, pady=(4, 10))
        self.start_button = self._button(bar, "Start capture", self.start, kind="primary")
        self.start_button.pack(side="left")
        self.stop_button = self._button(bar, "Stop", self.stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.stop_button.configure(state="disabled")
        self.finish_button = self._button(bar, "Finish & save", self.finish_now)
        self.finish_button.pack(side="left", padx=(8, 0))
        self._button(bar, "Diagnose", self.run_diagnose, kind="quiet").pack(side="right")
        self._button(bar, "Reset", self.reset, kind="quiet").pack(side="right", padx=(0, 8))

        # log
        log_card = self._card(expand=True)
        self.log = tk.Text(log_card, height=8, bg=CARD, fg=MUTED, bd=0, relief="flat",
                           font=("Consolas", 9), wrap="word", state="disabled",
                           highlightthickness=0)
        self.log.pack(fill="both", expand=True, padx=12, pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.say("Stop keeps what it has; Start carries on from there.")
        self.say("Finish & save writes the file and clears up for the next one.")
        self.root.after(120, self.refresh)

    # -- widgets ----------------------------------------------------------

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
            "quiet": ("#6E6A62", "#FFFFFF", "#87837A"),
        }[kind]
        return self.tk.Button(
            parent, text=text, command=command, bg=colours[0], fg=colours[1],
            activebackground=colours[2], activeforeground="#FFFFFF",
            disabledforeground="#C9C4BA",
            font=("Segoe UI Semibold", 9), bd=0, relief="flat",
            padx=14, pady=7, cursor="hand2", highlightthickness=0)

    def say(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # -- controls ---------------------------------------------------------

    def choose_out(self) -> None:
        chosen = self.filedialog.asksaveasfilename(
            title="Save the received file as",
            initialfile=self.out.name if self.out else "received.zip")
        if chosen:
            self.out = Path(chosen)
            self.out_label.configure(text=str(self.out))

    def profile_changed(self) -> None:
        if self.session is not None and not (self.session.rx and self.session.rx.chunks):
            self.session = None
        elif self.session is not None:
            self.say("profile changed - press Reset to apply it to a new capture")

    def region(self) -> dict:
        with MSS_CLASS() as sct:
            return dict(sct.monitors[0])  # every display at once

    def run_diagnose(self) -> None:
        grab = self.session.last_grab if self.session else None
        if grab is None:
            with MSS_CLASS() as sct:
                grab = grab_rgb(sct, self.region())
        self.say("--- diagnose ---")
        for line in diagnose(grab).splitlines():
            self.say("  " + line)

    def start(self) -> None:
        if self.session is not None and self.session.running.is_set():
            return
        if self.out is None:
            self.choose_out()
            if self.out is None:
                return

        name = self.profile.get()
        profile = None if name == "auto" else PROFILES[name]

        # nothing decoded means nothing worth keeping, and the profile it
        # auto-detected may have been wrong
        if self.session is not None and not (self.session.rx and self.session.rx.chunks):
            self.session = None

        if self.session is None:
            self.session = CaptureSession(profile, self.region(), default_workers())
            self.say(f"capture started ({'auto-detect' if profile is None else name})")
        else:
            kept = len(self.session.rx.chunks) if self.session.rx else 0
            self.say(f"resuming with {kept} frames already decoded")
        self.session.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.state.configure(text="waiting for frames")
        self.detail.configure(text="press Play in the sender")

    def stop(self) -> None:
        if self.session is not None:
            self.session.pause()
            s = self.session.stats()
            if s["total"]:
                self.say(f"paused with {s['have']:,}/{s['total']:,} frames kept")
                self.state.configure(text=f"paused - {s['have']:,}/{s['total']:,} kept")
            else:
                self.say(f"paused after {s['grabs']:,} grabs with nothing decoded")
                self.state.configure(text="paused - nothing decoded")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def finish_now(self) -> None:
        """Wrap up the current capture and get ready for the next file."""
        if self.session is None:
            self.say("nothing to finish")
            return
        self.session.pause()
        rx = self.session.rx
        if rx is not None and rx.complete:
            self.save(self.session)
        elif rx is not None and rx.total:
            missing = len(rx.missing())
            self.say(f"cannot save yet - {missing} of {rx.total} frames still missing")
            self.say("keep the sender looping and press Start again")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return
        else:
            self.say("nothing decoded, so nothing to save")
        self.clear_for_next()

    def clear_for_next(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        self.bar.configure(value=0)
        self.state.configure(text="ready for the next file")
        self.detail.configure(text="")
        self.lock.configure(text="idle", fg=MUTED)
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.out is not None:
            self.out = next_free(self.out)
            self.out_label.configure(text=str(self.out))

    def reset(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        self.bar.configure(value=0)
        self.state.configure(text="ready")
        self.detail.configure(text="")
        self.lock.configure(text="idle", fg=MUTED)
        self.say("reset - decoded frames discarded")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def quit(self) -> None:
        if self.session is not None:
            self.session.close()
        self.root.destroy()

    # -- live view --------------------------------------------------------

    def _show_preview(self, array: np.ndarray) -> None:
        h, w = array.shape[:2]
        header = f"P6 {w} {h} 255 ".encode()
        self._photo = self.tk.PhotoImage(data=header + array.tobytes(), format="PPM")
        self.view.configure(image=self._photo)

    def refresh(self) -> None:
        session = self.session
        if session is not None:
            if session.preview is not None:
                self._show_preview(session.preview)
            stats = session.stats()

            if session.error:
                self.state.configure(text=session.error)
            elif stats["total"]:
                self.bar.configure(maximum=stats["total"], value=stats["have"])
                pct = 100 * stats["have"] / stats["total"]
                self.state.configure(
                    text=f"{stats['have']:,} of {stats['total']:,} frames  ·  {pct:.0f}%")
                self.detail.configure(
                    text=f"{stats['mbps']:.2f} MiB/s   eta {stats['eta']:.0f}s   "
                         f"dup {stats['duplicates']:,}   miss {stats['failures']:,}")
            elif session.running.is_set():
                if stats["saw_markers"]:
                    self.detail.configure(
                        text=f"{stats['grabs']:,} grabs - markers seen, not decoding "
                             f"({stats['probe_fails']:,} tries)")
                else:
                    self.detail.configure(
                        text=f"{stats['grabs']:,} grabs - no frames on screen yet")

            if stats["locked"]:
                self.lock.configure(text=f"locked on {stats['profile']}", fg=OK)
            elif stats["saw_markers"] and session.running.is_set():
                self.lock.configure(text="frames found, decoding", fg=WARN)
            elif session.running.is_set():
                self.lock.configure(text="looking for frames", fg=MUTED)

            if session.complete:
                self.save(session)
                self.clear_for_next()
        self.root.after(120, self.refresh)

    def save(self, session: CaptureSession) -> None:
        assert session.rx is not None
        try:
            data = session.rx.assemble()
        except FrameError as exc:
            self.state.configure(text=f"failed: {exc}")
            self.say(f"assembly failed: {exc}")
            return

        assert self.out is not None
        target = next_free(self.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.saved.append(target)

        seconds = max(time.time() - (session.rx.first_at or session.started_at), 1e-3)
        self.say(f"SAVED {len(data):,} bytes to {target}")
        self.state.configure(text=f"saved {target.name}")
        self.detail.configure(
            text=f"checksum verified   {len(data) / 1048576 / seconds:.2f} MiB/s"
                 f"   {seconds:.1f}s   {session.rx.duplicates:,} duplicates skipped")

    def run(self) -> None:
        self.root.mainloop()


def run_cli(args) -> int:
    profile = None if args.profile == "auto" else PROFILES[args.profile]
    workers = args.workers or default_workers()

    with MSS_CLASS() as sct:
        if args.region:
            x, y, w, h = (int(v) for v in args.region.split(","))
            region = {"left": x, "top": y, "width": w, "height": h}
        elif args.monitor == "all":
            region = dict(sct.monitors[0])
        else:
            region = dict(sct.monitors[int(args.monitor)])
        if args.select:
            picked = pick_region(sct, region)
            if picked is None:
                return 2
            region = picked

    print(f"profile   {profile.name if profile else 'auto-detect'}")
    print(f"watching  {region['width']}x{region['height']} at "
          f"({region['left']}, {region['top']})  with {workers} decoder threads")
    print("waiting for frames - press Play in the VM\n")

    session = CaptureSession(profile, region, workers)
    session.start()
    try:
        while session.running.is_set() and not session.complete:
            if args.timeout and time.time() - session.started_at > args.timeout:
                print("\n\ntimed out", file=sys.stderr)
                break
            s = session.stats()
            if s["total"]:
                print(f"\r  {s['have']:,}/{s['total']:,} frames  "
                      f"{100 * s['have'] / s['total']:5.1f}%  {s['mbps']:5.2f} MiB/s"
                      f"  eta {s['eta']:5.0f}s  dup {s['duplicates']:,}"
                      f"  miss {s['failures']:,}   ", end="", flush=True)
            else:
                print(f"\r  scanning ... {s['grabs']:,} grabs, no frame yet   ",
                      end="", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n\nstopped", file=sys.stderr)
    finally:
        session.pause()

    rx = session.rx
    print()
    if rx is None or not rx.complete:
        if rx is None or rx.total == 0:
            print("\nno frames decoded.\n", file=sys.stderr)
            if session.last_grab is not None:
                print(diagnose(session.last_grab), file=sys.stderr)
        else:
            gone = rx.missing()
            print(f"\nincomplete: {len(gone)} of {rx.total} frames missing", file=sys.stderr)
            print(f"  {gone[:30]}{' ...' if len(gone) > 30 else ''}", file=sys.stderr)
            print("  let the sender keep looping and run this again", file=sys.stderr)
        session.close()
        return 1

    data = rx.assemble()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    seconds = max(time.time() - (rx.first_at or session.started_at), 1e-3)
    session.close()
    print(f"\nrestored {len(data):,} bytes -> {args.out}")
    print(f"checksum verified  |  {len(data) / 1048576 / seconds:.2f} MiB/s"
          f"  |  {seconds:.1f}s  |  {rx.duplicates:,} duplicates skipped")
    return 0


IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".webp"}


def load_image(path: Path) -> np.ndarray | None:
    """Read an image as RGB. np.fromfile copes with non-ASCII paths on Windows."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return None if img is None else img[:, :, ::-1]


def run_from_frames(args) -> int:
    """Rebuild the file from a folder of frame images, no screen involved."""
    folder: Path = args.from_frames
    if not folder.is_dir():
        print(f"error: {folder} is not a folder", file=sys.stderr)
        return 2

    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"error: no images in {folder}", file=sys.stderr)
        return 2

    first = load_image(files[0])
    if first is None:
        print(f"error: could not read {files[0].name}", file=sys.stderr)
        return 2

    if args.profile == "auto":
        profile = detect_profile(first)
        if profile is None:
            print("error: could not work out the profile from these images.\n",
                  file=sys.stderr)
            print(diagnose(first), file=sys.stderr)
            return 2
    else:
        profile = PROFILES[args.profile]

    workers = args.workers or default_workers()
    print(f"folder    {folder}")
    print(f"images    {len(files):,}")
    print(f"profile   {profile.name}  ({profile.payload_bytes:,} B/frame)\n")

    rx = Receiver(profile, workers)
    rx.start()
    unreadable = []
    try:
        for n, path in enumerate(files, 1):
            image = load_image(path)
            if image is None:
                unreadable.append(path.name)
                continue
            rx.jobs.put(image)  # blocks when full, which paces the reader
            if n % 10 == 0 or n == len(files):
                print(f"\r  read {n:,}/{len(files):,}   decoded {len(rx.chunks):,}",
                      end="", flush=True)

        idle = 0.0
        while not rx.complete and idle < 5.0:
            before = len(rx.chunks)
            time.sleep(0.2)
            idle = 0.0 if (len(rx.chunks) != before or not rx.jobs.empty()) else idle + 0.2
            print(f"\r  read {len(files):,}/{len(files):,}   "
                  f"decoded {len(rx.chunks):,}      ", end="", flush=True)
    finally:
        rx.shutdown()

    print()
    if unreadable:
        print(f"\n{len(unreadable)} file(s) could not be opened: {unreadable[:5]}",
              file=sys.stderr)

    if not rx.complete:
        if rx.total == 0:
            print("\nnothing decoded from these images.\n", file=sys.stderr)
            print(diagnose(first), file=sys.stderr)
        else:
            gone = rx.missing()
            print(f"\nincomplete: {len(gone)} of {rx.total} frames missing", file=sys.stderr)
            print(f"  {gone[:30]}{' ...' if len(gone) > 30 else ''}", file=sys.stderr)
            print("  some frame images are missing or damaged", file=sys.stderr)
        return 1

    data = rx.assemble()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"\nrestored {len(data):,} bytes -> {args.out}")
    print(f"checksum verified  |  {rx.total:,} frames  |  "
          f"{rx.failures:,} unreadable, {rx.duplicates:,} duplicates")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, help="file to write")
    ap.add_argument("-p", "--profile", default=DEFAULT_PROFILE, choices=CHOICES,
                    help="default 'auto' works it out from the screen")
    ap.add_argument("--workers", type=int, default=0, help="decoder threads (0 = auto)")
    ap.add_argument("--monitor", default="all", help="'all' or a monitor number")
    ap.add_argument("--region", help="x,y,w,h")
    ap.add_argument("--select", action="store_true", help="drag the region with the mouse")
    ap.add_argument("--timeout", type=float, default=0, help="give up after N seconds")
    ap.add_argument("--cli", action="store_true", help="no window, terminal only")
    ap.add_argument("--from-frames", type=Path, metavar="FOLDER",
                    help="rebuild from a folder of frame images instead of the screen")
    args = ap.parse_args()

    set_dpi_aware()
    if not HAVE_RS:
        print("note: reedsolo not installed - damaged frames are retried, not repaired\n")

    if args.from_frames is not None:
        if args.out is None:
            print("error: --from-frames needs -o/--out", file=sys.stderr)
            return 2
        return run_from_frames(args)

    if args.cli:
        if args.out is None:
            print("error: --cli needs -o/--out", file=sys.stderr)
            return 2
        return run_cli(args)

    ReceiverGUI(args.out, args.profile).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
