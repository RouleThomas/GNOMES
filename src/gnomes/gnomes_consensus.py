#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
import time
from datetime import datetime
import threading
import math
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np

# Plotting (always ON)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REQUIRED_COLS = ["sample_id", "bam", "condition", "target"]
OPTIONAL_CONTROL_COL = "bam_control"


# -----------------------------
# GNOME splash (ALWAYS prints once)
# -----------------------------
def print_gnome_splash():
    splash = r"""
		     .*#*:.#.           ______   __    __   ______   __       __  ________   ______  
                   **-    -*           /      \ /  \  /  | /      \ /  \     /  |/        | /      \
                :*+       #           /$$$$$$  |$$  \ $$ |/$$$$$$  |$$  \   /$$ |$$$$$$$$/ /$$$$$$  |
               **        =#           $$ | _$$/ $$$  \$$ |$$ |  $$ |$$$  \ /$$$ |$$ |__    $$ \__$$/ 
             =#          *:           $$ |/    |$$$$  $$ |$$ |  $$ |$$$$  /$$$$ |$$    |   $$      \ 
            **           *            $$ |$$$$ |$$ $$ $$ |$$ |  $$ |$$ $$ $$/$$ |$$$$$/     $$$$$$  |
           #=            *            $$ \__$$ |$$ |$$$$ |$$ \__$$ |$$ |$$$/ $$ |$$ |_____ /  \__$$ |
          #-             +*           $$    $$/ $$ | $$$ |$$    $$/ $$ | $/  $$ |$$       |$$    $$/ 
         #=               #            $$$$$$/  $$/   $$/  $$$$$$/  $$/      $$/ $$$$$$$$/  $$$$$$/  
        *=                -*
       **                  +*          GNOMES (Genome-wide NOrmalization of
      .#                    *-                 Mapped Epigenomic Signals)
      *=                    :*
      #                      *
     :**+                 .+*#          Thank you for using GNOMES!
     #       :+***##*-.      *+
     -###*- **       +#:.+*##*
     +*     #         *+    -*
     #      =#.      **      *-
    .*        -*#**#+        *+
   +#*+                     .#**
  +#  **                   =#  **
  *   =*                   :*   *=
 #+    #                   *+    #
 #  .  =*                 =*  =  *-
 #*#*    *+              **   :##*.
 *  +      *###      *##+      = =-
 *= *          #*-.=#+        :* #
  .#*                         *#=
    **:*=::=+         :+:.-*==*
     *        *     -:       *+
     +#*******+-=====*##******

"""
    print(splash, flush=True)


# -----------------------------
# GNOME walker (multi-line, only when interactive; never written to log)
# -----------------------------
class GnomeWalker:
    """
    Animated gnome shown ONLY when stdout is a TTY.
    Uses alternate screen buffer so it doesn't pollute scrollback.
    """
    def __init__(self, enabled=True, fps=10, track_width=30):
        self.enabled = bool(enabled) and sys.stdout.isatty()
        self.fps = max(1, int(fps))
        self.track_width = max(0, int(track_width))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._label = "Working"
        self._pos = 0
        self._frame_i = 0

        base = [
            "                                   ",
            "                     .*#*:.#.      ",
            "                   **-    -*       ",
            "                :*+       #        ",
            "               **        =#        ",
            "             =#          *:        ",
            "            **           *         ",
            "           #=            *         ",
            "          #-             +*        ",
            "         #=               #        ",
            "        *=                -*       ",
            "       **                  +*      ",
            "      .#                    *-     ",
            "      *=                    :*     ",
            "      #                      *     ",
            "     :**+                 .+*#     ",
            "     #       :+***##*-.      *+    ",
            "     -###*- **       +#:.+*##*     ",
            "     +*     #         *+    -*     ",
            "     #      =#.      **      *-    ",
            "    .*        -*#**#+        *+    ",
            "   +#*+                     .#**   ",
            "  +#  **                   =#  **  ",
            "  *   =*                   :*   *= ",
            " #+    #                   *+    # ",
            " #  .  =*                 =*  =  *-",
            " #*#*    *+              **   :##*.",
            " *  +      *###      *##+      = =-",
            " *= *          #*-.=#+        :* # ",
            "  .#*                         *#=  ",
            "    **:*=::=+         :+:.-*==*    ",
            "     *        *     -:       *+    ",
            "     +#*******+-=====*##******     ",
            "                                   ",
        ]

        f0 = base
        f1 = base.copy()
        bob_lines = [28, 29, 30, 31, 32]
        for i in bob_lines:
            f1[i] = " " + f1[i][:-1]
        f1[-3] = "     *        *     :-       *+    "
        f1[-2] = "     +#*******+-=====-*##******     "

        self.frames = [f0, f1]
        self.block_height = 1 + len(self.frames[0])

    def _ansi(self, s: str):
        sys.stdout.write(s)
        sys.stdout.flush()

    def _enter_alt_screen(self):
        self._ansi("\x1b[?1049h\x1b[H")
        self._ansi("\x1b[?25l")

    def _exit_alt_screen(self):
        self._ansi("\x1b[?25h")
        self._ansi("\x1b[?1049l")

    def _move_up(self, n: int):
        if n > 0:
            self._ansi(f"\x1b[{n}A")

    def _clear_line(self):
        self._ansi("\x1b[2K\r")

    def start(self, label="Working"):
        if not self.enabled or self._thread is not None:
            return
        self._label = label
        self._stop.clear()

        self._enter_alt_screen()
        sys.stdout.write("\n" * self.block_height)
        sys.stdout.flush()
        self._move_up(self.block_height)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled or self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

        with self._lock:
            self._move_up(self.block_height)
            for _ in range(self.block_height):
                self._clear_line()
                sys.stdout.write("\n")
            sys.stdout.flush()

        self._exit_alt_screen()

    def set_label(self, label: str):
        if not self.enabled:
            return
        with self._lock:
            self._label = label

    def _run(self):
        dt = 1.0 / float(self.fps)
        while not self._stop.is_set():
            with self._lock:
                frame = self.frames[self._frame_i % len(self.frames)]
                self._frame_i += 1

                x = self._pos % (self.track_width + 1)
                self._pos += 1
                pad = " " * x

                self._move_up(self.block_height)
                self._clear_line()
                sys.stdout.write(f"{self._label}\n")

                for ln in frame:
                    self._clear_line()
                    sys.stdout.write(pad + ln + "\n")

                sys.stdout.flush()
                self._move_up(self.block_height)

            time.sleep(dt)


class DelayedWalkController:
    """
    Starts the walker only if a step lasts longer than delay_s.
    """
    def __init__(self, walker: GnomeWalker, delay_s: float = 10.0):
        self.walker = walker
        self.delay_s = float(delay_s)
        self._timer = None
        self._walking = False
        self._label = "Working"
        self._lock = threading.Lock()

    def begin_step(self, label: str):
        with self._lock:
            self._label = label
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            if self._walking:
                self.walker.set_label(label)
                return

            self._timer = threading.Timer(self.delay_s, self._start)
            self._timer.daemon = True
            self._timer.start()

    def _start(self):
        with self._lock:
            self._timer = None
            self._walking = True
            self.walker.start(self._label)

    def end_step(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._walking:
                self.walker.stop()
                self._walking = False

    def set_label(self, label: str):
        with self._lock:
            self._label = label
            if self._walking:
                self.walker.set_label(label)

    def announce(self, line: str):
        print(line, flush=True)


# -----------------------------
# Pretty logging + timing
# -----------------------------
def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, ctrl=None):
    line = f"[{now()}] {msg}"
    if ctrl is not None:
        ctrl.announce(line)
    else:
        print(line, flush=True)


def log_done(msg, tstart, ctrl=None):
    log(f"{msg} (done in {time.time() - tstart:.1f}s)", ctrl=ctrl)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_exe(name):
    if shutil.which(name) is None:
        die(f"Required executable not found in PATH: {name}")


def run_cmd(cmd, log_fh=None):
    cmd_str = " ".join(cmd)
    if log_fh:
        log_fh.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] CMD: {cmd_str}\n")
        log_fh.flush()

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_fh:
        log_fh.write(res.stdout)
        log_fh.flush()

    if res.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd_str}\n{res.stdout}")
    return res.stdout


# -----------------------------
# Helpers
# -----------------------------
def _safe_slug(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def read_meta(meta_path):
    if not os.path.exists(meta_path):
        die(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path, sep="\t", dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        die(f"Metadata missing required columns: {missing} (required: {REQUIRED_COLS})")

    if OPTIONAL_CONTROL_COL not in df.columns:
        df[OPTIONAL_CONTROL_COL] = ""

    if (df["sample_id"].astype(str).str.strip() == "").any():
        die("Metadata contains empty sample_id.")
    if df["sample_id"].duplicated().any():
        dups = df[df["sample_id"].duplicated()]["sample_id"].tolist()
        die(f"Metadata contains duplicate sample_id(s): {dups}")

    if (df["bam"].astype(str).str.strip() == "").any():
        die("Metadata contains empty bam path(s).")
    missing_bams = [b for b in df["bam"].tolist() if b and not os.path.exists(b)]
    if missing_bams:
        die("Some BAMs listed in meta do not exist:\n" + "\n".join(sorted(set(missing_bams))))

    ctrl_vals = df[OPTIONAL_CONTROL_COL].astype(str).str.strip().tolist()
    missing_ctrl = [b for b in ctrl_vals if b and not os.path.exists(b)]
    if missing_ctrl:
        die("Some bam_control BAMs listed in meta do not exist:\n" + "\n".join(sorted(set(missing_ctrl))))

    return df


def qvalue_to_macs2_qscore(qval: float) -> float:
    if qval <= 0 or qval >= 1:
        die(f"--macs2-qvalue values must be in (0,1). Got: {qval}")
    return -math.log10(qval)


def filter_macs2_peakfile_by_qscore(in_peak: str, out_bed: str, qscore_thr: float) -> int:
    """
    Reads MACS2 narrowPeak/broadPeak (>=9 cols) and keeps rows with col9 >= qscore_thr.
    Writes BED3. Returns kept count.
    """
    kept = 0
    with open(in_peak) as fin, open(out_bed, "w") as fout:
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            try:
                score = float(parts[8])
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            if score >= qscore_thr and end > start:
                fout.write(f"{parts[0]}\t{start}\t{end}\n")
                kept += 1
    return kept


def sort_bed(in_bed: str, out_bed: str) -> int:
    rows = []
    with open(in_bed) as fin:
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                s = int(p[1])
                e = int(p[2])
            except ValueError:
                continue
            rows.append((p[0], s, e))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    with open(out_bed, "w") as fout:
        for c, s, e in rows:
            fout.write(f"{c}\t{s}\t{e}\n")
    return len(rows)


def bed_widths(bed_path: str) -> np.ndarray:
    widths = []
    with open(bed_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                s = int(p[1])
                e = int(p[2])
            except ValueError:
                continue
            if e > s:
                widths.append(e - s)
    return np.asarray(widths, dtype=np.int64)


def width_stats(widths: np.ndarray) -> Dict[str, float]:
    if widths.size == 0:
        return {
            "n_peaks": 0,
            "median_width": np.nan,
            "mean_width": np.nan,
            "sd_width": np.nan,
            "min_width": np.nan,
            "max_width": np.nan,
            "p25_width": np.nan,
            "p75_width": np.nan,
        }
    mean = float(np.mean(widths))
    sd = float(np.std(widths, ddof=1)) if widths.size >= 2 else 0.0
    return {
        "n_peaks": int(widths.size),
        "median_width": float(np.median(widths)),
        "mean_width": mean,
        "sd_width": sd,
        "min_width": int(np.min(widths)),
        "max_width": int(np.max(widths)),
        "p25_width": float(np.percentile(widths, 25)),
        "p75_width": float(np.percentile(widths, 75)),
    }


def plot_width_histogram(
    widths: np.ndarray,
    *,
    title: str,
    bin_bp: int,
    max_width: int,
):
    """
    One page plot. x = width (bp) binned every bin_bp.
    Everything > max_width goes to overflow bin.
    Adds vertical lines: median, p25, p75 (if any peaks).
    """
    fig = plt.figure(figsize=(8.5, 5.0))
    ax = plt.gca()

    if widths.size == 0:
        ax.text(0.5, 0.5, "No peaks", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("Peak width (bp)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig

    widths_cap = widths.copy()
    widths_cap[widths_cap > max_width] = max_width + 1  # overflow

    # bins: 0..max_width step bin_bp, plus overflow edge
    edges = list(range(0, max_width + bin_bp, bin_bp)) + [max_width + 2]
    weights = np.ones_like(widths_cap) / len(widths_cap)
    ax.hist(widths_cap, bins=edges, weights=weights)

    # vertical lines
    med = float(np.median(widths))
    p25 = float(np.percentile(widths, 25))
    p75 = float(np.percentile(widths, 75))
    ax.axvline(med, linestyle="--", linewidth=1.5)
    ax.axvline(p25, linestyle=":", linewidth=1.5)
    ax.axvline(p75, linestyle=":", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel(f"Peak width (bp) — bin={bin_bp} bp; overflow>{max_width} bp")
    ax.set_ylabel("Fraction of peaks")

    # make x tick label for overflow cleaner
    xt = ax.get_xticks()
    # keep default ticks; just limit view to include overflow
    ax.set_xlim(0, max_width + bin_bp)

    plt.tight_layout()
    return fig


def print_launch_summary(args, meta: Optional[pd.DataFrame] = None):
    lines = []
    lines.append("=" * 72)
    lines.append("GNOMES consensus — configuration")
    lines.append("-" * 72)
    lines.append("Inputs:")
    lines.append(f"  - meta: {args.meta}")
    lines.append(f"  - outdir: {args.outdir}")
    lines.append("")
    lines.append("MACS2:")
    lines.append(f"  - mode: {args.macs2_mode}")
    lines.append(f"  - format: {args.macs2_format}")
    lines.append(f"  - gsize: {args.macs2_gsize}")
    lines.append(f"  - qvalue list: {args.macs2_qvalue}")
    lines.append(f"  - merge -d list: {args.macs2_merge}")
    lines.append("")
    lines.append("Plots:")
    lines.append(f"  - width bin (bp): {args.plot_bin}")
    lines.append(f"  - plot max width (bp): {args.plot_max}")
    lines.append("  - output: consensus_width_distributions.pdf (multi-page; ALWAYS generated)")
    if meta is not None:
        lines.append("")
        lines.append("Meta summary:")
        lines.append(f"  - samples: {meta.shape[0]}")
        targets = list(pd.unique(meta["target"]))
        conds = list(pd.unique(meta["condition"]))
        lines.append(f"  - targets: {targets}")
        lines.append(f"  - conditions: {conds}")
        has_ctrl = (meta[OPTIONAL_CONTROL_COL].astype(str).str.strip() != "").any()
        lines.append(f"  - bam_control: {'present (MACS2 -c pooled per condition)' if has_ctrl else 'not present'}")
    lines.append("=" * 72)
    print("\n".join(lines), flush=True)


def print_final_gnome_summary(summary_lines: List[str]):
    gnome = r"""
                                                             
                                     **#*                    
                                 **#*   ##                   
                               ##      ##                    
                             **        *                     
                           *#         **                     
                          ##          *                      
                        *#            *                      
            ******     *#             **      ******         
           *      *   *#               #     **     *        
           #*    **  ##                ##    *#     *        
             #*#  *# #                  *=  ** *- *#         
             *#    #*# -*-              #* #*   .*           
              **    *     +#******#    *###     *#           
               **   ***#-*+       *##*** *     ##            
                *# **     *       *#    #*   **              
                 *##       ********      *  #*               
                   *                     *##                 
                   ###                 #**                   
                  *##*                 *##                   
                 #* #*                 **#*                  
                 #   #                ##  #                  
                **    **             *#   #*                 
                #*      ***      ####-    *#                 
             ##  ***       ** -#*-        *#                 
            ###*    *                     ##                 
            .####*   #                  #+#                  
             *#####  *#             #*     **                
               ##### #*  *-#********    *#*##                
                 *###            =*  ########                
                                  ########*                  
                   *   #           #**#=                     
                    *   #       *  :                         
                      *        *  #                          
                              *                              
"""
    g_lines = gnome.splitlines()
    while g_lines and g_lines[0].strip() == "":
        g_lines = g_lines[1:]
    while g_lines and g_lines[-1].strip() == "":
        g_lines = g_lines[:-1]

    s_lines = list(summary_lines)
    pad = 4
    width_left = max(len(x) for x in g_lines) if g_lines else 0
    n = max(len(g_lines), len(s_lines))
    out = []
    for i in range(n):
        left = g_lines[i] if i < len(g_lines) else ""
        right = s_lines[i] if i < len(s_lines) else ""
        out.append(left.ljust(width_left) + (" " * pad) + right)
    print("\n".join(out), flush=True)


def parse_float_list(csv_or_multi: str) -> List[float]:
    # accepts: "0.1,0.05,0.01" or repeated flags already joined by argparse (we'll handle there)
    parts = []
    for chunk in str(csv_or_multi).split(","):
        chunk = chunk.strip()
        if chunk == "":
            continue
        parts.append(float(chunk))
    return parts


def parse_int_list(csv_or_multi: str) -> List[int]:
    parts = []
    for chunk in str(csv_or_multi).split(","):
        chunk = chunk.strip()
        if chunk == "":
            continue
        parts.append(int(chunk))
    return parts


def fmt_qval(q: float) -> str:
    # nice, filesystem-safe
    s = f"{q:.10g}"
    s = s.replace(".", "p")
    return s


# -----------------------------
# Main
# -----------------------------
def main():
    t0 = time.time()
    print_gnome_splash()

    ap = argparse.ArgumentParser(
        prog="GNOMES consensus",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "GNOMES consensus\n"
            "---------------\n"
            "Call MACS2 peaks pooled per condition (per target), then build MANY consensus peak BEDs\n"
            "across user-provided qvalue and merge-distance grids.\n\n"
            "Outputs:\n"
            "  - consensus_beds/*.bed (one per candidate)\n"
            "  - consensus_summary.tsv\n"
            "  - consensus_width_distributions.pdf (multi-page; one candidate per page)\n"
        ),
    )

    # -----------------------------
    # Required inputs
    # -----------------------------
    req = ap.add_argument_group("Required inputs")

    req.add_argument("--meta", required=True,
                     help="samples.tsv with columns: sample_id bam condition target [bam_control optional]")

    req.add_argument("--outdir", required=True,
                     help="Output directory")

    # -----------------------------
    # MACS2 peak calling
    # -----------------------------
    macs = ap.add_argument_group("MACS2 peak calling")

    macs.add_argument("--macs2-format", default="AUTO",
                    help="MACS2 -f / --format Format of tag file (default AUTO, use BAMPE for Paired-End data; pass exactly as MACS2 expects)")

    macs.add_argument("--macs2-gsize", default="hs",
                    help="MACS2 -g / --gsize Mappable genome size (default hs; can be mm or numeric like 2.7e9)")

    macs.add_argument("--macs2-mode", default="broad", choices=["narrow", "broad"],
                    help="MACS2 peak type (default broad)")

    # -----------------------------
    # Consensus grid (qvalue x merge distance)
    # -----------------------------
    grid = ap.add_argument_group("Consensus grid")

    grid.add_argument("--macs2-qvalue", default="0.1,0.05,0.01,0.001,0.0001",
                      help=("Comma-separated Q-value cutoffs.\n"
                            "Internally converted to MACS2 qscore threshold (-log10(q)).\n"
                            "Default: 0.1,0.05,0.01,0.001,0.0001"))

    grid.add_argument("--macs2-merge", default="0,50,100,250,500",
                      help=("Comma-separated bedtools merge -d distances (bp).\n"
                            "Default: 0,50,100,250,500"))

    # -----------------------------
    # Peak width plots
    # -----------------------------
    plot = ap.add_argument_group("Peak width plots")

    plot.add_argument("--plot-bin", type=int, default=200,
                      help="Histogram bin width (bp) for peak widths (default 200)")

    plot.add_argument("--plot-max", type=int, default=5000,
                      help="Max width (bp) for x-axis; >plot-max goes to overflow bin (default 5000)")

    # -----------------------------
    # Miscellaneous
    # -----------------------------
    misc = ap.add_argument_group("Miscellaneous")

    misc.add_argument("--no-walk", action="store_true",
                      help="Disable GNOMES walking animation")

    args = ap.parse_args()

    # Parse lists
    try:
        qvals = parse_float_list(args.macs2_qvalue)
        merges = parse_int_list(args.macs2_merge)
    except Exception as e:
        die(f"Could not parse --macs2-qvalue/--macs2-merge: {e}")

    if not qvals:
        die("--macs2-qvalue parsed empty list.")
    if not merges:
        die("--macs2-merge parsed empty list.")

    # Required executables
    ensure_exe("macs2")
    ensure_exe("bedtools")

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "GNOMES_consensus.log")

    # Output folders
    peaks_dir = os.path.join(args.outdir, "01_macs2_peaks")
    beds_dir = os.path.join(args.outdir, "02_consensus_beds")
    tmp_dir = os.path.join(args.outdir, "tmp")
    os.makedirs(peaks_dir, exist_ok=True)
    os.makedirs(beds_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    pdf_path = os.path.join(args.outdir, "consensus_width_distributions.pdf")
    summary_tsv = os.path.join(args.outdir, "consensus_summary.tsv")

    walker = GnomeWalker(enabled=(not args.no_walk), fps=10, track_width=30)
    ctrl = DelayedWalkController(walker, delay_s=10.0)

    # bookkeeping summary
    meta = None
    has_ctrl_any = False
    n_targets = 0
    n_conditions = 0
    n_candidates_total = 0
    beds_written = 0

    try:
        with open(log_path, "w") as log_fh:
            log_fh.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
            log_fh.write(f"Args: {vars(args)}\n")
            log_fh.write(f"Parsed qvals: {qvals}\n")
            log_fh.write(f"Parsed merges: {merges}\n")
            log_fh.flush()

            # -------------------------
            # Step 1: Load meta
            # -------------------------
            ctrl.begin_step("Step 1/4 — Reading metadata")
            t = time.time()
            log("Reading metadata", ctrl=ctrl)

            meta = read_meta(args.meta)
            has_ctrl_any = (meta[OPTIONAL_CONTROL_COL].astype(str).str.strip() != "").any()
            targets = sorted(meta["target"].unique().tolist())
            conditions = sorted(meta["condition"].unique().tolist())
            n_targets = len(targets)
            n_conditions = len(conditions)

            print_launch_summary(args, meta=meta)

            log_done("Reading metadata", t, ctrl=ctrl)
            ctrl.end_step()

            # -------------------------
            # Step 2: MACS2 per (target, condition) pooled
            # -------------------------
            ctrl.begin_step("Step 2/4 — Calling MACS2 peaks (pooled per condition)")
            t = time.time()
            log("Calling MACS2 peaks pooled per condition (separately per target)", ctrl=ctrl)

            peakfile_by_target_cond: Dict[Tuple[str, str], str] = {}
            used_control_by_target_cond: Dict[Tuple[str, str], bool] = {}

            for ti, tgt in enumerate(targets, start=1):
                meta_t = meta.loc[meta["target"] == tgt].copy()
                conds_t = sorted(meta_t["condition"].unique().tolist())

                for ci, cond in enumerate(conds_t, start=1):
                    ctrl.set_label(f"Step 2/4 — MACS2: target {ti}/{n_targets} {tgt} | condition {ci}/{len(conds_t)} {cond}")

                    sub = meta_t.loc[meta_t["condition"] == cond].copy()
                    bams = sub["bam"].astype(str).str.strip().tolist()
                    bams = [b for b in bams if b]
                    if len(bams) == 0:
                        continue

                    # pooled controls if bam_control present AND complete for this group
                    controls_raw = sub[OPTIONAL_CONTROL_COL].astype(str).str.strip().tolist()
                    any_ctrl = any(c != "" for c in controls_raw)
                    controls = []
                    if any_ctrl:
                        missing_here = [sub.iloc[i]["sample_id"] for i, c in enumerate(controls_raw) if c == ""]
                        if missing_here:
                            die(
                                f"bam_control is partially missing for target={tgt}, condition={cond}. "
                                f"Either provide bam_control for ALL samples in that group or leave all blank.\n"
                                f"Missing control for sample_id: {missing_here}"
                            )
                        controls = [c for c in controls_raw if c]
                        used_control_by_target_cond[(tgt, cond)] = True
                    else:
                        used_control_by_target_cond[(tgt, cond)] = False

                    tgt_slug = _safe_slug(tgt)
                    cond_slug = _safe_slug(cond)

                    outdir_tc = os.path.join(peaks_dir, f"macs2__target_{tgt_slug}__cond_{cond_slug}")
                    os.makedirs(outdir_tc, exist_ok=True)

                    name = f"target_{tgt_slug}__cond_{cond_slug}__{args.macs2_mode}"

                    cmd = ["macs2", "callpeak", "-t", *bams]
                    if controls:
                        cmd += ["-c", *controls]
                    cmd += [
                        "-f", args.macs2_format,
                        "--keep-dup", "auto",
                        "--nomodel",
                        "-g", str(args.macs2_gsize),
                        "--outdir", outdir_tc,
                        "-n", name,
                    ]
                    if args.macs2_mode == "broad":
                        cmd.append("--broad")

                    run_cmd(cmd, log_fh=log_fh)

                    peakfile = os.path.join(
                        outdir_tc,
                        f"{name}_peaks.broadPeak" if args.macs2_mode == "broad" else f"{name}_peaks.narrowPeak"
                    )
                    if not os.path.exists(peakfile):
                        die(f"MACS2 did not produce expected peak file: {peakfile}")

                    peakfile_by_target_cond[(tgt, cond)] = peakfile

            if not peakfile_by_target_cond:
                die("No MACS2 peak files produced. Check your meta (targets/conditions/BAM paths).")

            log_done("MACS2 peak calling finished", t, ctrl=ctrl)
            ctrl.end_step()

            # -------------------------
            # Step 3: Build consensus BEDs across (qvalue x merge)
            # -------------------------
            ctrl.begin_step("Step 3/4 — Building consensus peak BEDs (grid search)")
            t = time.time()
            log("Building consensus peak BEDs across qvalue x merge", ctrl=ctrl)

            summary_rows = []

            # Multi-page PDF
            with PdfPages(pdf_path) as pdf:
                for ti, tgt in enumerate(targets, start=1):
                    tgt_slug = _safe_slug(tgt)
                    conds_t = sorted(meta.loc[meta["target"] == tgt, "condition"].unique().tolist())

                    # only keep those where macs2 ran
                    conds_t = [c for c in conds_t if (tgt, c) in peakfile_by_target_cond]
                    if not conds_t:
                        continue

                    # Pre-filter per qval, per condition into tmp beds (cheap, no macs2 rerun)
                    for qv in qvals:
                        qscore_thr = qvalue_to_macs2_qscore(qv)
                        qv_slug = fmt_qval(qv)

                        # make a per-(target,qval) subfolder in tmp
                        tmp_tq = os.path.join(tmp_dir, f"target_{tgt_slug}__q_{qv_slug}")
                        os.makedirs(tmp_tq, exist_ok=True)

                        per_cond_beds = []
                        for cond in conds_t:
                            cond_slug = _safe_slug(cond)
                            peakfile = peakfile_by_target_cond[(tgt, cond)]
                            out_bed = os.path.join(tmp_tq, f"peaks__cond_{cond_slug}__q_{qv_slug}.bed")

                            n_kept = filter_macs2_peakfile_by_qscore(peakfile, out_bed, qscore_thr)
                            if n_kept == 0:
                                log(f"WARNING: target={tgt} cond={cond} kept 0 peaks at q={qv} (qscore>={qscore_thr:.6f})", ctrl=ctrl)
                            per_cond_beds.append(out_bed)

                        # concat + sort once per qval (then merge multiple distances)
                        concat_path = os.path.join(tmp_tq, f"all_conds__q_{qv_slug}.concat.bed")
                        with open(concat_path, "w") as fout:
                            for bedp in per_cond_beds:
                                if os.path.exists(bedp) and os.path.getsize(bedp) > 0:
                                    with open(bedp) as fin:
                                        shutil.copyfileobj(fin, fout)

                        sorted_path = os.path.join(tmp_tq, f"all_conds__q_{qv_slug}.sorted.bed")
                        n_rows = sort_bed(concat_path, sorted_path)
                        if n_rows == 0:
                            # still write empty candidates for all merges (so summary is complete)
                            for md in merges:
                                n_candidates_total += 1
                                cand_name = (
                                    f"consensus__target_{tgt_slug}"
                                    f"__mode_{args.macs2_mode}"
                                    f"__q_{qv_slug}"
                                    f"__merge_{md}"
                                    f"__format_{_safe_slug(args.macs2_format)}"
                                    f"__gsize_{_safe_slug(args.macs2_gsize)}"
                                )
                                out_bed = os.path.join(beds_dir, cand_name + ".bed")
                                open(out_bed, "w").close()
                                beds_written += 1

                                widths = np.asarray([], dtype=np.int64)
                                st = width_stats(widths)

                                title = f"{cand_name}\n(n_peaks=0)"
                                fig = plot_width_histogram(
                                    widths,
                                    title=title,
                                    bin_bp=int(args.plot_bin),
                                    max_width=int(args.plot_max),
                                )
                                pdf.savefig(fig)
                                plt.close(fig)

                                row = {
                                    "target": tgt,
                                    "macs2_mode": args.macs2_mode,
                                    "macs2_format": args.macs2_format,
                                    "macs2_gsize": args.macs2_gsize,
                                    "qvalue": qv,
                                    "qscore_thr": qscore_thr,
                                    "merge_d": md,
                                    "bed": os.path.basename(out_bed),
                                    **st,
                                }
                                summary_rows.append(row)
                            continue

                        # Now build consensus for each merge distance
                        for md in merges:
                            ctrl.set_label(f"Step 3/4 — target {ti}/{n_targets} {tgt} | q={qv} | merge={md}")
                            n_candidates_total += 1

                            cand_name = (
                                f"consensus__target_{tgt_slug}"
                                f"__mode_{args.macs2_mode}"
                                f"__q_{qv_slug}"
                                f"__merge_{md}"
                                f"__format_{_safe_slug(args.macs2_format)}"
                                f"__gsize_{_safe_slug(args.macs2_gsize)}"
                            )
                            out_bed = os.path.join(beds_dir, cand_name + ".bed")

                            merge_cmd = ["bedtools", "merge", "-d", str(int(md)), "-i", sorted_path]
                            merged = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            if merged.returncode != 0:
                                die(f"bedtools merge failed for target={tgt} q={qv} merge={md}:\n{merged.stderr}")

                            with open(out_bed, "w") as f:
                                f.write(merged.stdout)
                            beds_written += 1

                            widths = bed_widths(out_bed)
                            st = width_stats(widths)

                            # Plot (one page)
                            title = (
                                f"{cand_name}\n"
                                f"n={st['n_peaks']} | median={st['median_width']:.1f} bp | "
                                f"p25={st['p25_width']:.1f} | p75={st['p75_width']:.1f}"
                            )
                            fig = plot_width_histogram(
                                widths,
                                title=title,
                                bin_bp=int(args.plot_bin),
                                max_width=int(args.plot_max),
                            )
                            pdf.savefig(fig)
                            plt.close(fig)

                            row = {
                                "target": tgt,
                                "macs2_mode": args.macs2_mode,
                                "macs2_format": args.macs2_format,
                                "macs2_gsize": args.macs2_gsize,
                                "qvalue": qv,
                                "qscore_thr": qscore_thr,
                                "merge_d": int(md),
                                "bed": os.path.basename(out_bed),
                                **st,
                            }
                            summary_rows.append(row)

            # write summary
            df_sum = pd.DataFrame(summary_rows)
            # nice ordering
            col_order = [
                "target",
                "macs2_mode", "macs2_format", "macs2_gsize",
                "qvalue", "qscore_thr", "merge_d", "bed",
                "n_peaks",
                "median_width", "mean_width", "sd_width",
                "min_width", "max_width",
                "p25_width", "p75_width",
            ]
            df_sum = df_sum[[c for c in col_order if c in df_sum.columns]]
            df_sum.to_csv(summary_tsv, sep="\t", index=False)

            log_done("Consensus grid search finished (BEDs + TSV + multi-page PDF)", t, ctrl=ctrl)
            ctrl.end_step()

            log_fh.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
            log_fh.flush()

    finally:
        ctrl.end_step()

    runtime_min = (time.time() - t0) / 60.0

    summary = []
    summary.append("")
    summary.append("")
    summary.append("GNOMES ran successfully ✅")
    summary.append("")
    summary.append("Command: consensus")
    summary.append(f"Samples: {meta.shape[0] if meta is not None else 'NA'}")
    summary.append(f"Targets: {sorted(meta['target'].unique().tolist()) if meta is not None else 'NA'}")
    summary.append(f"Conditions: {sorted(meta['condition'].unique().tolist()) if meta is not None else 'NA'}")
    summary.append(f"Control BAMs: {'YES (MACS2 -c pooled per condition)' if has_ctrl_any else 'NO'}")
    summary.append("")
    summary.append("Grid:")
    summary.append(f"  - qvalues: {qvals}")
    summary.append(f"  - merges: {merges}")
    summary.append(f"  - candidates total: {n_candidates_total}")
    summary.append("")
    summary.append("Outputs:")
    summary.append(f"  - BEDs: {beds_dir}  (written: {beds_written})")
    summary.append(f"  - Summary TSV: {summary_tsv}")
    summary.append(f"  - Width PDF: {pdf_path}")
    summary.append("")
    summary.append(f"Output directory: {args.outdir}")
    summary.append(f"Log file: {log_path}")
    summary.append(f"Runtime: {runtime_min:.1f} minutes")
    summary.append("")
    summary.append("")
    summary.append("Thank you for using GNOMES!")
    summary.append("")
    summary.append("If you use GNOMES in your work, please cite: Roule et al., [YEAR], [JOURNAL].")

    print_final_gnome_summary(summary)


if __name__ == "__main__":
    main()