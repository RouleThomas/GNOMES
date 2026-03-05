#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
import time
from datetime import datetime
import threading
from typing import Optional, Tuple, Dict, List

import pandas as pd
import numpy as np

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
 *= *          #*  =#+        :* #
  .#*                         *#=
    **:*=::=+         :+:.-*==*
     *        *     -:       *+
     +#*******+-=====*##******

"""
    print(splash, flush=True)


# -----------------------------
# GNOME walker (ONLY when interactive; never written to your log file)
# -----------------------------
class GnomeWalker:
    """
    Animated gnome shown ONLY when stdout is a TTY.
    Uses alternate screen buffer so it doesn't pollute scrollback.
    """
    def __init__(self, enabled=True, fps=10, track_width=26):
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
            " *= *          #*  =#+        :* # ",
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
        self._ansi("\x1b[?25l")  # hide cursor

    def _exit_alt_screen(self):
        self._ansi("\x1b[?25h")  # show cursor
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

        # reserve space
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

        # clear block
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
    Start walker only if a step lasts longer than delay_s.
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

    def set_label(self, label: str):
        with self._lock:
            self._label = label
            if self._walking:
                self.walker.set_label(label)

    def end_step(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._walking:
                self.walker.stop()
                self._walking = False

    def announce(self, line: str):
        print(line, flush=True)


# -----------------------------
# Pretty logging + timing
# -----------------------------
def now():
    return datetime.now().strftime("%H:%M:%S")


def log_step(msg, start_time=None, ctrl: Optional[DelayedWalkController] = None):
    if start_time:
        elapsed = time.time() - start_time
        line = f"[{now()}] {msg} (done in {elapsed:.1f}s)"
    else:
        line = f"[{now()}] {msg}"
    if ctrl is not None:
        ctrl.announce(line)
    else:
        print(line, flush=True)


def run_cmd(cmd, log_fh):
    cmd_str = " ".join(cmd)
    log_fh.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] CMD: {cmd_str}\n")
    log_fh.flush()

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_fh.write(res.stdout)
    log_fh.flush()

    if res.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd_str}\n{res.stdout}")


def ensure_exe(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")


def safe_name(s: str) -> str:
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in str(s)])


def print_launch_summary_normalize(args, meta: Optional[pd.DataFrame] = None):
    """
    Console-only summary (NOT written to the log), matching the style of diffbind step.
    If meta is provided, prints counts and whether control BAMs are present.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("GNOMES normalize — configuration")
    lines.append("-" * 72)
    lines.append("Inputs:")
    lines.append(f"  - meta: {args.meta}")
    lines.append(f"  - outdir: {args.outdir}")
    lines.append(f"  - chrom-sizes: {args.chrom_sizes}")
    if args.blacklist:
        lines.append(f"  - blacklist: {args.blacklist}")
    else:
        lines.append("  - blacklist: (none)")

    lines.append("")
    lines.append("Read extension / coverage:")
    lines.append(f"  - mode: {args.mode}")
    if args.mode == "SE":
        lines.append(f"  - se-fragment-length: {args.se_fragment_length}")
    lines.append(f"  - threads: {args.threads}")

    lines.append("")
    lines.append("")
    lines.append("QC:")
    if args.no_qc:
        lines.append("  - QC plots: OFF (--no-qc)")
    else:
        lines.append("  - QC plots: ON (09_qc)")
        lines.append("    • multiBigwigSummary bins → plotPCA + plotCorrelation")
        lines.append("    • produced for RAW and NORMALIZED views, per target")

    lines.append("")
    lines.append("Temporary folders / cleanup:")
    lines.append("  - always kept:")
    lines.append("    • 01_raw_bigwig, 06_normalized_bigwig, 08_median_bigwig")
    if args.no_qc:
        lines.append("    • 09_qc: (skipped)")
    else:
        lines.append("    • 09_qc")
    if args.keep_temp:
        lines.append("  - intermediate folders: KEPT (--keep-temp)")
        lines.append("    • 02_bedgraph, 03_bedgraph_blacklist, 04_local_maxima, 05_normalized_bedgraph, 07_median_bedgraph")
    else:
        lines.append("  - intermediate folders: REMOVED (default)")
        lines.append("    • 02_bedgraph, 03_bedgraph_blacklist, 04_local_maxima, 05_normalized_bedgraph, 07_median_bedgraph")

    if meta is not None:
        lines.append("")
        lines.append("Meta summary:")
        lines.append(f"  - samples: {meta.shape[0]}")
        targets = list(pd.unique(meta["target"]))
        conds = list(pd.unique(meta["condition"]))
        lines.append(f"  - targets: {targets}")
        lines.append(f"  - conditions: {conds}")
        has_ctrl = (meta[OPTIONAL_CONTROL_COL].astype(str).str.strip() != "").any()
        lines.append(f"  - bam_control: {'present (bamCompare path enabled)' if has_ctrl else 'not present'}")

    lines.append("=" * 72)
    print("\n".join(lines), flush=True)


def print_final_gnome_summary(summary_lines: List[str]):
    """
    Same end-style as diffbind: ASCII gnome + summary text on the right.
    """
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


# -----------------------------
# Core logic
# -----------------------------
def read_meta(meta_path):
    df = pd.read_csv(meta_path, sep="\t", dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing} (required: {REQUIRED_COLS})")

    # optional control column
    if OPTIONAL_CONTROL_COL not in df.columns:
        df[OPTIONAL_CONTROL_COL] = ""

    if (df["sample_id"].astype(str).str.strip() == "").any():
        raise ValueError("Metadata contains empty sample_id.")
    if df["sample_id"].duplicated().any():
        dups = df[df["sample_id"].duplicated()]["sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id detected: {dups}")

    # validate BAMs
    for bam in df["bam"]:
        bam = str(bam).strip()
        if bam == "":
            raise ValueError("Metadata contains empty bam path(s).")
        if (not bam.endswith(".bam")) or (not os.path.exists(bam)):
            raise ValueError(f"Invalid BAM file: {bam}")

    # validate control BAMs (if provided)
    for bamc in df[OPTIONAL_CONTROL_COL].tolist():
        bamc = str(bamc).strip()
        if bamc == "":
            continue
        if (not bamc.endswith(".bam")) or (not os.path.exists(bamc)):
            raise ValueError(f"Invalid bam_control BAM file: {bamc}")

    return df


def find_local_maxima_bedgraph(bedgraph_path, out_bed_path):
    data = pd.read_csv(
        bedgraph_path, sep="\t", header=None,
        names=["chrom", "start", "end", "score"]
    )

    scores = data["score"].to_numpy()
    if len(scores) < 3:
        open(out_bed_path, "w").close()
        return 0

    idx = []
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
            idx.append(i)

    data.iloc[idx].to_csv(out_bed_path, sep="\t", index=False, header=False)
    return len(idx)


def percentile_99_from_maxima(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["c", "s", "e", "v"])
    if df.empty:
        return None
    return float(np.percentile(df["v"], 99))


def build_pca_style_vectors_by_target(meta: pd.DataFrame, ref_condition: Optional[str] = None):
    meta = meta.copy().reset_index(drop=True)

    conds = list(pd.unique(meta["condition"]))
    if not conds:
        raise ValueError("No conditions found in metadata.")
    if ref_condition is None or ref_condition not in conds:
        ref_condition = conds[0]

    grey_palette = ["dimgray", "darkgray", "lightgray", "#555555", "#888888"]
    cond_to_color = {ref_condition: "black"}
    others = [c for c in conds if c != ref_condition]
    for i, c in enumerate(others):
        cond_to_color[c] = grey_palette[min(i, len(grey_palette) - 1)]

    marker_palette = ["o", "s", "^", "v", "D", "P", "X", "*", "+", "x"]

    rep_idx = []
    per_cond_counter = {}
    for c in meta["condition"].tolist():
        per_cond_counter[c] = per_cond_counter.get(c, 0) + 1
        rep_idx.append(per_cond_counter[c])
    meta["rep_idx"] = rep_idx

    labels = meta["sample_id"].tolist()
    colors = [cond_to_color[c] for c in meta["condition"].tolist()]
    markers = [marker_palette[(i - 1) % len(marker_palette)] for i in meta["rep_idx"].tolist()]

    return labels, colors, markers, ref_condition


def run_qc_deeptools_per_target(
    meta_target: pd.DataFrame,
    bw_dir: str,
    out_prefix: str,
    log_fh,
    threads: int = 1,
    ref_condition: Optional[str] = None,
):
    bw_paths = [os.path.join(bw_dir, f"{sid}.bw") for sid in meta_target["sample_id"].tolist()]
    missing = [p for p in bw_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError("Missing bigWigs for QC:\n" + "\n".join(missing))

    npz_path = f"{out_prefix}_multiBigwigSummary.npz"
    pca_pdf = f"{out_prefix}_PCA.pdf"
    hm_pdf = f"{out_prefix}_heatmap.pdf"

    labels, colors, markers, _ = build_pca_style_vectors_by_target(meta_target, ref_condition=ref_condition)

    cmd_mbs = ["multiBigwigSummary", "bins", "-b"] + bw_paths + [
        "-o", npz_path,
        "--numberOfProcessors", str(max(1, int(threads)))
    ]
    run_cmd(cmd_mbs, log_fh)

    cmd_pca = [
        "plotPCA",
        "-in", npz_path,
        "--transpose",
        "--ntop", "0",
        "--labels"
    ] + labels + [
        "--colors"
    ] + colors + [
        "--markers"
    ] + markers + [
        "--plotWidth", "8",
        "--plotHeight", "8",
        "-o", pca_pdf
    ]
    run_cmd(cmd_pca, log_fh)

    cmd_hm = [
        "plotCorrelation",
        "-in", npz_path,
        "--corMethod", "pearson",
        "--skipZeros",
        "--plotTitle", "Pearson Correlation",
        "--removeOutliers",
        "--labels"
    ] + labels + [
        "--whatToPlot", "heatmap",
        "--colorMap", "bwr",
        "--plotNumbers",
        "-o", hm_pdf
    ]
    run_cmd(cmd_hm, log_fh)

    return pca_pdf, hm_pdf, npz_path


def build_median_tracks(meta: pd.DataFrame, norm_bw_dir: str, chrom_sizes: str, outdir: str, log_fh):
    med_bg_dir = os.path.join(outdir, "07_median_bedgraph")
    med_bw_dir = os.path.join(outdir, "08_median_bigwig")
    os.makedirs(med_bg_dir, exist_ok=True)
    os.makedirs(med_bw_dir, exist_ok=True)

    groups = meta.groupby(["condition", "target"], sort=False)

    for (cond, tgt), g in groups:
        cond_s = safe_name(cond)
        tgt_s = safe_name(tgt)
        prefix = f"{cond_s}__{tgt_s}"

        bw_list = []
        for sid in g["sample_id"].tolist():
            bw = os.path.join(norm_bw_dir, f"{sid}.norm99.bw")
            if not os.path.exists(bw):
                raise RuntimeError(f"Missing normalized bigWig for median track: {bw}")
            bw_list.append(bw)

        out_bg = os.path.join(med_bg_dir, f"{prefix}_median.bedGraph")
        out_bg_sorted = os.path.join(med_bg_dir, f"{prefix}_median.sorted.bedGraph")
        out_bw = os.path.join(med_bw_dir, f"{prefix}_median.bw")

        cmd_wiggle = ["wiggletools", "write_bg", out_bg, "median"] + bw_list
        run_cmd(cmd_wiggle, log_fh)

        with open(out_bg_sorted, "w") as out:
            subprocess.run(["bedtools", "sort", "-i", out_bg], stdout=out, check=True)

        run_cmd(["bedGraphToBigWig", out_bg_sorted, chrom_sizes, out_bw], log_fh)

    return med_bg_dir, med_bw_dir


def rm_tree_if_exists(path, log_fh=None):
    if os.path.exists(path):
        if log_fh is not None:
            log_fh.write(f"Removing temporary folder: {path}\n")
            log_fh.flush()
        shutil.rmtree(path, ignore_errors=True)


# -----------------------------
# Main
# -----------------------------
def main():
    t0 = time.time()
    print_gnome_splash()

    parser = argparse.ArgumentParser(
        prog="GNOMES norm",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "GNOMES normalize\n"
            "---------------\n"
            "Convert BAMs into raw bigWigs, estimate per-sample local-maxima P99, then\n"
            "scale signals to a reference sample per target and write normalized bigWigs.\n"
        ),
        epilog=(
            "Example:\n"
            "  GNOMES norm \\\n"
            "    --meta meta/samples.tsv \\\n"
            "    --outdir output/gnomes_run \\\n"
            "    --blacklist meta/hg38-blacklist.v2.bed \\\n"
            "    --chrom-sizes meta/GRCh38_chrom_sizes.tab \\\n"
            "    --threads 8 \\\n"
            "    --mode SE \\\n"
            "    --se-fragment-length 200\n\n"
            "Meta file requirements:\n"
            "  Required columns: sample_id, bam, condition, target\n"
            "  Optional column:  bam_control (control BAM for IP-control subtraction)\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "normalize",
        formatter_class=argparse.RawTextHelpFormatter,
        help="Run GNOMES normalization (BAM → RAW bigWig → P99 scaling → normalized bigWig).",
        description=(
            "This command will:\n"
            "  1) Generate RAW bigWigs from BAMs (bamCoverage OR bamCompare if bam_control)\n"
            "  2) Convert RAW bigWigs → bedGraphs (bigWigToBedGraph)\n"
            "  3) (Optional) remove blacklist intervals\n"
            "  4) Identify local maxima per bedGraph and compute P99 per sample\n"
            "  5) Compute per-sample scaling factors within each target\n"
            "  6) Apply scaling and write normalized bigWigs: <sample_id>.norm99.bw\n"
            "  7) Build median tracks per (condition, target) using wiggletools median\n"
            "  8) (Optional) QC plots (PCA + correlation heatmaps) for RAW + NORMALIZED\n"
        ),
    )

    # -----------------------------
    # Required inputs
    # -----------------------------
    req = p.add_argument_group("Required inputs")

    req.add_argument("--meta", required=True,
                     help="samples.tsv with columns: sample_id bam condition target [bam_control optional]")

    req.add_argument("--outdir", required=True,
                     help="Output directory")

    req.add_argument("--chrom-sizes", required=True, dest="chrom_sizes",
                     help="Tab-separated file with chrom sizes required for bedGraphToBigWig (e.g. GRCh38_chrom_sizes.tab)")

    # -----------------------------
    # Optional inputs
    # -----------------------------
    opt = p.add_argument_group("Optional inputs")

    opt.add_argument("--blacklist", default=None,
                     help="Optional BED blacklist to remove from bedGraphs")

    # -----------------------------
    # Coverage / read layout
    # -----------------------------
    cov = p.add_argument_group("Coverage / read layout")

    cov.add_argument("--mode", choices=["PE", "SE"], required=True,
                     help="Read layout: Paired-End (PE) or Single-End (SE) (required)")

    cov.add_argument("--se-fragment-length", type=int, default=200,
                     help="If --mode SE, extend reads to this fragment length (default 200)")

    cov.add_argument("--threads", type=int, default=4,
                     help="Threads for deepTools (default 4)")

    # -----------------------------
    # Retention / QC
    # -----------------------------
    qc = p.add_argument_group("Retention / QC")

    qc.add_argument("--keep-temp", action="store_true",
                    help=("Keep intermediate folders (02..05 and 07).\n"
                          "Default: delete them at the end."))

    qc.add_argument("--no-qc", action="store_true",
                    help=("Skip QC plots (09_qc).\n"
                          "Default: generate QC (RAW + NORMALIZED) per target."))

    # -----------------------------
    # Miscellaneous
    # -----------------------------
    misc = p.add_argument_group("Miscellaneous")

    misc.add_argument("--no-walk", action="store_true",
                      help="Disable GNOMES walking animation")


    args = parser.parse_args()

    if args.command != "normalize":
        raise RuntimeError("Only 'normalize' is implemented in this script.")

    # executables always needed
    for exe in [
        "bamCoverage", "bigWigToBedGraph", "bedGraphToBigWig", "bedtools",
        "wiggletools"
    ]:
        ensure_exe(exe)

    # bamCompare only needed if user uses bam_control in meta (we verify after reading meta),
    # but requiring it upfront gives a clean error message.
    ensure_exe("bamCompare")

    # QC tools only if QC enabled
    if not args.no_qc:
        for exe in ["multiBigwigSummary", "plotPCA", "plotCorrelation"]:
            ensure_exe(exe)

    os.makedirs(args.outdir, exist_ok=True)

    raw_bw = f"{args.outdir}/01_raw_bigwig"
    bg = f"{args.outdir}/02_bedgraph"
    bg_bl = f"{args.outdir}/03_bedgraph_blacklist"
    maxima = f"{args.outdir}/04_local_maxima"
    norm_bg = f"{args.outdir}/05_normalized_bedgraph"
    norm_bw = f"{args.outdir}/06_normalized_bigwig"

    for d in [raw_bw, bg, bg_bl, maxima, norm_bg, norm_bw]:
        os.makedirs(d, exist_ok=True)

    log_path = f"{args.outdir}/GNOMES_normalize.log"


    WALK_FPS = 10
    WALK_WIDTH = 26
    WALK_DELAY = 10.0

    walker = GnomeWalker(enabled=(not args.no_walk), fps=WALK_FPS, track_width=WALK_WIDTH)
    ctrl = DelayedWalkController(walker, delay_s=WALK_DELAY)


    # bookkeeping for end summary
    meta = None
    has_ctrl = False
    targets = []
    conds = []
    n_samples = 0
    n_targets = 0
    qc_ran = False
    kept_temp = bool(args.keep_temp)

    try:
        with open(log_path, "w") as log:
            log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
            log.write(f"Args: {vars(args)}\n")
            log.flush()

            # -----------------------------
            # Load metadata
            # -----------------------------
            ctrl.begin_step("Step 1/7 — Reading metadata")
            t = time.time()
            log_step("Reading metadata", ctrl=ctrl)

            meta = read_meta(args.meta)
            n_samples = int(meta.shape[0])
            targets = list(pd.unique(meta["target"]))
            conds = list(pd.unique(meta["condition"]))
            n_targets = len(targets)
            has_ctrl = (meta[OPTIONAL_CONTROL_COL].astype(str).str.strip() != "").any()

            # print run summary (console) after meta is valid
            print_launch_summary_normalize(args, meta=meta)

            log_step("Reading metadata", start_time=t, ctrl=ctrl)
            ctrl.end_step()

            # -----------------------------
            # BAM → bigWig → bedGraph → blacklist
            # (if bam_control provided, do IP-control subtraction with bamCompare)
            # -----------------------------
            ctrl.begin_step("Step 2/7 — BAM → bigWig → bedGraph")
            t = time.time()
            log_step("Converting BAM → bigWig → bedGraph", ctrl=ctrl)

            for i, r in meta.reset_index(drop=True).iterrows():
                sid = r.sample_id
                ctrl.set_label(f"Step 2/7 — BAM→BW→BG: {sid} ({i+1}/{meta.shape[0]})")

                bw = f"{raw_bw}/{sid}.bw"
                bgf = f"{bg}/{sid}.bedGraph"
                bgf_bl = f"{bg_bl}/{sid}.bedGraph"

                bam_control = str(r.bam_control).strip()

                if bam_control != "":
                    # subtract control at BAM→BW stage: bamCompare --operation subtract
                    cmd = [
                        "bamCompare",
                        "--bamfile1", r.bam,
                        "--bamfile2", bam_control,
                        "--outFileName", bw,
                        "--outFileFormat", "bigwig",
                        "--binSize", "1",
                        "--numberOfProcessors", str(args.threads),
                        "--operation", "subtract",
                        "--scaleFactorsMethod", "None",
                    ]
                    if args.mode == "PE":
                        cmd += ["--extendReads"]
                    else:
                        cmd += ["--extendReads", str(args.se_fragment_length)]

                    run_cmd(cmd, log)
                else:
                    cmd = [
                        "bamCoverage",
                        "--bam", r.bam,
                        "--outFileName", bw,
                        "--outFileFormat", "bigwig",
                        "--binSize", "1",
                        "--numberOfProcessors", str(args.threads),
                        "--scaleFactor", "1",
                    ]
                    if args.mode == "PE":
                        cmd += ["--extendReads"]
                    else:
                        cmd += ["--extendReads", str(args.se_fragment_length)]

                    run_cmd(cmd, log)

                run_cmd(["bigWigToBedGraph", bw, bgf], log)

                if args.blacklist:
                    with open(bgf_bl, "w") as out:
                        subprocess.run(
                            ["bedtools", "intersect", "-v", "-a", bgf, "-b", args.blacklist],
                            stdout=out, check=True
                        )
                else:
                    shutil.copy(bgf, bgf_bl)

            log_step("BAM → bigWig → bedGraph", start_time=t, ctrl=ctrl)
            ctrl.end_step()

            # -----------------------------
            # Local maxima
            # -----------------------------
            ctrl.begin_step("Step 3/7 — Identifying local maxima")
            t = time.time()
            log_step("Identifying local maxima", ctrl=ctrl)

            for i, sid in enumerate(meta.sample_id.tolist()):
                ctrl.set_label(f"Step 3/7 — Local maxima: {sid} ({i+1}/{meta.shape[0]})")
                find_local_maxima_bedgraph(
                    f"{bg_bl}/{sid}.bedGraph",
                    f"{maxima}/{sid}.local_maxima.bed"
                )

            log_step("Identifying local maxima", start_time=t, ctrl=ctrl)
            ctrl.end_step()

            # -----------------------------
            # P99 + scaling
            # -----------------------------
            ctrl.begin_step("Step 4/7 — Computing P99 + scaling factors")
            t = time.time()
            log_step("Computing 99th percentiles and scaling factors", ctrl=ctrl)

            p99 = {}
            for i, sid in enumerate(meta.sample_id.tolist()):
                ctrl.set_label(f"Step 4/7 — P99: {sid} ({i+1}/{meta.shape[0]})")
                val = percentile_99_from_maxima(f"{maxima}/{sid}.local_maxima.bed")
                if val is None:
                    raise RuntimeError(f"No maxima for {sid}")
                p99[sid] = val

            # reference per target = first sample listed for that target in the meta file
            ref_by_target = {}
            for tname in meta.target.unique():
                ref = meta[meta.target == tname].iloc[0].sample_id
                ref_by_target[tname] = ref

            scaling = {}
            for _, r in meta.iterrows():
                scaling[r.sample_id] = p99[ref_by_target[r.target]] / p99[r.sample_id]

            log_step("Computing P99 + scaling factors", start_time=t, ctrl=ctrl)
            ctrl.end_step()

            # -----------------------------
            # Write scaling factor table
            # -----------------------------
            ctrl.begin_step("Step 5/7 — Writing scaling factors table")
            t = time.time()
            log_step("Writing scaling factor table", ctrl=ctrl)

            sf_path = f"{args.outdir}/scaling_factors.tsv"
            sf_rows = []
            for _, r in meta.iterrows():
                sid = r.sample_id
                tgt = r.target
                ref_sid = ref_by_target[tgt]
                sf_rows.append({
                    "sample_id": sid,
                    "condition": r.condition,
                    "target": tgt,
                    "bam": r.bam,
                    "bam_control": str(r.bam_control).strip(),
                    "p99": p99[sid],
                    "reference_sample": ref_sid,
                    "reference_p99": p99[ref_sid],
                    "scaling_factor": scaling[sid],
                })
            pd.DataFrame(sf_rows).to_csv(sf_path, sep="\t", index=False)
            log.write(f"Wrote scaling factors: {sf_path}\n")
            log.flush()

            log_step("Writing scaling factors table", start_time=t, ctrl=ctrl)
            ctrl.end_step()

            # -----------------------------
            # Normalize + bigWig
            # -----------------------------
            ctrl.begin_step("Step 6/7 — Generating normalized bigWigs")
            t = time.time()
            log_step("Generating normalized bigWig files", ctrl=ctrl)

            for i, sid in enumerate(meta.sample_id.tolist()):
                ctrl.set_label(f"Step 6/7 — Normalize: {sid} ({i+1}/{meta.shape[0]})")

                df = pd.read_csv(f"{bg_bl}/{sid}.bedGraph", sep="\t", header=None)
                df[3] *= scaling[sid]

                norm_bgf = f"{norm_bg}/{sid}.norm99.bedGraph"
                norm_bgf_sorted = f"{norm_bg}/{sid}.norm99.sorted.bedGraph"
                norm_bwf = f"{norm_bw}/{sid}.norm99.bw"

                df.to_csv(norm_bgf, sep="\t", index=False, header=False)
                with open(norm_bgf_sorted, "w") as out:
                    subprocess.run(["bedtools", "sort", "-i", norm_bgf], stdout=out, check=True)

                run_cmd(["bedGraphToBigWig", norm_bgf_sorted, args.chrom_sizes, norm_bwf], log)

            log_step("Generating normalized bigWigs", start_time=t, ctrl=ctrl)
            ctrl.end_step()

            # -----------------------------
            # Median tracks
            # -----------------------------
            ctrl.begin_step("Step 7/7 — Generating median tracks")
            t = time.time()
            log_step("Generating median tracks per (condition, target) using wiggletools median", ctrl=ctrl)

            med_bg_dir, med_bw_dir = build_median_tracks(
                meta=meta,
                norm_bw_dir=norm_bw,
                chrom_sizes=args.chrom_sizes,
                outdir=args.outdir,
                log_fh=log
            )

            log_step("Generating median tracks", start_time=t, ctrl=ctrl)
            ctrl.end_step()

            # -----------------------------
            # QC per target (RAW + NORMALIZED) [OPTIONAL]
            # -----------------------------
            if not args.no_qc:
                ctrl.begin_step("QC — Generating PCA + heatmap per target (raw + normalized)")
                t = time.time()
                log_step("Generating QC plots per target (PCA + heatmap) from RAW + NORMALIZED bigWigs", ctrl=ctrl)

                qc_dir = os.path.join(args.outdir, "09_qc")
                os.makedirs(qc_dir, exist_ok=True)

                # reference condition = first condition in meta file
                cond_order = list(pd.unique(meta["condition"]))
                ref_condition = cond_order[0] if cond_order else None

                # NORMALIZED view: symlinks <sid>.bw -> <sid>.norm99.bw (created once)
                norm_view = os.path.join(qc_dir, "_norm_view_bw")
                os.makedirs(norm_view, exist_ok=True)
                for sid in meta["sample_id"].tolist():
                    src = os.path.join(norm_bw, f"{sid}.norm99.bw")
                    dst = os.path.join(norm_view, f"{sid}.bw")
                    if os.path.islink(dst) or os.path.exists(dst):
                        os.remove(dst)
                    os.symlink(os.path.abspath(src), dst)

                targets_list = list(pd.unique(meta["target"]))
                for ti, tgt in enumerate(targets_list, start=1):
                    tgt_s = safe_name(tgt)
                    meta_t = meta.loc[meta["target"] == tgt].reset_index(drop=True)

                    ctrl.set_label(f"QC — target {ti}/{len(targets_list)}: {tgt}")

                    # RAW
                    raw_prefix = os.path.join(qc_dir, f"raw__{tgt_s}")
                    run_qc_deeptools_per_target(
                        meta_target=meta_t,
                        bw_dir=raw_bw,
                        out_prefix=raw_prefix,
                        log_fh=log,
                        threads=args.threads,
                        ref_condition=ref_condition
                    )
                    os.replace(f"{raw_prefix}_PCA.pdf", os.path.join(qc_dir, f"PCA_raw__{tgt_s}.pdf"))
                    os.replace(f"{raw_prefix}_heatmap.pdf", os.path.join(qc_dir, f"heatmap_raw__{tgt_s}.pdf"))

                    # NORMALIZED
                    norm_prefix = os.path.join(qc_dir, f"normalized__{tgt_s}")
                    run_qc_deeptools_per_target(
                        meta_target=meta_t,
                        bw_dir=norm_view,
                        out_prefix=norm_prefix,
                        log_fh=log,
                        threads=args.threads,
                        ref_condition=ref_condition
                    )
                    os.replace(f"{norm_prefix}_PCA.pdf", os.path.join(qc_dir, f"PCA_normalized__{tgt_s}.pdf"))
                    os.replace(f"{norm_prefix}_heatmap.pdf", os.path.join(qc_dir, f"heatmap_normalized__{tgt_s}.pdf"))

                qc_ran = True
                log_step("QC plots written (09_qc)", start_time=t, ctrl=ctrl)
                ctrl.end_step()

            # -----------------------------
            # Cleanup temp folders [DEFAULT: delete]
            # Keep: 01_raw_bigwig, 06_normalized_bigwig, 08_median_bigwig, 09_qc (if generated)
            # Delete (unless --keep-temp): 02,03,04,05,07
            # -----------------------------
            if not args.keep_temp:
                ctrl.begin_step("Cleanup — Removing intermediate folders")
                t = time.time()
                log_step("Removing intermediate folders (use --keep-temp to keep them)", ctrl=ctrl)

                rm_tree_if_exists(bg, log)
                rm_tree_if_exists(bg_bl, log)
                rm_tree_if_exists(maxima, log)
                rm_tree_if_exists(norm_bg, log)
                rm_tree_if_exists(os.path.join(args.outdir, "07_median_bedgraph"), log)

                log_step("Cleanup", start_time=t, ctrl=ctrl)
                ctrl.end_step()
            else:
                log.write("Keeping intermediate folders because --keep-temp was set.\n")
                log.flush()

            log.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
            log.flush()

    finally:
        # stop animation / cancel pending timer, no matter what happens
        try:
            ctrl.end_step()
        except Exception:
            pass

    # End-of-run message + GNOMES-style ASCII summary (matching diffbind)
    runtime_min = (time.time() - t0) / 60.0

    summary = []
    summary.append("")
    summary.append("")
    summary.append("GNOMES ran successfully ✅")
    summary.append("")
    summary.append("Command: normalize")
    summary.append(f"Samples: {n_samples}")
    summary.append(f"Targets: {targets if targets else 'NA'}")
    summary.append(f"Conditions: {conds if conds else 'NA'}")
    summary.append(f"Control BAMs: {'YES (bamCompare subtract)' if has_ctrl else 'NO'}")
    summary.append(f"QC: {'ON (09_qc)' if qc_ran else 'OFF'}")
    summary.append(f"Keep temp: {'YES' if kept_temp else 'NO'}")
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