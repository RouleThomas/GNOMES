#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
import time
import re
from datetime import datetime
import threading

import pandas as pd
import numpy as np

REQUIRED_COLS = ["sample_id", "bam", "condition", "target"]


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
    def __init__(self, enabled=True, fps=8, track_width=26):
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
    def __init__(self, walker: GnomeWalker, delay_s: float = 5.0):
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
        # Progress lines go to stdout (scrollback). Walker never touches your log file anyway.
        print(line, flush=True)


# -----------------------------
# Helpers
# -----------------------------
def now():
    return datetime.now().strftime("%H:%M:%S")


def log_step(msg, start_time=None, ctrl: DelayedWalkController | None = None):
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


# -----------------------------
# Core logic
# -----------------------------
def read_meta(meta_path):
    df = pd.read_csv(meta_path, sep="\t", dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id detected")

    for bam in df["bam"]:
        if not bam.endswith(".bam") or not os.path.exists(bam):
            raise ValueError(f"Invalid BAM file: {bam}")

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


def build_pca_style_vectors_by_target(meta: pd.DataFrame, ref_condition: str | None = None):
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
    ref_condition: str | None = None,
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


# -----------------------------
# Main
# -----------------------------
def main():
    t0 = time.time()
    print_gnome_splash()

    parser = argparse.ArgumentParser("normdb")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("normalize")
    p.add_argument("--meta", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--blacklist")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--mode", choices=["PE", "SE"], required=True)
    p.add_argument("--se-fragment-length", type=int, default=200)
    p.add_argument("--reference", default="auto")

    # walker controls
    p.add_argument("--no-walk", action="store_true", help="Disable the GNOME walking animation")
    p.add_argument("--walk-fps", type=int, default=8, help="Animation FPS (default 8)")
    p.add_argument("--walk-width", type=int, default=26, help="How far GNOME walks horizontally (default 26)")
    p.add_argument("--walk-delay", type=float, default=5.0, help="Start walking only if a step lasts >= this many seconds (default 5)")

    args = parser.parse_args()

    for exe in [
        "bamCoverage", "bigWigToBedGraph", "bedGraphToBigWig", "bedtools",
        "wiggletools", "multiBigwigSummary", "plotPCA", "plotCorrelation"
    ]:
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

    log_path = f"{args.outdir}/normdb_normalize.log"

    walker = GnomeWalker(enabled=(not args.no_walk), fps=args.walk_fps, track_width=args.walk_width)
    ctrl = DelayedWalkController(walker, delay_s=args.walk_delay)

    try:
        with open(log_path, "w") as log:
            log.write(f"Started: {datetime.now()}\n")
            log.write(f"Args: {vars(args)}\n")
            log.flush()

            # -----------------------------
            # Load metadata
            # -----------------------------
            ctrl.begin_step("Step 1/7 — Reading metadata")
            step = time.time()
            log_step("Reading metadata", ctrl=ctrl)
            meta = read_meta(args.meta)
            ctrl.end_step()

            # -----------------------------
            # BAM → bigWig → bedGraph → blacklist
            # -----------------------------
            ctrl.begin_step("Step 2/7 — BAM → bigWig → bedGraph")
            step = time.time()
            log_step("Converting BAM → bigWig → bedGraph", ctrl=ctrl)
            for i, r in meta.reset_index(drop=True).iterrows():
                sid = r.sample_id
                ctrl.set_label(f"Step 2/7 — BAM→BW→BG: {sid} ({i+1}/{meta.shape[0]})")

                bw = f"{raw_bw}/{sid}.bw"
                bgf = f"{bg}/{sid}.bedGraph"
                bgf_bl = f"{bg_bl}/{sid}.bedGraph"

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

            ctrl.end_step()

            # -----------------------------
            # Local maxima
            # -----------------------------
            ctrl.begin_step("Step 3/7 — Identifying local maxima")
            step = time.time()
            log_step("Identifying local maxima", ctrl=ctrl)
            for i, sid in enumerate(meta.sample_id.tolist()):
                ctrl.set_label(f"Step 3/7 — Local maxima: {sid} ({i+1}/{meta.shape[0]})")
                find_local_maxima_bedgraph(
                    f"{bg_bl}/{sid}.bedGraph",
                    f"{maxima}/{sid}.local_maxima.bed"
                )
            ctrl.end_step()

            # -----------------------------
            # P99 + scaling
            # -----------------------------
            ctrl.begin_step("Step 4/7 — Computing P99 + scaling factors")
            step = time.time()
            log_step("Computing 99th percentiles and scaling factors", ctrl=ctrl)

            p99 = {}
            for i, sid in enumerate(meta.sample_id.tolist()):
                ctrl.set_label(f"Step 4/7 — P99: {sid} ({i+1}/{meta.shape[0]})")
                val = percentile_99_from_maxima(f"{maxima}/{sid}.local_maxima.bed")
                if val is None:
                    raise RuntimeError(f"No maxima for {sid}")
                p99[sid] = val

            ref_by_target = {}
            for t in meta.target.unique():
                ref = meta[meta.target == t].iloc[0].sample_id
                ref_by_target[t] = ref

            scaling = {}
            for _, r in meta.iterrows():
                scaling[r.sample_id] = p99[ref_by_target[r.target]] / p99[r.sample_id]

            ctrl.end_step()

            # -----------------------------
            # Write scaling factor table
            # -----------------------------
            ctrl.begin_step("Step 5/7 — Writing scaling factors table")
            step = time.time()
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
                    "p99": p99[sid],
                    "reference_sample": ref_sid,
                    "reference_p99": p99[ref_sid],
                    "scaling_factor": scaling[sid],
                })
            pd.DataFrame(sf_rows).to_csv(sf_path, sep="\t", index=False)
            log.write(f"Wrote scaling factors: {sf_path}\n")
            log.flush()

            ctrl.end_step()

            # -----------------------------
            # Normalize + bigWig
            # -----------------------------
            ctrl.begin_step("Step 6/7 — Generating normalized bigWigs")
            step = time.time()
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

            ctrl.end_step()

            # -----------------------------
            # Median tracks
            # -----------------------------
            ctrl.begin_step("Step 7/7 — Generating median tracks")
            step = time.time()
            log_step("Generating median tracks per (condition, target) using wiggletools median", ctrl=ctrl)
            med_bg_dir, med_bw_dir = build_median_tracks(
                meta=meta,
                norm_bw_dir=norm_bw,
                chrom_sizes=args.chrom_sizes,
                outdir=args.outdir,
                log_fh=log
            )
            ctrl.end_step()

            # -----------------------------
            # QC per target (RAW + NORMALIZED)
            # -----------------------------
            ctrl.begin_step("QC — Generating PCA + heatmap per target (raw + normalized)")
            step = time.time()
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

            targets = list(pd.unique(meta["target"]))
            for ti, tgt in enumerate(targets, start=1):
                tgt_s = safe_name(tgt)
                meta_t = meta.loc[meta["target"] == tgt].reset_index(drop=True)

                ctrl.set_label(f"QC — target {ti}/{len(targets)}: {tgt}")

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

            ctrl.end_step()

            log.write(f"Finished: {datetime.now()}\n")
            log.flush()

    finally:
        # stop animation / cancel pending timer, no matter what happens
        ctrl.end_step()

    print(f"\n✔ GNOMES normalize finished in {(time.time() - t0)/60:.1f} minutes")
    print(f"Output directory: {args.outdir}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
