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

import pandas as pd

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
 *= *          #*-.=#+        :* #
  .#*                         *#=
    **:*=::=+         :+:.-*==*
     *        *     -:       *+
     +#*******+-=====*##******

"""
    print(splash, flush=True)


# -----------------------------
# GNOME walker (multi-line, only when interactive)
# -----------------------------
class GnomeWalker:
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

    def announce(self, line: str):
        if not self.enabled:
            print(line, flush=True)
            return
        with self._lock:
            self._move_up(self.block_height)
            self._clear_line()
            sys.stdout.write(line + "\n")
            sys.stdout.write("\n" * (self.block_height - 1))
            sys.stdout.flush()
            self._move_up(self.block_height)

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
    Starts the walker only if a step lasts longer than `delay_s`.
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

            self._timer = threading.Timer(self.delay_s, self._start_if_still_needed)
            self._timer.daemon = True
            self._timer.start()

    def _start_if_still_needed(self):
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
# IO helpers
# -----------------------------
def read_meta(meta_path):
    if not os.path.exists(meta_path):
        die(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path, sep="\t", dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        die(f"Metadata missing required columns: {missing} (required: {REQUIRED_COLS})")

    if (df["sample_id"].str.strip() == "").any():
        die("Metadata contains empty sample_id.")
    if df["sample_id"].duplicated().any():
        dups = df[df["sample_id"].duplicated()]["sample_id"].tolist()
        die(f"Metadata contains duplicate sample_id(s): {dups}")

    if (df["bam"].str.strip() == "").any():
        die("Metadata contains empty bam path(s).")
    missing_bams = [b for b in df["bam"].tolist() if b and not os.path.exists(b)]
    if missing_bams:
        die("Some BAMs listed in meta do not exist:\n" + "\n".join(sorted(set(missing_bams))))

    return df


def parse_contrast(s):
    parts = s.split(":")
    if len(parts) != 3:
        die("--contrast must look like: condition:KO:WT")
    return parts[0], parts[1], parts[2]


def read_computeMatrix_vector(txt_path):
    if not os.path.exists(txt_path):
        die(f"computeMatrix matrix file not found: {txt_path}")

    df = pd.read_csv(txt_path, sep="\t", header=None, skiprows=3)
    if df.shape[1] < 1:
        die(f"computeMatrix matrix file seems empty/unexpected: {txt_path}")

    return df.iloc[:, 0].astype(float).tolist()


def bed_to_region_ids(bed_path):
    if not os.path.exists(bed_path):
        die(f"computeMatrix sorted regions file not found: {bed_path}")

    rows = []
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            rows.append((chrom, start, end))

    if not rows:
        die(f"Could not parse any regions from: {bed_path}")

    return [f"{c}_{s}_{e}" for (c, s, e) in rows]


def write_bed3_from_results(results_tsv: str, out_bed: str):
    """
    Accepts results_signif_gain/loss TSV produced by R:
      chr, start, end, region_id, ...
    Writes BED3 (chr start end). Returns number of written rows.
    """
    if not os.path.exists(results_tsv):
        return 0

    df = pd.read_csv(results_tsv, sep="\t", dtype=str).fillna("")
    if df.shape[0] == 0:
        open(out_bed, "w").close()
        return 0

    if all(c in df.columns for c in ["chr", "start", "end"]):
        bed = df[["chr", "start", "end"]].copy()
    elif "region_id" in df.columns:
        tmp = df["region_id"].str.split("_", expand=True)
        if tmp.shape[1] < 3:
            raise ValueError(f"Cannot parse region_id into chr/start/end in: {results_tsv}")
        bed = pd.DataFrame({"chr": tmp[0], "start": tmp[1], "end": tmp[2]})
    else:
        raise ValueError(f"Expected chr/start/end or region_id in: {results_tsv}")

    bed["start"] = pd.to_numeric(bed["start"], errors="coerce").astype("Int64")
    bed["end"] = pd.to_numeric(bed["end"], errors="coerce").astype("Int64")
    bed = bed.dropna(subset=["chr", "start", "end"])
    bed = bed[bed["end"] > bed["start"]]

    bed.to_csv(out_bed, sep="\t", header=False, index=False)
    return bed.shape[0]


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


def qvalue_to_macs2_qscore(qval: float) -> float:
    if qval <= 0 or qval >= 1:
        die(f"--macs2-qvalue must be in (0,1). Got: {qval}")
    return -math.log10(qval)


def filter_macs2_peakfile_by_qscore(in_peak: str, out_bed: str, qscore_thr: float) -> int:
    """
    Reads MACS2 narrowPeak/broadPeak (>= 9 cols) and keeps rows with col9 >= qscore_thr.
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
                s = int(p[1]); e = int(p[2])
            except ValueError:
                continue
            rows.append((p[0], s, e))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    with open(out_bed, "w") as fout:
        for c, s, e in rows:
            fout.write(f"{c}\t{s}\t{e}\n")
    return len(rows)


def print_launch_summary(args):
    lines = []
    lines.append("=" * 72)
    lines.append("GNOME — configuration")
    lines.append("-" * 72)

    lines.append("Inputs:")
    lines.append(f"  - meta: {args.meta}")
    if args.call_peaks:
        lines.append("  - regions: (will be built via --call-peaks)")
    else:
        lines.append(f"  - regions: {args.regions if args.regions else '(missing)'}")
    lines.append(f"  - bigwig-dir (normalized): {args.bigwig_dir}")
    if args.bigwig_dir_raw:
        lines.append(f"  - bigwig-dir-raw: {args.bigwig_dir_raw}")
    else:
        lines.append("  - bigwig-dir-raw: (not provided; raw PCA/heatmap skipped)")

    lines.append("")
    lines.append("DESeq2:")
    lines.append(f"  - contrast: {args.contrast}")
    lines.append(f"  - target: {args.target if args.target else '(auto from meta)'}")
    lines.append(f"  - alpha (padj): {args.alpha}")
    lines.append(f"  - lfc threshold: {args.lfc}")
    lines.append(f"  - min-counts (rowSums): {args.min_counts}")

    lines.append("")
    lines.append("Modes / outputs:")
    lines.append(f"  - regions mode: {'MACS2 consensus (--call-peaks)' if args.call_peaks else 'User BED (--regions)'}")
    lines.append(f"  - deepTools heatmap/profile: {'OFF (--no-plotHeatmap)' if args.no_plotHeatmap else 'ON (normalized only)'}")
    if not args.no_plotHeatmap:
        lines.append(f"    • referencePoint={args.hm_referencePoint}, -b {args.hm_before}, -a {args.hm_after}, threads={args.hm_threads}")

    if args.call_peaks:
        qscore_thr = qvalue_to_macs2_qscore(args.macs2_qvalue)
        lines.append("")
        lines.append("MACS2 (only because --call-peaks enabled):")
        lines.append(f"  - mode: {args.macs2_mode}")
        lines.append(f"  - format: {args.macs2_format}")
        lines.append(f"  - gsize: {args.macs2_gsize}")
        lines.append(f"  - qvalue: {args.macs2_qvalue} (qscore>= {qscore_thr:.6f})")
        lines.append(f"  - peak merge -d: {args.macs2_merge}")

    lines.append("=" * 72)
    print("\n".join(lines), flush=True)




def print_final_gnome_summary(summary_lines):
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


def count_bed_rows(bed_path: str) -> int:
    if not bed_path or not os.path.exists(bed_path):
        return 0
    n = 0
    with open(bed_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                n += 1
    return n


def collect_bigwigs(meta_t: pd.DataFrame, bigwig_dir: str, *, suffix: str, ctrl=None):
    if not os.path.isdir(bigwig_dir):
        die(f"bigwig directory not found or not a directory: {bigwig_dir}")

    bw_paths, bw_labels, missing = [], [], []
    for _, r in meta_t.iterrows():
        sid = r["sample_id"]
        bw = os.path.join(bigwig_dir, f"{sid}{suffix}")
        if not os.path.exists(bw):
            missing.append(bw)
        else:
            bw_paths.append(bw)
            bw_labels.append(sid)

    if missing:
        msg = "Missing bigWig(s):\n" + "\n".join(missing)
        if ctrl is not None:
            ctrl.end_step()
        die(msg)

    return bw_paths, bw_labels


def run_raw_pca_corr_from_counts(*, counts_path: str, coldata_path: str, contrast_var: str,
                                test_level: str, ref_level: str, out_pca: str, out_corr: str,
                                outdir: str, log_fh):
    r_script_path = os.path.join(outdir, "run_pca_corr_only_rawbigwig.R")
    r_script = f"""
suppressPackageStartupMessages({{
  library(DESeq2)
  library(tidyverse)
  library(ggplot2)
}})

counts_file <- "{counts_path}"
coldata_file <- "{coldata_path}"
out_pca <- "{out_pca}"
out_corr <- "{out_corr}"

contrast_var <- "{contrast_var}"
test_level <- "{test_level}"
ref_level <- "{ref_level}"

counts_df <- read.delim(counts_file, check.names=FALSE)
stopifnot("region_id" %in% colnames(counts_df))

count_mat <- counts_df %>% select(-region_id) %>% as.matrix()
rownames(count_mat) <- counts_df$region_id
mode(count_mat) <- "integer"

coldata <- read.delim(coldata_file, check.names=FALSE)
stopifnot("sample_id" %in% colnames(coldata))
stopifnot(contrast_var %in% colnames(coldata))
stopifnot(all(coldata$sample_id == colnames(count_mat)))

dds <- DESeqDataSetFromMatrix(
  countData = count_mat,
  colData = as.data.frame(coldata %>% column_to_rownames("sample_id")),
  design = as.formula(paste0("~ ", contrast_var))
)

dds[[contrast_var]] <- relevel(dds[[contrast_var]], ref = ref_level)
dds <- estimateSizeFactors(dds)
vsd <- vst(dds, blind=FALSE)
mat <- assay(vsd)

pca <- prcomp(t(mat), scale.=FALSE)
pca_df <- as.data.frame(pca$x[, 1:2, drop=FALSE]) %>%
  rownames_to_column("sample_id") %>%
  left_join(coldata, by="sample_id")

conds <- unique(pca_df[[contrast_var]])
ref_condition <- ref_level

grey_palette <- c("grey", "grey50", "grey65", "grey35", "grey80", "grey20")
cond_to_color <- setNames(rep("grey", length(conds)), conds)
cond_to_color[ref_condition] <- "black"

others <- setdiff(conds, ref_condition)
for (i in seq_along(others)) {{
  cond_to_color[others[i]] <- grey_palette[min(i, length(grey_palette))]
}}

p_pca <- ggplot(pca_df, aes(x=PC1, y=PC2, color=.data[[contrast_var]], label=sample_id)) +
  geom_point(size=3) +
  geom_text(vjust=-0.8, size=3) +
  scale_color_manual(values=cond_to_color) +
  theme_bw() +
  labs(
    title = paste0("PCA (vst counts from RAW bigWigs) — ref=", ref_condition),
    x = paste0("PC1 (", round(100*summary(pca)$importance[2,1], 1), "%)"),
    y = paste0("PC2 (", round(100*summary(pca)$importance[2,2], 1), "%)")
  ) +
  guides(color=guide_legend(title=contrast_var))

ggsave(out_pca, plot=p_pca, width=7.5, height=6.5)

cormat <- cor(mat, method="pearson")
pdf(out_corr, width=8.5, height=8.5)
heatmap(cormat, symm=TRUE, margins=c(12,12), main="Sample correlation (vst counts from RAW bigWigs)")
dev.off()
"""
    with open(r_script_path, "w") as f:
        f.write(r_script)

    run_cmd(["Rscript", r_script_path], log_fh=log_fh)


# -----------------------------
# Main
# -----------------------------
def main():
    t0 = time.time()
    print_gnome_splash()

    ap = argparse.ArgumentParser(
        prog="normdb diffbind",
        description="Diff binding on BED regions (computeMatrix-based, DESeq2) + optional deepTools plotHeatmap (normalized only) + optional MACS2 peak calling consensus regions"
    )
    ap.add_argument("--meta", required=True, help="samples.tsv with columns: sample_id bam condition target")
    ap.add_argument("--regions", default=None, help="BED file of regions (chr start end). Required unless --call-peaks.")
    ap.add_argument("--call-peaks", action="store_true",
                    help="Call peaks with MACS2 (pooled per condition) and build consensus peaks. Only allowed if --regions is NOT set.")
    ap.add_argument("--bigwig-dir", required=True,
                    help="Directory containing normalized bigWigs (expects <sample_id>.norm99.bw)")
    ap.add_argument("--bigwig-dir-raw", default=None,
                    help="Optional directory containing RAW bigWigs to also generate DESeq2 PCA/correlation from RAW-derived counts (expects <sample_id>.bw). NOTE: deepTools heatmap/profile is NOT run on raw.")
    ap.add_argument("--contrast", required=True, help="e.g. condition:KO:WT")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for plots + signif tables")
    ap.add_argument("--lfc", type=float, default=0.0, help="log2FC threshold for plots + signif tables")
    ap.add_argument("--target", default=None, help="If meta contains multiple targets, specify one (e.g. H3K27me3)")
    ap.add_argument("--min-counts", type=int, default=100,
                    help="Filter out regions with low counts: keep rowSums(counts) >= this value (default 100)")

    ap.add_argument("--no-walk", action="store_true", help="Disable the GNOME walking animation")

    # ---- deepTools (normalized ONLY) ----
    ap.add_argument("--no-plotHeatmap", action="store_true",
                    help="Disable deepTools computeMatrix(reference-point) + plotHeatmap step (enabled by default). (Normalized bigWigs only)")
    ap.add_argument("--hm-referencePoint", default="center", choices=["TSS", "TES", "center"],
                    help="computeMatrix reference-point --referencePoint (default center)")
    ap.add_argument("--hm-before", type=int, default=5000, help="computeMatrix -b (bp upstream) (default 5000)")
    ap.add_argument("--hm-after", type=int, default=5000, help="computeMatrix -a (bp downstream) (default 5000)")
    ap.add_argument("--hm-threads", type=int, default=6, help="computeMatrix -p threads (default 6)")

    ap.add_argument("--hm-colorMap", default="bwr", help="plotHeatmap --colorMap (default bwr)")
    ap.add_argument("--hm-whatToShow", default="heatmap and colorbar",
                    help="plotHeatmap --whatToShow (default: 'heatmap and colorbar')")
    ap.add_argument("--hm-heatmapHeight", type=float, default=10.0, help="plotHeatmap --heatmapHeight (default 10)")
    ap.add_argument("--hm-heatmapWidth", type=float, default=2.0, help="plotHeatmap --heatmapWidth (default 2)")
    ap.add_argument("--hm-samplesLabel", nargs="+", default=None,
                    help="plotHeatmap --samplesLabel list (default: sample_id list from meta)")

    ap.add_argument("--no-plotProfile", action="store_true",
                    help="Disable deepTools plotProfile step (enabled by default if heatmap is enabled and signif regions exist).")
    ap.add_argument("--no-pp-perGroup", action="store_true",
                    help="Disable plotProfile --perGroup (enabled by default).")
    ap.add_argument("--pp-plotWidth", type=float, default=8.0,
                    help="plotProfile --plotWidth (default 8)")
    ap.add_argument("--pp-colors", nargs="+", default=None,
                    help="plotProfile --colors list. If not provided, auto uses black for ref and grey for others (repeated as needed).")

    # ---- MACS2 controls (only for --call-peaks) ----
    ap.add_argument("--macs2-format", default="AUTO",
                    help="MACS2 -f / --format (default AUTO; pass exactly as MACS2 expects)")
    ap.add_argument("--macs2-gsize", default="hs",
                    help="MACS2 -g / --gsize (default hs; can be mm or numeric like 2.7e9)")
    ap.add_argument("--macs2-mode", default="broad", choices=["narrow", "broad"],
                    help="MACS2 peak type (default broad)")
    ap.add_argument("--macs2-qvalue", type=float, default=0.005,
                    help="Q-value cutoff in probability space (default 0.005). Internally converted to MACS2 qscore threshold (-log10(q)).")
    ap.add_argument("--macs2-merge", type=int, default=100,
                    help="bedtools merge -d distance for consensus peaks (default 100 bp)")

    args = ap.parse_args()

    print_launch_summary(args)

    # Enforce modes
    if args.call_peaks and args.regions:
        die("Do not provide --regions when using --call-peaks (regions will be built from MACS2 consensus).")
    if (not args.call_peaks) and (not args.regions):
        die("You must provide --regions unless you use --call-peaks.")
    if args.regions and (not os.path.exists(args.regions)):
        die(f"--regions not found: {args.regions}")

    # Required executables
    ensure_exe("computeMatrix")
    ensure_exe("Rscript")
    if not args.no_plotHeatmap:
        ensure_exe("plotHeatmap")
        ensure_exe("plotProfile")
    if args.call_peaks:
        ensure_exe("macs2")
        ensure_exe("bedtools")

    if not os.path.isdir(args.bigwig_dir):
        die(f"--bigwig-dir not found or not a directory: {args.bigwig_dir}")
    if args.bigwig_dir_raw is not None and (not os.path.isdir(args.bigwig_dir_raw)):
        die(f"--bigwig-dir-raw not found or not a directory: {args.bigwig_dir_raw}")

    os.makedirs(args.outdir, exist_ok=True)

    regions_dir = os.path.join(args.outdir, "02_regions")
    os.makedirs(regions_dir, exist_ok=True)

    cm_dir = os.path.join(args.outdir, "03_computeMatrix")
    os.makedirs(cm_dir, exist_ok=True)

    dt_dir = os.path.join(args.outdir, "04_deeptools_heatmap")
    os.makedirs(dt_dir, exist_ok=True)

    log_path = os.path.join(args.outdir, "normdb_diffbind.log")

    walker = GnomeWalker(enabled=(not args.no_walk), fps=10, track_width=30)
    ctrl = DelayedWalkController(walker, delay_s=8.0)

    def _fail(msg: str):
        ctrl.end_step()
        die(msg)

    # ---- Step accounting (FIXED) ----
    step_plan = [
        "Reading metadata",
        "Collecting normalized bigWigs",
    ]
    if args.bigwig_dir_raw is not None:
        step_plan.append("Collecting RAW bigWigs (for DESeq2 PCA/corr only)")
    step_plan.append("Preparing regions (user BED or MACS2 consensus)")
    step_plan.append("computeMatrix per sample (normalized bigWigs)")
    step_plan.append("Building count matrix (normalized)")
    step_plan.append("DESeq2 + plots (normalized)")
    if args.bigwig_dir_raw is not None:
        step_plan.append("RAW bigWig DESeq2 PCA + correlation heatmap (optional)")
    if not args.no_plotHeatmap:
        step_plan.append("deepTools plotHeatmap + plotProfile (normalized only)")

    step_total = len(step_plan)
    step_num = 0

    # will be used for final summary
    total_regions_used = None
    n_gain = 0
    n_loss = 0
    n_sig = 0

    regions_path = args.regions
    regions_mode = "User-provided BED (--regions)"
    target = None
    contrast_var = None
    contrast_test = None
    contrast_ref = None

    bw_paths_raw, bw_labels_raw = None, None
    macs2_summary = None

    try:
        with open(log_path, "w") as log_fh:
            log_fh.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
            log_fh.write(f"Args: {vars(args)}\n")

            # -------------------------
            # Step 1: Load + filter meta
            # -------------------------
            step_num += 1
            ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
            step = time.time()
            log("Reading metadata", ctrl=ctrl)

            meta = read_meta(args.meta)

            targets = sorted(meta["target"].unique().tolist())
            if args.target is None:
                if len(targets) != 1:
                    _fail(f"Meta contains multiple targets {targets}. Please provide --target <one>.")
                target = targets[0]
            else:
                if args.target not in targets:
                    _fail(f"--target {args.target} not found in meta targets: {targets}")
                target = args.target

            meta_t = meta.loc[meta["target"] == target].copy()
            if meta_t.shape[0] < 2:
                _fail(f"Not enough samples for target={target} (need >=2).")

            contrast_var, contrast_test, contrast_ref = parse_contrast(args.contrast)
            if contrast_var not in meta_t.columns:
                _fail(f"Contrast variable '{contrast_var}' not found in meta columns: {list(meta_t.columns)}")

            levels = sorted(set(meta_t[contrast_var].tolist()))
            if contrast_test not in levels or contrast_ref not in levels:
                _fail(f"Meta does not contain both contrast levels: {contrast_test}, {contrast_ref}. Found: {levels}")

            log(f"Using target={target} with n={meta_t.shape[0]} samples; contrast={contrast_var}:{contrast_test}:{contrast_ref}", ctrl=ctrl)
            ctrl.end_step()

            # -------------------------
            # Step 2: Collect normalized bigWigs
            # -------------------------
            step_num += 1
            ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
            step = time.time()

            bw_paths, bw_labels = collect_bigwigs(meta_t, args.bigwig_dir, suffix=".norm99.bw", ctrl=ctrl)

            log_done(f"Found {len(bw_paths)} normalized bigWigs", ctrl=ctrl)
            ctrl.end_step()

            # -------------------------
            # Optional Step: Collect RAW bigWigs (ONLY for DESeq2 PCA/corr)
            # -------------------------
            if args.bigwig_dir_raw is not None:
                step_num += 1
                ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
                step = time.time()

                bw_paths_raw, bw_labels_raw = collect_bigwigs(meta_t, args.bigwig_dir_raw, suffix=".bw", ctrl=ctrl)

                log_done(f"Found {len(bw_paths_raw)} RAW bigWigs", ctrl=ctrl)
                ctrl.end_step()

            # -------------------------
            # Step: Prepare regions (always a numbered step now)
            # -------------------------
            step_num += 1
            ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
            step = time.time()

            if args.call_peaks:
                log("Calling MACS2 peaks pooled per condition and building consensus BED", ctrl=ctrl)

                qscore_thr = qvalue_to_macs2_qscore(args.macs2_qvalue)
                macs2_summary = {
                    "format (-f)": args.macs2_format,
                    "gsize (-g)": args.macs2_gsize,
                    "mode": args.macs2_mode,
                    "qvalue (prob)": args.macs2_qvalue,
                    "qscore threshold (-log10(q))": f"{qscore_thr:.6f}",
                    "merge distance (bedtools -d)": args.macs2_merge,
                    "pooling": f"All replicates pooled per {contrast_var} level (no control, --nomodel, --keep-dup auto)",
                }

                consensus_bed = os.path.join(regions_dir, "consensus_peaks.bed")
                peak_bed_by_level = {}

                for lvl in levels:
                    ctrl.set_label(f"Step {step_num}/{step_total} — MACS2: {lvl}")
                    lvl_slug = _safe_slug(lvl)

                    bams = meta_t.loc[meta_t[contrast_var] == lvl, "bam"].tolist()
                    if len(bams) == 0:
                        continue

                    lvl_outdir = os.path.join(regions_dir, f"macs2_{lvl_slug}")
                    os.makedirs(lvl_outdir, exist_ok=True)

                    name = f"{lvl_slug}_{_safe_slug(target)}_pool"

                    cmd = ["macs2", "callpeak", "-t", *bams,
                           "-f", args.macs2_format,
                           "--keep-dup", "auto",
                           "--nomodel",
                           "-g", str(args.macs2_gsize),
                           "--outdir", lvl_outdir,
                           "-n", name]
                    if args.macs2_mode == "broad":
                        cmd.append("--broad")

                    run_cmd(cmd, log_fh=log_fh)

                    peakfile = os.path.join(
                        lvl_outdir,
                        f"{name}_peaks.broadPeak" if args.macs2_mode == "broad" else f"{name}_peaks.narrowPeak"
                    )
                    if not os.path.exists(peakfile):
                        _fail(f"MACS2 did not produce expected peak file: {peakfile}")

                    lvl_filtered_bed = os.path.join(regions_dir, f"peaks_{lvl_slug}.bed")
                    n_kept = filter_macs2_peakfile_by_qscore(peakfile, lvl_filtered_bed, qscore_thr)
                    if n_kept == 0:
                        log(f"WARNING: no peaks passed qscore >= {qscore_thr:.6f} for level={lvl}", ctrl=ctrl)

                    peak_bed_by_level[lvl] = lvl_filtered_bed

                if not peak_bed_by_level:
                    _fail("No peak BEDs produced by MACS2. Check meta/contrast grouping and BAM inputs.")

                concat_path = os.path.join(regions_dir, "all_conditions_peaks.concat.bed")
                with open(concat_path, "w") as fout:
                    for bedp in peak_bed_by_level.values():
                        if os.path.exists(bedp) and os.path.getsize(bedp) > 0:
                            with open(bedp) as fin:
                                shutil.copyfileobj(fin, fout)

                sorted_path = os.path.join(regions_dir, "all_conditions_peaks.sorted.bed")
                n_rows = sort_bed(concat_path, sorted_path)
                if n_rows == 0:
                    _fail("After filtering, no peaks remained across all conditions; cannot build consensus.")

                merge_cmd = ["bedtools", "merge", "-d", str(args.macs2_merge), "-i", sorted_path]
                merged = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if merged.returncode != 0:
                    _fail(f"bedtools merge failed:\n{merged.stderr}")

                with open(consensus_bed, "w") as f:
                    f.write(merged.stdout)

                if os.path.getsize(consensus_bed) == 0:
                    _fail("Consensus peaks BED is empty after merge; cannot proceed.")

                regions_path = consensus_bed
                regions_mode = "MACS2 pooled-per-condition peaks → qscore filter → consensus merge (--call-peaks)"
                log_done(f"Consensus peaks written: {consensus_bed}", step, ctrl=ctrl)

            else:
                log("Validating user-provided BED regions", ctrl=ctrl)
                # Keep regions_path as provided; optionally copy to outdir for provenance
                src = args.regions
                dst = os.path.join(regions_dir, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copyfile(src, dst)
                regions_path = dst
                regions_mode = "User-provided BED (--regions)"
                log_done(f"Using regions BED: {regions_path}", step, ctrl=ctrl)

            ctrl.end_step()

            total_regions_used = count_bed_rows(regions_path)


            # -------------------------
            # Step: computeMatrix per sample (normalized bigWigs)
            # -------------------------
            step_num += 1
            ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
            step = time.time()
            log("computeMatrix: counting signal in regions (scale-regions; sum)", ctrl=ctrl)

            matrix_txts, sorted_beds = {}, {}
            for sid, bw in zip(bw_labels, bw_paths):
                ctrl.set_label(f"Step {step_num}/{step_total} — computeMatrix: {sid}")

                base = f"{sid}-{target}"
                out_gz = os.path.join(cm_dir, f"{base}.gz")
                out_txt = os.path.join(cm_dir, f"{base}.txt")
                out_bed = os.path.join(cm_dir, f"{base}.bed")

                cmd = [
                    "computeMatrix", "scale-regions",
                    "-S", bw,
                    "-R", regions_path,
                    "--outFileName", out_gz,
                    "--outFileNameMatrix", out_txt,
                    "--outFileSortedRegions", out_bed,
                    "--missingDataAsZero",
                    "--averageTypeBins", "sum",
                    "--binSize", "100",
                    "--regionBodyLength", "100",
                ]
                run_cmd(cmd, log_fh=log_fh)

                matrix_txts[sid] = out_txt
                sorted_beds[sid] = out_bed

            log_done("computeMatrix finished for all samples", step, ctrl=ctrl)
            ctrl.end_step()

            # -------------------------
            # Step: Build count matrix (normalized)
            # -------------------------
            step_num += 1
            ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
            step = time.time()
            log("Building region count matrix from computeMatrix outputs", ctrl=ctrl)

            first_sid = bw_labels[0]
            region_ids = bed_to_region_ids(sorted_beds[first_sid])
            n_regions = len(region_ids)

            counts = pd.DataFrame({"region_id": region_ids})
            for sid in bw_labels:
                vec = read_computeMatrix_vector(matrix_txts[sid])
                if len(vec) != n_regions:
                    _fail(
                        f"Region count mismatch for {sid}: "
                        f"{len(vec)} values in matrix but {n_regions} regions in sorted BED."
                    )
                counts[sid] = pd.Series(vec).round().astype(int)

            counts_path = os.path.join(args.outdir, "counts_matrix.tsv")
            counts.to_csv(counts_path, sep="\t", index=False)

            meta_t2 = meta_t.set_index("sample_id").loc[bw_labels].reset_index()
            coldata_path = os.path.join(args.outdir, "coldata.tsv")
            meta_t2[["sample_id", contrast_var]].to_csv(coldata_path, sep="\t", index=False)

            ctrl.end_step()

            # -------------------------
            # Step: DESeq2 + plots (normalized)
            # -------------------------
            step_num += 1
            ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
            step = time.time()
            log("Running DESeq2 + plots", ctrl=ctrl)

            results_path = os.path.join(args.outdir, "results_all_regions.tsv")
            gain_path = os.path.join(args.outdir, "results_signif_gain.tsv")
            loss_path = os.path.join(args.outdir, "results_signif_loss.tsv")
            volcano_pdf = os.path.join(args.outdir, "volcano.pdf")
            ma_pdf = os.path.join(args.outdir, "MA.pdf")
            corr_pdf = os.path.join(args.outdir, "sample_correlation_heatmap.pdf")
            pca_pdf = os.path.join(args.outdir, "PCA_vst.pdf")
            r_script_path = os.path.join(args.outdir, "run_deseq2.R")

            r_script = f"""
suppressPackageStartupMessages({{
  library(DESeq2)
  library(tidyverse)
  library(ggplot2)
  library(EnhancedVolcano)
}})

alpha <- {args.alpha}
lfc_thr <- {args.lfc}
min_counts <- {args.min_counts}

counts_file <- "{counts_path}"
coldata_file <- "{coldata_path}"

out_results <- "{results_path}"
out_gain <- "{gain_path}"
out_loss <- "{loss_path}"
out_volcano <- "{volcano_pdf}"
out_ma <- "{ma_pdf}"
out_corr <- "{corr_pdf}"
out_pca <- "{pca_pdf}"

contrast_var <- "{contrast_var}"
test_level <- "{contrast_test}"
ref_level <- "{contrast_ref}"

counts_df <- read.delim(counts_file, check.names=FALSE)
stopifnot("region_id" %in% colnames(counts_df))

count_mat <- counts_df %>% select(-region_id) %>% as.matrix()
rownames(count_mat) <- counts_df$region_id
mode(count_mat) <- "integer"

coldata <- read.delim(coldata_file, check.names=FALSE)
stopifnot("sample_id" %in% colnames(coldata))
stopifnot(contrast_var %in% colnames(coldata))
stopifnot(all(coldata$sample_id == colnames(count_mat)))

dds <- DESeqDataSetFromMatrix(
  countData = count_mat,
  colData = as.data.frame(coldata %>% column_to_rownames("sample_id")),
  design = as.formula(paste0("~ ", contrast_var))
)

keep <- rowSums(counts(dds)) >= min_counts
dds <- dds[keep,]

dds[[contrast_var]] <- relevel(dds[[contrast_var]], ref = ref_level)
dds <- DESeq(dds)

coef_name <- resultsNames(dds)[2]
use_apeglm <- requireNamespace("apeglm", quietly=TRUE)

if (use_apeglm) {{
  res <- lfcShrink(dds, coef=coef_name, type="apeglm")
}} else {{
  res <- lfcShrink(dds, coef=coef_name, type="normal")
}}

res_df <- as.data.frame(res) %>%
  rownames_to_column("region_id") %>%
  separate(region_id, into=c("chr","start","end"), sep="_", remove=FALSE, convert=TRUE) %>%
  relocate(chr, start, end, region_id)

write.table(res_df, file=out_results, sep="\\t", quote=FALSE, row.names=FALSE)

res_sig <- res_df %>%
  filter(!is.na(padj)) %>%
  filter(padj < alpha, abs(log2FoldChange) >= lfc_thr) %>%
  mutate(direction = ifelse(log2FoldChange > 0, "Gain", "Loss"))

write.table(res_sig %>% filter(direction=="Gain"), file=out_gain, sep="\\t", quote=FALSE, row.names=FALSE)
write.table(res_sig %>% filter(direction=="Loss"), file=out_loss, sep="\\t", quote=FALSE, row.names=FALSE)

res_plot <- res_df %>%
  mutate(
    padj_plot = ifelse(is.na(padj), 1, padj),
    direction = case_when(
      padj_plot < alpha & log2FoldChange >=  lfc_thr ~ "Gain",
      padj_plot < alpha & log2FoldChange <= -lfc_thr ~ "Loss",
      TRUE ~ "NS"
    )
  )

col_map <- c("NS"="grey70", "Gain"="orange", "Loss"="skyblue")

pdf(out_volcano, width=4.8, height=4.1)
EnhancedVolcano(
  res_plot,
  lab = rep("", nrow(res_plot)),
  x = "log2FoldChange",
  y = "padj_plot",
  pCutoff = alpha,
  FCcutoff = lfc_thr,
  colCustom = col_map[res_plot$direction],
  colAlpha = 0.75,
  pointSize = 1.1,
  labSize = 2.0,
  title = paste0(test_level, " vs ", ref_level),
  subtitle = NULL,
  legendPosition = "right",
  legendLabSize = 11,
  legendIconSize = 3.5
) +
  theme_bw() +
  guides(colour = guide_legend(title = NULL), fill = guide_legend(title = NULL))
dev.off()

ma_df <- as.data.frame(res) %>%
  rownames_to_column("region_id") %>%
  mutate(
    padj_plot = ifelse(is.na(padj), 1, padj),
    direction = case_when(
      padj_plot < alpha & log2FoldChange >=  lfc_thr ~ "Gain",
      padj_plot < alpha & log2FoldChange <= -lfc_thr ~ "Loss",
      TRUE ~ "NS"
    )
  )

p_ma <- ggplot(ma_df, aes(x = baseMean, y = log2FoldChange, color = direction)) +
  geom_point(size = 1.0, alpha = 0.75) +
  scale_x_log10() +
  geom_hline(yintercept = c(-lfc_thr, lfc_thr), linetype = "dashed") +
  theme_bw() +
  labs(
    title = paste0("MA: ", test_level, " vs ", ref_level),
    x = "Mean of normalized counts (log10)",
    y = "Log2 fold change"
  ) +
  scale_color_manual(values = col_map)

ggsave(out_ma, plot = p_ma, width = 5.0, height = 4.2)

vsd <- vst(dds, blind=FALSE)
mat <- assay(vsd)

pca <- prcomp(t(mat), scale.=FALSE)
pca_df <- as.data.frame(pca$x[, 1:2, drop=FALSE]) %>%
  rownames_to_column("sample_id") %>%
  left_join(coldata, by="sample_id")

conds <- unique(pca_df[[contrast_var]])
ref_condition <- ref_level

grey_palette <- c("grey", "grey50", "grey65", "grey35", "grey80", "grey20")
cond_to_color <- setNames(rep("grey", length(conds)), conds)
cond_to_color[ref_condition] <- "black"

others <- setdiff(conds, ref_condition)
for (i in seq_along(others)) {{
  cond_to_color[others[i]] <- grey_palette[min(i, length(grey_palette))]
}}

p_pca <- ggplot(pca_df, aes(x=PC1, y=PC2, color=.data[[contrast_var]], label=sample_id)) +
  geom_point(size=3) +
  geom_text(vjust=-0.8, size=3) +
  scale_color_manual(values=cond_to_color) +
  theme_bw() +
  labs(
    title = paste0("PCA (vst) — ref=", ref_condition),
    x = paste0("PC1 (", round(100*summary(pca)$importance[2,1], 1), "%)"),
    y = paste0("PC2 (", round(100*summary(pca)$importance[2,2], 1), "%)")
  ) +
  guides(color=guide_legend(title=contrast_var))

ggsave(out_pca, plot=p_pca, width=7.5, height=6.5)

cormat <- cor(mat, method="pearson")
pdf(out_corr, width=8.5, height=8.5)
heatmap(cormat, symm=TRUE, margins=c(12,12), main="Sample correlation (vst counts)")
dev.off()
"""
            with open(r_script_path, "w") as f:
                f.write(r_script)

            run_cmd(["Rscript", r_script_path], log_fh=log_fh)
            log_done("DESeq2 + plots finished", step, ctrl=ctrl)
            ctrl.end_step()

            # counts of significant regions for summary
            try:
                n_gain = pd.read_csv(gain_path, sep="\t").shape[0] if os.path.exists(gain_path) else 0
                n_loss = pd.read_csv(loss_path, sep="\t").shape[0] if os.path.exists(loss_path) else 0
            except Exception:
                n_gain, n_loss = 0, 0
            n_sig = int(n_gain) + int(n_loss)

            # -------------------------
            # Optional Step: RAW bigWig PCA + correlation heatmap (DESeq2-style)
            # IMPORTANT: no deepTools on raw (per your request)
            # -------------------------
            if args.bigwig_dir_raw is not None:
                step_num += 1
                ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
                step = time.time()
                log("Generating PCA + correlation heatmap from RAW bigWig-derived counts (DESeq2 VST)", ctrl=ctrl)

                cm_raw_dir = os.path.join(args.outdir, "03b_computeMatrix_RAW")
                os.makedirs(cm_raw_dir, exist_ok=True)

                matrix_txts_raw, sorted_beds_raw = {}, {}
                for sid, bw in zip(bw_labels_raw, bw_paths_raw):
                    ctrl.set_label(f"Step {step_num}/{step_total} — RAW computeMatrix: {sid}")

                    base = f"{sid}-{target}-RAW"
                    out_gz = os.path.join(cm_raw_dir, f"{base}.gz")
                    out_txt = os.path.join(cm_raw_dir, f"{base}.txt")
                    out_bed = os.path.join(cm_raw_dir, f"{base}.bed")

                    cmd = [
                        "computeMatrix", "scale-regions",
                        "-S", bw,
                        "-R", regions_path,
                        "--outFileName", out_gz,
                        "--outFileNameMatrix", out_txt,
                        "--outFileSortedRegions", out_bed,
                        "--missingDataAsZero",
                        "--averageTypeBins", "sum",
                        "--binSize", "100",
                        "--regionBodyLength", "100",
                    ]
                    run_cmd(cmd, log_fh=log_fh)

                    matrix_txts_raw[sid] = out_txt
                    sorted_beds_raw[sid] = out_bed

                first_sid_raw = bw_labels_raw[0]
                region_ids_raw = bed_to_region_ids(sorted_beds_raw[first_sid_raw])
                n_regions_raw = len(region_ids_raw)

                counts_raw = pd.DataFrame({"region_id": region_ids_raw})
                for sid in bw_labels_raw:
                    vec = read_computeMatrix_vector(matrix_txts_raw[sid])
                    if len(vec) != n_regions_raw:
                        _fail(
                            f"RAW region count mismatch for {sid}: "
                            f"{len(vec)} values in matrix but {n_regions_raw} regions in sorted BED."
                        )
                    counts_raw[sid] = pd.Series(vec).round().astype(int)

                counts_raw_path = os.path.join(args.outdir, "counts_matrix_RAW_bigwig.tsv")
                counts_raw.to_csv(counts_raw_path, sep="\t", index=False)

                pca_raw_pdf = os.path.join(args.outdir, "PCA_vst_RAW_bigwig.pdf")
                corr_raw_pdf = os.path.join(args.outdir, "sample_correlation_heatmap_RAW_bigwig.pdf")

                run_raw_pca_corr_from_counts(
                    counts_path=counts_raw_path,
                    coldata_path=coldata_path,
                    contrast_var=contrast_var,
                    test_level=contrast_test,
                    ref_level=contrast_ref,
                    out_pca=pca_raw_pdf,
                    out_corr=corr_raw_pdf,
                    outdir=args.outdir,
                    log_fh=log_fh,
                )

                log_done("RAW bigWig PCA + correlation written", step, ctrl=ctrl)
                ctrl.end_step()

            # -------------------------
            # Optional Step: deepTools heatmap + profile (normalized only)
            # (NO raw deepTools generation, even if --bigwig-dir-raw is provided)
            # -------------------------
            if not args.no_plotHeatmap:
                step_num += 1
                ctrl.begin_step(f"Step {step_num}/{step_total} — {step_plan[step_num-1]}")
                step = time.time()
                log("Preparing gain/loss BEDs + running deepTools on NORMALIZED bigWigs", ctrl=ctrl)

                gain_bed = os.path.join(dt_dir, "signif_gain.bed")
                loss_bed = os.path.join(dt_dir, "signif_loss.bed")

                n_gain_bed = write_bed3_from_results(gain_path, gain_bed)
                n_loss_bed = write_bed3_from_results(loss_path, loss_bed)

                if n_gain_bed == 0 and n_loss_bed == 0:
                    log("No significant gain/loss regions found; skipping plotHeatmap/plotProfile.", ctrl=ctrl)
                    log_done("deepTools heatmap/profile skipped (no signif regions)", step, ctrl=ctrl)
                    ctrl.end_step()
                else:
                    matrix_gz = os.path.join(
                        dt_dir,
                        f"matrix_refpoint_{args.hm_referencePoint}_b{args.hm_before}_a{args.hm_after}.gz"
                    )
                    heatmap_pdf = os.path.join(
                        dt_dir,
                        f"heatmap_refpoint_{args.hm_referencePoint}_b{args.hm_before}_a{args.hm_after}.pdf"
                    )
                    profile_pdf = os.path.join(
                        dt_dir,
                        f"profile_refpoint_{args.hm_referencePoint}_b{args.hm_before}_a{args.hm_after}.pdf"
                    )

                    region_files = []
                    region_labels = []
                    if n_gain_bed > 0:
                        region_files.append(gain_bed); region_labels.append("Gain")
                    if n_loss_bed > 0:
                        region_files.append(loss_bed); region_labels.append("Loss")

                    samples_label = args.hm_samplesLabel if args.hm_samplesLabel else bw_labels
                    if len(samples_label) != len(bw_paths):
                        _fail(f"--hm-samplesLabel provided {len(samples_label)} labels but there are {len(bw_paths)} samples.")

                    cm_cmd = [
                        "computeMatrix", "reference-point",
                        "--referencePoint", args.hm_referencePoint,
                        "-b", str(args.hm_before),
                        "-a", str(args.hm_after),
                        "-R", *region_files,
                        "-S", *bw_paths,
                        "--missingDataAsZero",
                        "-o", matrix_gz,
                        "-p", str(args.hm_threads),
                    ]
                    run_cmd(cm_cmd, log_fh=log_fh)

                    ph_cmd = [
                        "plotHeatmap",
                        "-m", matrix_gz,
                        "-out", heatmap_pdf,
                        "--samplesLabel", *samples_label,
                        "--colorMap", args.hm_colorMap,
                        "--whatToShow", args.hm_whatToShow,
                        "--heatmapHeight", str(args.hm_heatmapHeight),
                        "--heatmapWidth", str(args.hm_heatmapWidth),
                    ]
                    if len(region_labels) > 0:
                        ph_cmd += ["--regionsLabel", *region_labels]
                    run_cmd(ph_cmd, log_fh=log_fh)

                    if not args.no_plotProfile:
                        if args.pp_colors:
                            colors = args.pp_colors
                        else:
                            cond_by_sid = meta_t.set_index("sample_id")[contrast_var].to_dict()
                            colors = [("black" if cond_by_sid.get(sid, "") == contrast_ref else "grey") for sid in bw_labels]
                        if len(colors) != len(bw_paths):
                            if len(colors) < len(bw_paths) and len(colors) > 0:
                                colors = colors + [colors[-1]] * (len(bw_paths) - len(colors))
                            colors = colors[:len(bw_paths)]

                        pp_cmd = [
                            "plotProfile",
                            "-m", matrix_gz,
                            "-out", profile_pdf,
                            "--samplesLabel", *samples_label,
                            "--colors", *colors,
                            "--plotWidth", str(args.pp_plotWidth),
                        ]
                        if not args.no_pp_perGroup:
                            pp_cmd.append("--perGroup")
                        run_cmd(pp_cmd, log_fh=log_fh)

                    log_done(f"deepTools outputs written in: {dt_dir}", step, ctrl=ctrl)
                    ctrl.end_step()

        log(f"All done. Total runtime: {(time.time()-t0)/60:.1f} minutes", ctrl=None)

    finally:
        ctrl.end_step()

    print(f"Outputs in: {args.outdir}")
    print(f"Log: {log_path}")

    summary = []
    summary.append("")
    summary.append("")
    summary.append("GNOMES ran successfully ✅")
    summary.append("")
    summary.append(f"Target: {target}")
    summary.append(f"Contrast: {contrast_var}:{contrast_test}:{contrast_ref}")
    summary.append(f"Regions analyzed: {total_regions_used if total_regions_used is not None else 'NA'}")
    summary.append(f"Significant regions (padj<{args.alpha}, |LFC|>={args.lfc}): {n_sig}")
    summary.append(f"  - Gain: {n_gain}")
    summary.append(f"  - Loss: {n_loss}")
    summary.append("")
    summary.append("")
    summary.append("")
    summary.append("If you use GNOMES in your work, please cite: Roule et al., [YEAR], [JOURNAL].")

    print_final_gnome_summary(summary)


if __name__ == "__main__":
    main()
