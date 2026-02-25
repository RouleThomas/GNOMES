<p align="center">
  <img src="assets/GNOMES_logo1.png" alt="GNOMES logo" width="850">
</p>

<h1 align="center">GNOMES</h1>

<p align="center">
  <b>Genome-wide NOrmalization of Mapped Epigenomic Signals</b><br>
  Normalization + differential binding for Cut&Run / ChIP-seq, with reproducible QC and publication-ready outputs.
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#workflow-and-methodology">Workflow and methodology</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#command-examples">Command examples</a> •
  <a href="#inputs">Inputs</a> •
  <a href="#outputs">Outputs</a> •
  <a href="#contributing">Troubleshooting</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**GNOMES** is a user-friendly framework to:

1. **Normalize** epigenomic signal tracks from BAM files using a robust percentile-based scaling strategy.
2. **Perform differential binding analysis** over user-defined regions, or automatically generated peaks, using DESeq2 with built-in quality control.

GNOMES is designed for **Cut&Run** and **ChIP-seq** (Single-End or Paired-End), and supports both histone marks and transcription factors.

---

## Workflow and methodology

GNOMES is a **two-step method**; signal normalization followed by differential binding analysis:

### **Step 1 — Normalization (`GNOMES norm`)**

**Goal**: Generate scaled bigWig tracks across samples.

**Pipeline**:
1. Convert BAM → raw bigWig → bedGraph (*optional* blacklist filtering + control(input/IGG) subtraction)
2. Identify local maxima in each bedGraph
3. Compute the 99th percentile (P99) of local signal maxima per sample
4. Within each target, select a reference sample and compute scaling factor (SF): `SF(sample) = P99(reference_sample) / P99(sample)`
5. Apply SF to generate normalized signal tracks

**Outputs**:
- normalized bigWigs (`*.norm99.bw`)
- scaling factors table (`scaling_factors.tsv`)
- median tracks per (condition, target)
- QC plots (PCA + correlation heatmap) for **raw** and **normalized** bigWigs

This percentile-based normalization is designed to stabilize signal distributions without assuming equal total occupancy across samples.


### **Step 2 — Differential binding (`GNOMES diff`)**

**Goal**: Identify regions with significant binding changes (ie. gained or lost) between biological conditions.

GNOMES supports two strategies:
- User-provided BED regions (`--regions`)
- Automatically generated MACS2 consensus peaks (`--call-peaks`)

  
**Pipeline**:
1. Quantify per-region signal from normalized bigWigs
2. Convert signal into count-like matrices
3. Perform differential analysis using DESeq2 or edgeR

**Outputs**:
- volcano + MA plot
- PCA + sample correlation heatmap (from DESeq2 VST)
- significant gain/loss region tables (TSV)
- **optional** deepTools heatmap + profile plot over significant regions
- **optional** PCA + sample correlation heatmap (from DESeq2 VST) for RAW bigWig files
---


## Installation

### Option 1 — Conda (recommended)

```bash
# Clone the repo
git clone https://github.com/RouleThomas/GNOMES.git
cd GNOMES

# Installation
## Conda
conda env create -f env/GNOMES-environment.yml
## Mamba
mamba env create -f env/GNOMES-environment.yml

conda activate GNOMES


# Install dependencies
pip install -e .

# Test
GNOMES norm --help
GNOMES diff --help
```




### Option 2 — Apptainer / Singularity

***--> Soon available***




## Quick start

GNOMES is a two-step pipeline: **normalize** first, then **diff**.

### **Step 1 — Normalize**

```
GNOMES norm \
  --meta <metadata.tsv> \
  --outdir <output_directory> \
  --chrom-sizes <chrom_sizes.txt> \
  --mode <SE|PE>
```


### **Step 2 — Differential binding (using your own BED regions)**

```
GNOMES diff \
  --regions <regions.bed> \
  --meta <metadata.tsv> \
  --bigwig-dir <normalized_bigwig_directory> \
  --contrast <column:group1:group2> \
  --target <mark_or_TF> \
  --outdir <diff_output_directory>
```

### **Step 2 — Differential binding (MACS2 consensus peaks)**

```
GNOMES diff \
  --call-peaks \
  --meta <metadata.tsv> \
  --bigwig-dir <normalized_bigwig_directory> \
  --contrast <column:group1:group2> \
  --target <mark_or_TF> \
  --outdir <diff_output_directory>
```


## Command examples

**Example 1 — Single-End (with blacklist) and Differential binding using user BED regions with DESEQ2**

```bash
GNOMES norm \
  --meta meta/samples.tsv \
  --outdir output/gnomes_run \
  --blacklist meta/hg38-blacklist.v2.bed \
  --chrom-sizes meta/GRCh38_chrom_sizes.tab \
  --threads 8 \
  --mode SE \
  --se-fragment-length 200

GNOMES diff \
  --meta meta/samples.tsv \
  --regions regions/promoters.bed \
  --bigwig-dir output/gnomes_run/06_normalized_bigwig \
  --contrast condition:KO:WT \
  --target H3K27me3 \
  --outdir output/gnomes_run_diff \
  --diff-method deseq2 \
  --deseq2-alpha 0.05 \
  --deseq2-lfc 0.58 \
  --deseq2-min-counts 100 \
  --deseq2-sizefactors auto
```

**Example 2 — Paired-End (without blacklist) and Differential binding using MACS2 consensus peaks with edgeR**

```bash
GNOMES norm \
  --meta meta/samples.tsv \
  --outdir output/gnomes_run \
  --chrom-sizes meta/GRCh38_chrom_sizes.tab \
  --threads 8 \
  --mode PE

GNOMES diff \
  --meta meta/samples.tsv \
  --call-peaks \
  --bigwig-dir output/gnomes_run/06_normalized_bigwig \
  --contrast condition:KO:WT \
  --target H3K27me3 \
  --outdir output/gnomes_run_diff \
  --alpha 0.05 \
  --lfc 0.5 \
  --macs2-mode broad \
  --macs2-qvalue 0.005 \
  --macs2-merge 100 \
  --diff-method edger \
  --edger-alpha 0.05 \
  --edger-lfc 0.58 \
  --edger-min-counts 100 \
  --edger-norm TMM \
```



## Inputs

### Metadata file (`--meta`)

Tab-separated file with required columns:

| column       | description                                    |
| ------------ | ---------------------------------------------- |
| `sample_id`  | unique sample name (used for output filenames) |
| `bam`        | path to BAM file                               |
| `condition`  | condition label (ie. WT, KO)                   |
| `target`     | mark/TF name (ie. H3K27me3, EZH2)              |
| **OPTIONAL*** `bam_control` | path to BAM control (input/IGG) file           |

**When provided, control sample is subtracted from corresponding IP sample*

Example:
```
sample_id	bam	condition	target  bam_control
WT_1	/path/WT_H3K27me3_1.bam	WT	H3K27me3  /path/WT_input_1.bam
WT_2	/path/WT_H3K27me3_2.bam	WT	H3K27me3  /path/WT_input_2.bam
KO_1	/path/KO_H3K27me3_1.bam	KO	H3K27me3  /path/KO_input_1.bam
KO_2	/path/KO_H3K27me3_2.bam	KO	H3K27me3  /path/KO_input_2.bam
```



### Regions BED (`--regions`, diff step only)

Standard BED3:
```
chr  start  end
```

If you use `--call-peaks`, GNOMES builds regions automatically from MACS2 pooled-per-condition peaks.


## Outputs

### Step 1 (Normalization) output structure

**`--outdir` contains**:
- `01_raw_bigwig/sample_id.bw` (raw)
- **OPTIONAL** `02_bedgraph/sample_id.bedGraph`
- **OPTIONAL** `03_bedgraph_blacklist/`
blacklist-filtered bedGraph (or identical copy if no blacklist)
- **OPTIONAL** `04_local_maxima/sample_id.local_maxima.bed`
- **OPTIONAL** `05_normalized_bedgraph/sample_id.norm99.bedGraph` (+ sorted)
- `06_normalized_bigwig/sample_id.norm99.bw`
- **OPTIONAL** `07_median_bedgraph/`
- `08_median_bigwig/`
median tracks per (condition, target)
- `09_qc/`
PCA + correlation heatmap for raw and normalized bigWigs (per target)
- `scaling_factors.tsv`
P99 and scaling factor per sample
- `normdb_normalize.log`
full command log (including all tool calls)


### Step 2 (Differential binding) output structure

**`--outdir` contains**:
- `02_regions/`
copied user BED, or MACS2 consensus peaks BED
- `03_computeMatrix/`
per-sample computeMatrix outputs for normalized bigWigs
- `04_deeptools_heatmap/` (optional)
heatmap/profile over significant gain/loss regions (normalized only)
- `counts_matrix.tsv`
per-region counts from normalized bigWigs
- `coldata.tsv`
sample_id + condition column used in DESeq2
- `results_all_regions.tsv`
complete DESeq2 results table
- `results_signif_gain.tsv` / `results_signif_loss.tsv`
significant regions split by direction
- `volcano.pdf`, `MA.pdf`, `PCA_vst.pdf`, `sample_correlation_heatmap.pdf`
- Optional raw-derived:
    - `counts_matrix_RAW_bigwig.tsv`
    - `PCA_vst_RAW_bigwig.pdf`
    - `sample_correlation_heatmap_RAW_bigwig.pdf`
- `normdb_diffbind.log`
full command log + exact tool calls for reproducibility




## Contributing

Contributions are welcome:
- bug reports and feature requests via GitHub Issues


Please include:
- command used
- log file


## Citation

If you use GNOMES in your work, please cite:

***Roule T. et al. GNOMES: Genome-wide NOrmalization of Mapped Epigenomic Signals. [Journal] (YEAR).***

(Preprint / DOI coming soon.)




