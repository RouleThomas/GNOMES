import argparse
import os
import sys
import subprocess

def _run(script_path: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, script_path] + extra_args
    return subprocess.call(cmd)

def main():
    ap = argparse.ArgumentParser(
        prog="GNOMES",
        description="GNOMES: normalization + differential binding for epigenomic tracks"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_norm = sub.add_parser("norm", help="Normalize BAMs → normalized bigWigs + QC")
    p_norm.add_argument("args", nargs=argparse.REMAINDER)

    p_diff = sub.add_parser("diff", help="Differential binding using normalized bigWigs + DESeq2")
    p_diff.add_argument("args", nargs=argparse.REMAINDER)

    args = ap.parse_args()

    # repo root = one level above this file's folder
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    norm_script = os.path.join(repo_root, "scripts", "gnomes_norm.py")
    diff_script = os.path.join(repo_root, "scripts", "gnomes_diff.py")

    if not os.path.exists(norm_script):
        raise FileNotFoundError(f"Missing script: {norm_script}")
    if not os.path.exists(diff_script):
        raise FileNotFoundError(f"Missing script: {diff_script}")

    if args.cmd == "norm":
        # your normalization script has a subcommand named "normalize"
        # so we inject it transparently for the user
        sys.exit(_run(norm_script, ["normalize"] + args.args))

    if args.cmd == "diff":
        sys.exit(_run(diff_script, args.args))
