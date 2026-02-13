#!/usr/bin/env python3
import sys


HELP = """\
GNOMES — Genome-wide NOrmalization of Mapped Epigenomic Signals

Usage:
  GNOMES norm [options]    Normalize BAM → bigWigs (P99 scaling)
  GNOMES diff [options]    Differential binding analysis (DESeq2)

Run:
  GNOMES <command> --help
to see all options for a command.
"""


def main():
    argv = sys.argv[1:]

    # Global help
    if len(argv) == 0 or argv[0] in ("-h", "--help"):
        print(HELP)
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "norm":
        # gnomes_norm.py currently expects: normdb normalize ...
        from gnomes import gnomes_norm
        sys.argv = ["normdb", "normalize"] + rest
        return gnomes_norm.main()

    if cmd == "diff":
        # IMPORTANT:
        # If your diff script expects: normdb diff ...
        from gnomes import gnomes_diff
        sys.argv = ["normdb", "diff"] + rest
        return gnomes_diff.main()

        # If instead your diff script is already "one-level" (no subcommand),
        # use this instead:
        # sys.argv = ["gnomes_diff"] + rest
        # return gnomes_diff.main()

    print(f"Unknown command: {cmd}\n")
    print(HELP)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
