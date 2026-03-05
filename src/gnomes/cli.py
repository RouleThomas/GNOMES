#!/usr/bin/env python3
import sys

HELP = """\
GNOMES — Genome-wide NOrmalization of Mapped Epigenomic Signals

Usage:
  GNOMES norm [options]        Normalize BAM → bigWigs (P99 scaling)
  GNOMES diff [options]        Differential binding analysis (DESeq2/edgeR)
  GNOMES consensus [options]   Build MACS2 pooled-per-condition consensus peak BEDs (grid search)

Run:
  GNOMES <command> --help
to see all options for a command.
"""


def _is_help(x: str) -> bool:
    return x in ("-h", "--help")


def main() -> int:
    argv = sys.argv[1:]

    # Global help
    if len(argv) == 0 or _is_help(argv[0]):
        print(HELP)
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "norm":
        # gnomes_norm.py historically had a subcommand: "normalize"
        # Keep user-facing CLI as: GNOMES norm ...
        from gnomes import gnomes_norm

        if len(rest) == 0 or _is_help(rest[0]):
            # Show the norm help (at the injected normalize level)
            sys.argv = ["GNOMES norm", "normalize", "--help"]
            return gnomes_norm.main()

        # If they already provided the subcommand explicitly, keep it
        if rest[0] == "normalize":
            sys.argv = ["GNOMES norm"] + rest
            return gnomes_norm.main()

        # Otherwise inject normalize
        sys.argv = ["GNOMES norm", "normalize"] + rest
        return gnomes_norm.main()

    elif cmd == "diff":
        from gnomes import gnomes_diff

        if len(rest) == 0 or _is_help(rest[0]):
            sys.argv = ["GNOMES diff", "--help"]
            return gnomes_diff.main()

        sys.argv = ["GNOMES diff"] + rest
        return gnomes_diff.main()

    elif cmd == "consensus":
        from gnomes import gnomes_consensus

        if len(rest) == 0 or _is_help(rest[0]):
            sys.argv = ["GNOMES consensus", "--help"]
            return gnomes_consensus.main()

        sys.argv = ["GNOMES consensus"] + rest
        return gnomes_consensus.main()

    else:
        print(f"Unknown command: {cmd}\n")
        print(HELP)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())