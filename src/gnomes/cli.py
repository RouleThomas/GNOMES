#!/usr/bin/env python3
import sys

HELP = """\
GNOMES — Genome-wide NOrmalization of Mapped Epigenomic Signals

Usage:
  GNOMES norm [options]    Normalize BAM → bigWigs (P99 scaling)
  GNOMES diff [options]    Differential binding analysis (DESeq2/edgeR)

Run:
  GNOMES <command> --help
to see all options for a command.
"""


def main() -> int:
    argv = sys.argv[1:]

    # Global help
    if len(argv) == 0 or argv[0] in ("-h", "--help"):
        print(HELP)
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "norm":
        # Your gnomes_norm.py defines a subcommand: "normalize"
        # We want users to type: GNOMES norm --meta ...
        # So we inject "normalize" unless they already provided it.
        from gnomes import gnomes_norm

        # If user asked help at this level, show the norm module help
        if len(rest) == 0 or rest[0] in ("-h", "--help"):
            sys.argv = ["GNOMES norm", "normalize", "--help"]
            return gnomes_norm.main()

        # If they already provided the subcommand explicitly, keep it
        if rest[0] == "normalize":
            sys.argv = ["GNOMES norm"] + rest
            return gnomes_norm.main()

        # Otherwise inject normalize
        sys.argv = ["GNOMES norm", "normalize"] + rest
        return gnomes_norm.main()

    if cmd == "diff":
        from gnomes import gnomes_diff

        # diff has no subcommand; pass through
        if len(rest) == 0 or rest[0] in ("-h", "--help"):
            sys.argv = ["GNOMES diff", "--help"]
            return gnomes_diff.main()

        sys.argv = ["GNOMES diff"] + rest
        return gnomes_diff.main()

    print(f"Unknown command: {cmd}\n")
    print(HELP)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())