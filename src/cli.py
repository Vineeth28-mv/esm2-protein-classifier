"""Shared command-line helpers."""

import os
import sys


def run(main):
    """Run ``main()``; show expected errors as a short message, not a traceback.

    Set the environment variable ESM2CLF_DEBUG=1 to see full tracebacks.
    """
    if os.environ.get("ESM2CLF_DEBUG"):
        return main()
    try:
        return main()
    except (ValueError, FileNotFoundError, OSError) as err:
        print(f"\nError: {err}", file=sys.stderr)
        print("(set ESM2CLF_DEBUG=1 to see the full technical details)", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
