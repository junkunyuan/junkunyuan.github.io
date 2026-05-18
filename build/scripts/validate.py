"""Validate all paper YAML files. Exits non-zero on any error."""

from __future__ import annotations

import glob
import os
import sys

from build.lib.schema import SchemaError, validate_all

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    pub = os.path.join(PROJECT_ROOT, "data", "publications.yaml")
    domains = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "reading_list", "*.yaml")))
    try:
        validate_all(pub, domains)
    except SchemaError as e:
        print(e)
        return 1
    print(f"OK — {len(domains)} domain file(s) + publications list pass validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
