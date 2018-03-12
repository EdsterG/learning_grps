#!/usr/bin/env bash
set -e

# Check if pip installed
if ! command -v pip; then
    echo "ERROR: pip not installed" >&2
    exit 1
fi

pip install numpy h5py matplotlib
