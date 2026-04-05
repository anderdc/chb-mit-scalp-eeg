#!/bin/bash
# run jupyter lab with uv

# Compute path to project root (directory of this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"
uv run jupyter lab
