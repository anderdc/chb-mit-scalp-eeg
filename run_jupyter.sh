#!/bin/bash
# run jupyter lab with venv (must set up venv before this at root of project)

# Compute path to project root (directory of this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to venv python
PY="$SCRIPT_DIR/venv/bin/python"

"$PY" -m jupyter lab
