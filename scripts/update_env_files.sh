#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "Error: No Conda environment is activated."
  echo "Please activate a Conda environment before running this script."
  exit 1
fi

ENV_NAME=$CONDA_DEFAULT_ENV

echo "Creating/updating environment.yml in $PARENT_DIR..."
if ! conda env export | grep -v "^prefix: " > "$PARENT_DIR/environment.yml"; then
  echo "Error: Failed to export Conda environment."
  exit 2
fi

echo "Creating/updating requirements.txt in $PARENT_DIR..."
if ! pip freeze > "$PARENT_DIR/requirements.txt"; then
  echo "Error: Failed to generate requirements.txt."
  exit 3
fi

echo "Environment files created/updated successfully!"
echo "Environment name: $ENV_NAME"
echo "- $PARENT_DIR/environment.yml"
echo "- $PARENT_DIR/requirements.txt"
