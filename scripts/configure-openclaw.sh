#!/usr/bin/env bash
# configure-openclaw.sh — One-command OpenClaw configuration
# 
# Usage:
#   bash scripts/configure-openclaw.sh        # Add vLLM as available provider
#   bash scripts/configure-openclaw.sh --primary    # Also set as primary model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 is required but not installed"
    exit 1
fi

python3 "${SCRIPT_DIR}/configure-openclaw.py" "$@"
