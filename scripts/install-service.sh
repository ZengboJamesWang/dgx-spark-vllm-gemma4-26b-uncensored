#!/bin/bash
set -e

# Install systemd user service for auto-starting vLLM on boot

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="vllm-gemma4-26b.service"
USER_SERVICE_DIR="$HOME/.config/systemd/user"

echo "=================================="
echo "vLLM Systemd Service Installer"
echo "=================================="

mkdir -p "$USER_SERVICE_DIR"

# Copy service file
cp "$REPO_DIR/systemd/$SERVICE_NAME" "$USER_SERVICE_DIR/"

# Expand %h to actual home directory in the service file
sed -i "s|%h|$HOME|g" "$USER_SERVICE_DIR/$SERVICE_NAME"

# Reload systemd
systemctl --user daemon-reload

# Enable service (auto-start on login/boot)
systemctl --user enable "$SERVICE_NAME"

echo ""
echo "✅ Service installed: $SERVICE_NAME"
echo ""
echo "Commands:"
echo "  Start now:   systemctl --user start $SERVICE_NAME"
echo "  Stop:        systemctl --user stop $SERVICE_NAME"
echo "  Status:      systemctl --user status $SERVICE_NAME"
echo "  Disable:     systemctl --user disable $SERVICE_NAME"
echo ""
echo "The vLLM container will now automatically start when you log in."
echo "To start it immediately, run: systemctl --user start $SERVICE_NAME"
