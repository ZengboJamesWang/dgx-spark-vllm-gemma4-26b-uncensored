#!/usr/bin/env python3
"""
Configure OpenClaw to use the local vLLM Gemma-4-26B endpoint.

This script safely patches ~/.openclaw/openclaw.json to add:
- vLLM provider configuration
- vLLM plugin enablement
- Model alias
- Optionally set as primary model

Usage:
    bash scripts/configure-openclaw.sh        # Add as available provider
    bash scripts/configure-openclaw.sh --primary    # Also set as primary model
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"
BACKUP_DIR = Path.home() / ".openclaw" / "backups"

VLLM_PROVIDER = {
    "baseUrl": "http://localhost:8000/v1",
    "apiKey": "vllm-local",
    "api": "openai-completions",
    "models": [
        {
            "id": "gemma-4-26b-uncensored-vllm",
            "name": "gemma-4-26b-uncensored-vllm",
            "reasoning": False,
            "input": ["text"],
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0
            },
            "contextWindow": 262000,
            "maxTokens": 16384
        }
    ]
}

VLLM_MODEL_ALIAS = {
    "alias": "gemma4-26b-vllm"
}

def backup_config():
    """Create timestamped backup of openclaw.json"""
    if not CONFIG_PATH.exists():
        print(f"❌ OpenClaw config not found at {CONFIG_PATH}")
        print("   Please install and run OpenClaw first to generate the config.")
        sys.exit(1)
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"openclaw.json.backup.{timestamp}"
    shutil.copy2(CONFIG_PATH, backup_path)
    print(f"📦 Backup created: {backup_path}")
    return backup_path

def load_config():
    """Load and return openclaw.json"""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def save_config(config):
    """Save config back to openclaw.json with nice formatting"""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write('\n')

def add_vllm_provider(config):
    """Add vLLM provider to models.providers"""
    if "models" not in config:
        config["models"] = {}
    if "providers" not in config["models"]:
        config["models"]["providers"] = {}
    
    config["models"]["providers"]["vllm"] = VLLM_PROVIDER
    print("✅ Added vLLM provider")

def enable_vllm_plugin(config):
    """Enable vLLM plugin in plugins.entries"""
    if "plugins" not in config:
        config["plugins"] = {"entries": {}}
    if "entries" not in config["plugins"]:
        config["plugins"]["entries"] = {}
    
    config["plugins"]["entries"]["vllm"] = {"enabled": True}
    print("✅ Enabled vLLM plugin")

def add_model_alias(config):
    """Add model alias to agents.defaults.models"""
    if "agents" not in config:
        config["agents"] = {"defaults": {"models": {}}}
    if "defaults" not in config["agents"]:
        config["agents"]["defaults"] = {}
    if "models" not in config["agents"]["defaults"]:
        config["agents"]["defaults"]["models"] = {}
    
    config["agents"]["defaults"]["models"]["vllm/gemma-4-26b-uncensored-vllm"] = VLLM_MODEL_ALIAS
    print("✅ Added model alias 'gemma4-26b-vllm'")

def set_primary_model(config):
    """Set vLLM as primary model in agents.defaults.model"""
    if "agents" not in config:
        config["agents"] = {"defaults": {}}
    if "defaults" not in config["agents"]:
        config["agents"]["defaults"] = {}
    
    config["agents"]["defaults"]["model"] = {
        "primary": "vllm/gemma-4-26b-uncensored-vllm",
        "fallbacks": [
            "minimax/MiniMax-M2.5",
            "ollama/gemma4:26b"
        ]
    }
    print("✅ Set vLLM as primary model")

def main():
    print("🔧 OpenClaw vLLM Configuration Tool")
    print("=" * 50)
    print()
    
    # Check for --primary flag
    set_primary = "--primary" in sys.argv
    
    # Backup
    backup_path = backup_config()
    
    # Load
    print(f"📂 Loading config from {CONFIG_PATH}")
    config = load_config()
    
    # Apply changes
    print()
    print("Applying changes...")
    add_vllm_provider(config)
    enable_vllm_plugin(config)
    add_model_alias(config)
    
    if set_primary:
        set_primary_model(config)
    else:
        print("ℹ️  Use --primary flag to also set vLLM as your default model")
    
    # Save
    save_config(config)
    print()
    print("=" * 50)
    print("✅ Configuration complete!")
    print()
    print("Next steps:")
    print("  1. Restart OpenClaw gateway if running:")
    print("     systemctl --user restart openclaw-gateway")
    print("  2. Test with: openclaw chat 'Hello!'")
    print()
    if not set_primary:
        print("To set as primary model, run:")
        print("  bash scripts/configure-openclaw.sh --primary")

if __name__ == "__main__":
    main()
