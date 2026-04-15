"""Idempotent installer for grow_ai PostToolUse hook.

Installs/updates the grow_ai capture pipeline hook in ~/.claude/settings.json.
Idempotent: safe to run multiple times.
"""

import json
import os
import sys
from pathlib import Path


def get_settings_path() -> Path:
    """Get path to Claude Code settings.json."""
    home = Path.home()
    settings_path = home / ".claude" / "settings.json"
    return settings_path


def load_settings(settings_path: Path) -> dict:
    """Load settings from JSON file.

    Args:
        settings_path: Path to settings.json

    Returns:
        Settings dict, or empty structure if file doesn't exist
    """
    if settings_path.exists():
        with open(settings_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {settings_path} contains invalid JSON")
                sys.exit(1)
    else:
        # Create minimal settings structure
        return {
            "$schema": "https://json.schemastore.org/claude-code-settings.json",
        }


def save_settings(settings_path: Path, settings: dict) -> None:
    """Save settings to JSON file.

    Args:
        settings_path: Path to settings.json
        settings: Settings dict to save
    """
    # Ensure directory exists
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)


def get_capture_command() -> str:
    """Get the capture pipeline command.

    Returns the full command path to run the capture pipeline.
    """
    # Determine if we're on Windows or Unix
    if sys.platform == "win32":
        python_exe = sys.executable
        return f"{python_exe} -m grow_ai.capture"
    else:
        # Unix: use which to find python, or absolute path
        python_exe = sys.executable
        return f"{python_exe} -m grow_ai.capture"


def install_hook(settings_path: Path, dry_run: bool = False) -> bool:
    """Install PostToolUse hook idempotently.

    Args:
        settings_path: Path to settings.json
        dry_run: If True, don't write changes

    Returns:
        True if changes were made, False if already installed
    """
    settings = load_settings(settings_path)

    # Ensure hooks section exists
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Ensure PostToolUse section exists
    if "PostToolUse" not in settings["hooks"]:
        settings["hooks"]["PostToolUse"] = [
            {
                "matcher": "",
                "hooks": [],
            }
        ]

    # Get or create the default hook matcher (empty string = match all)
    post_tool_use = settings["hooks"]["PostToolUse"]
    default_matcher = None

    for item in post_tool_use:
        if item.get("matcher", "") == "":
            default_matcher = item
            break

    if default_matcher is None:
        default_matcher = {"matcher": "", "hooks": []}
        post_tool_use.append(default_matcher)

    # Check if our hook already exists
    capture_command = get_capture_command()
    hooks_list = default_matcher.get("hooks", [])

    hook_exists = False
    for hook in hooks_list:
        if hook.get("type") == "command" and "grow_ai.capture" in hook.get("command", ""):
            hook_exists = True
            break

    if hook_exists:
        print("[OK] grow_ai capture hook already installed")
        return False

    # Add our hook
    new_hook = {
        "type": "command",
        "command": capture_command,
    }

    if "hooks" not in default_matcher:
        default_matcher["hooks"] = []

    default_matcher["hooks"].append(new_hook)

    if dry_run:
        print("DRY RUN: Would install hook:")
        print(json.dumps(new_hook, indent=2))
        return True

    # Write settings
    save_settings(settings_path, settings)
    print(f"[OK] Installed grow_ai capture hook to {settings_path}")
    print(f"  Command: {capture_command}")
    return True


def main():
    """Entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Install grow_ai capture pipeline hook into Claude Code settings"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without modifying settings",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="Path to settings.json (default: ~/.claude/settings.json)",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the hook instead of installing it",
    )

    args = parser.parse_args()

    settings_path = args.settings or get_settings_path()

    if args.uninstall:
        uninstall_hook(settings_path, dry_run=args.dry_run)
    else:
        changed = install_hook(settings_path, dry_run=args.dry_run)
        sys.exit(0 if changed or args.dry_run else 1)


def uninstall_hook(settings_path: Path, dry_run: bool = False) -> bool:
    """Remove PostToolUse hook.

    Args:
        settings_path: Path to settings.json
        dry_run: If True, don't write changes

    Returns:
        True if changes were made, False if hook wasn't installed
    """
    if not settings_path.exists():
        print("Settings file not found")
        return False

    settings = load_settings(settings_path)

    if "hooks" not in settings or "PostToolUse" not in settings["hooks"]:
        print("No PostToolUse hook section found")
        return False

    post_tool_use = settings["hooks"]["PostToolUse"]
    changed = False

    for item in post_tool_use:
        if item.get("matcher", "") == "":
            hooks_list = item.get("hooks", [])
            # Remove grow_ai capture hook
            original_count = len(hooks_list)
            item["hooks"] = [
                h for h in hooks_list
                if not ("grow_ai.capture" in h.get("command", ""))
            ]
            if len(item["hooks"]) < original_count:
                changed = True

    if not changed:
        print("grow_ai capture hook not found")
        return False

    if dry_run:
        print("DRY RUN: Would remove grow_ai capture hook")
        return True

    save_settings(settings_path, settings)
    print(f"[OK] Removed grow_ai capture hook from {settings_path}")
    return True


if __name__ == "__main__":
    main()
