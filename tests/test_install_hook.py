"""Tests for hook installer."""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.install_hook import (
    install_hook,
    uninstall_hook,
    load_settings,
    save_settings,
    get_settings_path,
    get_capture_command,
)


class TestHookInstaller:
    """Test suite for hook installer."""

    @pytest.fixture
    def temp_settings(self):
        """Create a temporary settings file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
            # Write minimal settings
            json.dump(
                {
                    "$schema": "https://json.schemastore.org/claude-code-settings.json",
                },
                f,
            )
        yield path
        # Cleanup
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_settings_with_hooks(self):
        """Create a temporary settings file with existing hooks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
            json.dump(
                {
                    "$schema": "https://json.schemastore.org/claude-code-settings.json",
                    "hooks": {
                        "PostToolUse": [
                            {
                                "matcher": "",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "some-other-hook",
                                    }
                                ],
                            }
                        ]
                    },
                },
                f,
            )
        yield path
        if path.exists():
            path.unlink()

    def test_get_settings_path(self):
        """Test getting settings path."""
        path = get_settings_path()
        assert path.name == "settings.json"
        assert ".claude" in str(path)

    def test_get_capture_command(self):
        """Test getting capture command."""
        command = get_capture_command()
        assert "grow_ai.capture" in command
        assert "python" in command.lower() or command.endswith(".exe")

    def test_load_settings_existing_file(self, temp_settings):
        """Test loading settings from existing file."""
        settings = load_settings(temp_settings)
        assert isinstance(settings, dict)
        assert "$schema" in settings

    def test_load_settings_nonexistent_file(self):
        """Test loading settings from nonexistent file."""
        nonexistent = Path("/tmp/nonexistent-settings-xyz123.json")
        settings = load_settings(nonexistent)
        assert isinstance(settings, dict)
        assert "$schema" in settings

    def test_save_settings(self, temp_settings):
        """Test saving settings."""
        settings = {"test": "value"}
        save_settings(temp_settings, settings)

        # Verify saved
        loaded = load_settings(temp_settings)
        assert loaded["test"] == "value"

    def test_install_hook_creates_structure(self, temp_settings):
        """Test installing hook creates hooks structure if missing."""
        assert not load_settings(temp_settings).get("hooks")

        install_hook(temp_settings)

        settings = load_settings(temp_settings)
        assert "hooks" in settings
        assert "PostToolUse" in settings["hooks"]

    def test_install_hook_adds_hook(self, temp_settings):
        """Test installing hook adds the hook."""
        install_hook(temp_settings)

        settings = load_settings(temp_settings)
        hooks = settings["hooks"]["PostToolUse"][0]["hooks"]
        assert len(hooks) == 1
        assert hooks[0]["type"] == "command"
        assert "grow_ai.capture" in hooks[0]["command"]

    def test_install_hook_idempotent(self, temp_settings):
        """Test installing hook is idempotent."""
        result1 = install_hook(temp_settings)
        hooks_after_first = len(load_settings(temp_settings)["hooks"]["PostToolUse"][0]["hooks"])

        result2 = install_hook(temp_settings)
        hooks_after_second = len(load_settings(temp_settings)["hooks"]["PostToolUse"][0]["hooks"])

        assert result1 is True  # First install makes changes
        assert result2 is False  # Second install doesn't
        assert hooks_after_first == hooks_after_second == 1

    def test_install_hook_preserves_existing(self, temp_settings_with_hooks):
        """Test installing hook preserves existing hooks."""
        install_hook(temp_settings_with_hooks)

        settings = load_settings(temp_settings_with_hooks)
        hooks = settings["hooks"]["PostToolUse"][0]["hooks"]
        assert len(hooks) == 2
        assert hooks[0]["command"] == "some-other-hook"
        assert "grow_ai.capture" in hooks[1]["command"]

    def test_uninstall_hook(self, temp_settings):
        """Test uninstalling hook."""
        install_hook(temp_settings)
        assert len(load_settings(temp_settings)["hooks"]["PostToolUse"][0]["hooks"]) == 1

        uninstall_hook(temp_settings)
        assert len(load_settings(temp_settings)["hooks"]["PostToolUse"][0]["hooks"]) == 0

    def test_uninstall_preserves_other_hooks(self, temp_settings_with_hooks):
        """Test uninstalling preserves other hooks."""
        install_hook(temp_settings_with_hooks)
        uninstall_hook(temp_settings_with_hooks)

        hooks = load_settings(temp_settings_with_hooks)["hooks"]["PostToolUse"][0]["hooks"]
        assert len(hooks) == 1
        assert hooks[0]["command"] == "some-other-hook"

    def test_uninstall_not_found(self, temp_settings):
        """Test uninstalling when hook not found."""
        result = uninstall_hook(temp_settings)
        assert result is False

    def test_dry_run_doesnt_modify(self, temp_settings):
        """Test dry run doesn't modify settings."""
        install_hook(temp_settings, dry_run=True)

        # Verify no changes
        assert "hooks" not in load_settings(temp_settings)

    def test_install_creates_directory(self):
        """Test installing creates .claude directory if needed."""
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"

            install_hook(settings_path)

            assert settings_path.exists()
            assert settings_path.parent.exists()
