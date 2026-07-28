"""Tests for the Compress & Publish pipeline stage."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vs_pipeline.stages.compress import compress_glb, publish_property
from vs_pipeline.storage import LocalStorageProvider


def test_compress_glb_passthrough_when_no_draco(tmp_path: Path):
    in_file = tmp_path / "in.glb"
    out_file = tmp_path / "out.glb"
    in_file.write_bytes(b"dummy")
    
    # has_draco = False -> returns False immediately without calling subprocess
    res = compress_glb(in_file, out_file, has_draco=False)
    assert res is False


@patch("subprocess.run")
def test_compress_glb_with_draco(mock_run, tmp_path: Path):
    in_file = tmp_path / "in.glb"
    out_file = tmp_path / "out.glb"
    in_file.write_bytes(b"dummy")
    
    res = compress_glb(in_file, out_file, has_draco=True)
    assert res is True
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][:2] == ["gltf-transform", "optimize"]


def test_publish_property_missing_proxy_out(tmp_path: Path):
    # property_dir has no proxy_out
    with pytest.raises(FileNotFoundError, match="Missing proxy_out"):
        publish_property(tmp_path, tmp_path / "published")


@patch("vs_pipeline.stages.compress.check_tool")
@patch("vs_pipeline.stages.compress.get_storage_provider")
def test_publish_property_success(mock_get_storage, mock_check_tool, tmp_path: Path):
    # Setup mock storage
    mock_storage = MagicMock(spec=LocalStorageProvider)
    mock_storage.upload_file.side_effect = lambda local_path, key, ct: f"/fake_cdn/{key}"
    mock_get_storage.return_value = mock_storage
    
    # Mock check_tool to pretend draco is absent
    mock_check_tool.return_value = False

    # Setup property dir
    proxy_dir = tmp_path / "proxy_out"
    proxy_dir.mkdir()
    (proxy_dir / "proxy_shell.glb").write_text("shell")
    
    details_dir = tmp_path / "details_out"
    details_dir.mkdir()
    (details_dir / "proxy_details.glb").write_text("details")
    
    env_dir = tmp_path / "env_out"
    env_dir.mkdir()
    (env_dir / "test.env").write_text("env")
    
    out_dir = tmp_path / "published"
    
    publish_property(tmp_path, out_dir)
    
    # Verify manifest
    manifest_file = out_dir / "published_manifest.json"
    assert manifest_file.exists()
    
    manifest = json.loads(manifest_file.read_text())
    assert manifest["proxy_shell"] == "/fake_cdn/proxy_shell.glb"
    assert manifest["proxy_details"] == "/fake_cdn/proxy_details.glb"
    assert manifest["env_test"] == "/fake_cdn/env/test.env"
    
    # Verify storage calls
    assert mock_storage.upload_file.call_count == 3
