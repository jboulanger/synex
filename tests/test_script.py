"""Tests for the batch-processing script."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

spec = spec_from_file_location("batch_script", Path(__file__).parents[1] / "script.py")
assert spec is not None
assert spec.loader is not None
script = module_from_spec(spec)
spec.loader.exec_module(script)


def test_process_image_passes_arguments_to_synxm(monkeypatch, tmp_path):
    """The worker forwards the image, output directory, and parameters."""
    filename = tmp_path / "image.czi"
    parameters = {"channels": ["RIM", "PSD"]}
    calls = []

    monkeypatch.setattr(
        script.synxm,
        "process",
        lambda *args: calls.append(args),
    )

    result = script.process_image((filename, tmp_path, parameters))

    assert result is None
    assert calls == [(filename, tmp_path, parameters)]


def test_process_image_propagates_processing_errors(monkeypatch, tmp_path):
    """A failed image task is reported to the parent process."""
    filename = tmp_path / "image.czi"

    def fail(*args):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(script.synxm, "process", fail)

    with pytest.raises(RuntimeError, match="processing failed"):
        script.process_image((filename, tmp_path, {}))
