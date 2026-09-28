import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "capacity_sequence", Path(__file__).parents[1] / "scripts/execute_teacher_capacity_sequence.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("fail_at", [None, 1])
def test_sequence_waits_for_success_and_never_retries(tmp_path, monkeypatch, fail_at):
    stages = [{"key": key, "command": [key], "log_path": str(tmp_path / f"{key}.log"),
               "stage_receipt": str(tmp_path / f"{key}.json"), "steps": steps}
              for key, steps in (("hplus_real", 6000), ("hplus_render_adapt", 2000),
                                 ("vit7b_real", 6000), ("vit7b_render_adapt", 2000))]
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            self.index, self.pid = len(calls), 100 + len(calls)
            calls.append(command[0])

        def wait(self):
            if self.index == fail_at:
                return 17
            stage = stages[self.index]
            Path(stage["stage_receipt"]).write_text(json.dumps(
                {"status": "completed", "steps": stage["steps"]}))
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    manifest = tmp_path / "launch.json"
    manifest.write_text(json.dumps({"status": "locked_authorized", "locked_files_sha256": {},
        "output_root": str(tmp_path), "workspace_root": str(tmp_path), "environment": {},
        "stages": stages}))
    if fail_at is None:
        runner.execute(manifest)
    else:
        with pytest.raises(RuntimeError, match="no retry"):
            runner.execute(manifest)
    receipt = json.loads((tmp_path / "execution_receipt.json").read_text())
    assert receipt["status"] == ("completed" if fail_at is None else "failed")
    assert calls == [stage["key"] for stage in stages[:4 if fail_at is None else fail_at + 1]]
    with pytest.raises(FileExistsError):
        runner.execute(manifest)
