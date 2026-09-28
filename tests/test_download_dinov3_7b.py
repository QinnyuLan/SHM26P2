"""No network/GPU: pinned declarations, interrupted transfer, resume and SHA guards."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import requests

spec = importlib.util.spec_from_file_location(
    "download_dinov3_7b", Path(__file__).parents[1] / "scripts/download_dinov3_7b.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    def __init__(self, payload, start=0, total=None, fail=False, status=206):
        self.payload, self.fail, self.status_code = payload, fail, status
        total = len(payload) + start if total is None else total
        self.headers = {"Content-Range": f"bytes {start}-{total-1}/{total}",
                        "Content-Length": str(total-start)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload
        if self.fail:
            raise requests.ConnectionError("synthetic interrupted network")


def test_listing_rejects_changed_sha_or_size():
    expected = {"config.json": (3, "abc")}
    listing = {"Code": 200, "Success": True,
               "Data": {"Files": [{"Path": "config.json", "Size": 3, "Sha256": "abc"}]}}
    assert module.validate_listing(listing, expected)["config.json"]["size"] == 3
    listing["Data"]["Files"][0]["Sha256"] = "changed"
    with pytest.raises(module.ContractError, match="declaration"):
        module.validate_listing(listing, expected)


def test_transfer_resume_after_network_error_and_atomic_finish(tmp_path):
    data = b"abcdef0123456789"
    calls = []

    def get(url, **kwargs):
        start = int(kwargs["headers"]["Range"].split("=")[1][:-1])
        calls.append(start)
        if start == 0:
            return Response(data[:6], total=len(data), fail=True)
        return Response(data[start:], start=start)

    result = module.download_file(tmp_path, "weights", len(data), hashlib.sha256(data).hexdigest(),
                                  get=get, pause=lambda _: None)
    assert calls == [0, 6]
    assert result["attempts"] == 2
    assert (tmp_path / "weights").read_bytes() == data
    assert not (tmp_path / "weights.part").exists()


def test_restart_uses_existing_partial_and_does_not_duplicate(tmp_path):
    data = b"abcxyz"
    (tmp_path / "weights.part").write_bytes(data[:3])

    def get(url, **kwargs):
        assert kwargs["headers"]["Range"] == "bytes=3-"
        return Response(data[3:], start=3)

    result = module.download_file(tmp_path, "weights", 6, hashlib.sha256(data).hexdigest(), get=get)
    assert result["resumed_from_bytes"] == 3
    assert (tmp_path / "weights").read_bytes() == data


def test_ignored_resume_range_never_appends_full_body(tmp_path):
    (tmp_path / "weights.part").write_bytes(b"abc")
    with pytest.raises(module.ContractError, match="ignored Range"):
        module.download_file(tmp_path, "weights", 6, "irrelevant",
                             get=lambda *a, **kw: Response(b"abcdef", status=200))
    assert (tmp_path / "weights.part").read_bytes() == b"abc"
    assert not (tmp_path / "weights").exists()


def test_completed_partial_hash_mismatch_is_preserved_without_network(tmp_path):
    (tmp_path / "weights.part").write_bytes(b"corrupt")

    def get(*args, **kwargs):
        raise AssertionError("complete partial must be checked without network")

    with pytest.raises(module.ContractError, match="SHA256"):
        module.download_file(tmp_path, "weights", 7, hashlib.sha256(b"correct").hexdigest(), get=get)
    assert (tmp_path / "weights.part").read_bytes() == b"corrupt"
    assert not (tmp_path / "weights").exists()


def test_existing_verified_file_skips_download(tmp_path):
    data = b"correct"
    (tmp_path / "weights").write_bytes(data)

    def get(*args, **kwargs):
        raise AssertionError("existing valid target must not download")

    result = module.download_file(tmp_path, "weights", len(data), hashlib.sha256(data).hexdigest(),
                                  get=get)
    assert result["status"] == "verified_existing"


def test_transient_retries_are_bounded_and_partial_survives(tmp_path):
    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs["headers"]["Range"])
        raise requests.Timeout("synthetic timeout")

    with pytest.raises(requests.Timeout):
        module.download_file(tmp_path, "weights", 7, "unused", get=get,
                             attempts=3, pause=lambda _: None)
    assert len(calls) == 3
    assert not (tmp_path / "weights").exists()


def test_pinned_manifest_matches_read_only_modelscope_audit():
    import json
    source = json.loads((Path(__file__).parents[1] /
                         "artifacts/dinov3_7b_metadata_audit.json").read_text())
    assert module.REVISION == source["revision"]
    assert len(module.validate_listing(source["listing"])) == 12
    assert sum(size for name, (size, _) in module.PINNED.items()
               if name.endswith(".safetensors")) == 26864210088
