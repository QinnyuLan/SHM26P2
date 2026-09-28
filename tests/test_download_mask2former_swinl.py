import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('download_mask2former_swinl',
    Path(__file__).resolve().parents[1]/'scripts/download_mask2former_swinl.py')
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class Response:
    def __init__(self, data, status, headers):
        self.data, self.status_code, self.headers = data, status, headers

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def test_resume_exact_range_and_verify_before_rename(tmp_path):
    payload = b'abcdef'
    (tmp_path/'model.part').write_bytes(payload[:2])
    def get(url, **kwargs):
        assert kwargs['headers']['Range'] == 'bytes=2-'
        assert download.REVISION in url
        return Response(payload[2:], 206, {'Content-Range': 'bytes 2-5/6', 'Content-Length': '4'})
    declaration = {'size': 6, 'sha256': hashlib.sha256(payload).hexdigest()}
    result = download.download_file(tmp_path, 'model', declaration, get=get)
    assert result['resumed_from_bytes'] == 2 and result['attempts'] == 1
    assert (tmp_path/'model').read_bytes() == payload and not (tmp_path/'model.part').exists()


@pytest.mark.parametrize('status,headers', [(200, {'Content-Length': '6'}),
    (206, {'Content-Range': 'bytes 0-5/6', 'Content-Length': '6'}),
    (206, {'Content-Range': 'bytes 2-5/6', 'Content-Encoding': 'gzip'})])
def test_invalid_resume_contract_rejected(status, headers):
    with pytest.raises(download.ContractError):
        download.validate_response(Response(b'', status, headers), 2, 6)


def test_bad_completed_file_is_preserved_and_not_downloaded(tmp_path):
    p = tmp_path/'model'
    p.write_bytes(b'bad')
    def forbidden(*args, **kwargs):
        raise AssertionError('must not overwrite or request')
    with pytest.raises(download.ContractError, match='SHA mismatch'):
        download.download_file(tmp_path, 'model', {'size': 3, 'sha256': '0'*64}, get=forbidden)
    assert p.read_bytes() == b'bad'


def test_small_json_git_blob_validates_and_returns_actual_sha256(tmp_path):
    content = b'{"x":1}\n'
    p = tmp_path/'config.json'
    p.write_bytes(content)
    blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    assert download.verify_file(p, {'size': len(content), 'blob_id': blob}) == hashlib.sha256(content).hexdigest()
    with pytest.raises(download.ContractError, match='Git blob mismatch'):
        download.verify_file(p, {'size': len(content), 'blob_id': '0'*40})


def test_wrong_lfs_server_hash_rejected():
    rows = []
    for name, pinned in download.FILES.items():
        row = {'rfilename': name, 'size': pinned['size'], 'blobId': pinned['blob_id']}
        if 'sha256' in pinned:
            row['lfs'] = {'size': pinned['size'], 'sha256': '0'*64}
        rows.append(row)
    with pytest.raises(download.ContractError, match='LFS'):
        download.validate_metadata({'id': download.MODEL_ID, 'sha': download.REVISION, 'siblings': rows})
