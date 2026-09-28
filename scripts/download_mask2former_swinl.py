"""Download three pinned official Swin-L files to the data volume, without a cache."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

MODEL_ID = 'facebook/mask2former-swin-large-ade-semantic'
REVISION = 'aa25c92404a40599614215e76514c79b427c7527'
DEFAULT_OUTPUT = Path('/mnt/data/SHM2026/models/mask2former-swin-large-ade-semantic-aa25c924')
API = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}'
FILES = {
    'config.json': {'size': 82540, 'blob_id': '6cc89c35ce70d7227b8956eab0db1c6240ae18a7'},
    'preprocessor_config.json': {'size': 538, 'blob_id': '0723e20e276898ab516725965619f83deadbdc34'},
    'model.safetensors': {'size': 866052064, 'blob_id': '50fd1febdfa679c3aae8d6b5573645fa4b0c02d9',
                          'sha256': 'b143c144341c15b4f20165cc6d2c9305fb1b66792f68a6e0e06d2b20dc063b14'},
}


class ContractError(RuntimeError):
    """Do not automatically retry integrity or source-contract failures."""


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def validate_metadata(metadata):
    if metadata.get('id') != MODEL_ID or metadata.get('sha') != REVISION:
        raise ContractError('Different model or revision')
    rows = {r['rfilename']: r for r in metadata['siblings']}
    for name, pinned in FILES.items():
        row = rows.get(name, {})
        if row.get('size') != pinned['size'] or row.get('blobId') != pinned['blob_id']:
            raise ContractError(f'Server metadata differs from pin: {name}')
        if 'sha256' in pinned and (row.get('lfs', {}).get('sha256') != pinned['sha256']
                                  or row.get('lfs', {}).get('size') != pinned['size']):
            raise ContractError('Weight LFS SHA/size differs from approved pin')
    return {name: rows[name] for name in FILES}


def verify_file(path, declaration):
    if path.stat().st_size != declaration['size']:
        raise ContractError(f'Wrong file size, bytes preserved: {path}')
    actual = sha256(path)
    if 'sha256' in declaration:
        if actual != declaration['sha256']:
            raise ContractError(f'Weight SHA mismatch, bytes preserved: {path}')
    else:
        data = path.read_bytes()
        git_blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if git_blob != declaration['blob_id']:
            raise ContractError(f'Git blob mismatch, bytes preserved: {path}')
        json.loads(data)
    return actual


def validate_response(response, offset, size):
    response.raise_for_status()
    if response.status_code == 206:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if not match or tuple(map(int, match.groups())) != (offset, size-1, size):
            raise ContractError('Invalid Content-Range; preserve partial')
    elif response.status_code != 200 or offset != 0:
        raise ContractError('Server ignored resumed byte range; preserve partial')
    length = response.headers.get('Content-Length')
    if length is not None and int(length) != size-offset:
        raise ContractError('Response size differs from fixed metadata')
    if response.headers.get('Content-Encoding', 'identity') != 'identity':
        raise ContractError('Encoded response cannot be resumed as exact bytes')


def download_file(output, name, declaration, *, attempts=3, get=requests.get, pause=time.sleep, progress=None):
    target, partial = output/name, output/(name+'.part')
    size = declaration['size']
    if target.exists():
        return {'sha256': verify_file(target, declaration), 'bytes': size, 'attempts': 0,
                'status': 'verified_existing'}
    initial = partial.stat().st_size if partial.exists() else 0
    if initial > size:
        raise ContractError('Oversized partial preserved')
    used = 0
    for attempt in range(1, attempts+1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        used = attempt
        try:
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
            with get(url, headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'},
                     stream=True, timeout=(15, 60)) as response:
                validate_response(response, offset, size)
                received = offset
                with partial.open('ab') as stream:
                    for block in response.iter_content(chunk_size=4*1024**2):
                        if not block:
                            continue
                        if received+len(block) > size:
                            raise ContractError('Download exceeds declared size; preserve partial')
                        stream.write(block)
                        received += len(block)
                        if progress:
                            progress(name, received, size)
                    stream.flush()
                    os.fsync(stream.fileno())
                if received != size:
                    raise requests.ConnectionError('Incomplete response; partial preserved')
            break
        except requests.RequestException:
            if attempt == attempts:
                raise
            pause(min(2**attempt, 8))
    actual = verify_file(partial, declaration)
    partial.replace(target)
    return {'sha256': actual, 'bytes': size, 'attempts': used, 'resumed_from_bytes': initial, 'status': 'verified'}


def run(output, attempts):
    response = requests.get(API, params={'blobs': 'true'}, timeout=(15, 60))
    response.raise_for_status()
    metadata = response.json()
    declarations = validate_metadata(metadata)
    source = {'provider': 'HuggingFace', 'model_id': MODEL_ID, 'revision': REVISION,
              'api': API, 'server_declarations': declarations, 'pinned_files': FILES}
    source_path = output/'download_source_manifest.json'
    if source_path.exists() and json.loads(source_path.read_text()) != source:
        raise ContractError('Existing directory has different source binding')
    if not source_path.exists():
        atomic_json(source_path, source)
    receipt_path = output/'download_provenance.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        actual = {name: verify_file(output/name, declaration) for name, declaration in FILES.items()}
        if (receipt.get('status') != 'completed' or receipt.get('revision') != REVISION
                or receipt.get('files_sha256') != actual or receipt.get('model_dir') != str(output)):
            raise ContractError('Existing receipt inconsistent; no overwrite')
        print(json.dumps({'status': 'verified_existing', 'files_sha256': actual}), flush=True)
        return
    remaining = sum(max(0, d['size']-max((output/n).stat().st_size if (output/n).exists() else 0,
                     (output/(n+'.part')).stat().st_size if (output/(n+'.part')).exists() else 0)) for n, d in FILES.items())
    if shutil.disk_usage(output).free < remaining+1024**3:
        raise ContractError('Insufficient data-volume space plus 1 GiB reserve')
    started, last_progress = time.monotonic(), [0.]
    state = {'status': 'running', 'pid': os.getpid(), 'model_id': MODEL_ID, 'revision': REVISION,
             'started_utc': datetime.now(UTC).isoformat(), 'attempts_per_file_max': attempts, 'files': {}}
    atomic_json(output/'download_state.json', state)
    def progress(name, count, size):
        now = time.monotonic()
        if now-last_progress[0] >= 5 or count == size:
            last_progress[0] = now
            print(json.dumps({'file': name, 'bytes': count, 'total_bytes': size,
                              'elapsed_seconds': now-started}), flush=True)
    try:
        for name, declaration in FILES.items():
            state['files'][name] = download_file(output, name, declaration, attempts=attempts, progress=progress)
            atomic_json(output/'download_state.json', state)
        actual = {name: verify_file(output/name, d) for name, d in FILES.items()}
        receipt = {'status': 'completed', 'provider': 'HuggingFace', 'model_id': MODEL_ID,
                   'revision': REVISION, 'model_dir': str(output), 'files_sha256': actual,
                   'files': state['files'], 'source_manifest_sha256': sha256(source_path),
                   'server_declarations': declarations, 'script_sha256': sha256(Path(__file__)),
                   'started_utc': state['started_utc'], 'finished_utc': datetime.now(UTC).isoformat(),
                   'seconds': time.monotonic()-started, 'no_shared_cache': True,
                   'temporary_policy': 'same-directory .part; rename only after exact integrity check'}
        atomic_json(receipt_path, receipt)
        state['status'] = 'completed'
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', partials_preserved=True)
        raise
    finally:
        state['seconds'] = time.monotonic()-started
        atomic_json(output/'download_state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--attempts', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to('/mnt/data'):
        raise ValueError('All downloads and partials must remain on /mnt/data')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.download.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(output, args.attempts)


if __name__ == '__main__':
    main()
