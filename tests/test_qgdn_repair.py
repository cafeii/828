import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/qgdn'))
from repair_fineweb import checked_directory, download_verified, reference_verified_source


@pytest.fixture
def server():
    state = {'responses': [], 'requests': 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            index = state['requests']
            state['requests'] += 1
            status, payload, headers = state['responses'][min(index, len(state['responses']) - 1)]
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    service = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f'http://127.0.0.1:{service.server_port}/shard'
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


def test_corrupt_same_size_response_is_retried_and_valid_bytes_are_published(tmp_path, server):
    state, url = server
    good = b'PAR1' + b'a' * 100 + b'PAR1'
    state['responses'] = [(200, b'x' * len(good), {}), (200, good, {'Content-Length': str(len(good))})]
    final = tmp_path / 'shard.parquet'
    result = download_verified(url, final, len(good), hashlib.sha256(good).hexdigest(), backoff=0)
    assert result['status'] == 'passed' and result['attempts'] == 2
    assert final.read_bytes() == good
    assert list(tmp_path.iterdir()) == [final]
    # A resumed repair rechecks an existing file and does not download it again.
    assert download_verified(url, final, len(good), hashlib.sha256(good).hexdigest())['mode'] == 'reused_download'
    assert state['requests'] == 2


@pytest.mark.parametrize('status,payload,headers', [
    (206, b'complete', {'Content-Range': 'bytes 0-7/8'}),
    (200, b'cut', {'Content-Length': '8'}),
    (200, b'cut', {}),
    (200, b'completeEXTRA', {}),
])
def test_partial_truncated_or_oversized_download_is_never_published(tmp_path, server, status, payload, headers):
    state, url = server
    state['responses'] = [(status, payload, headers)]
    final = tmp_path / 'shard.parquet'
    result = download_verified(url, final, 8, hashlib.sha256(b'complete').hexdigest(), attempts=1)
    assert result['status'] == 'failed'
    assert not final.exists() and not list(tmp_path.iterdir())


def test_existing_unverified_data_is_preserved_and_symlink_downloads_are_refused(tmp_path):
    source = tmp_path / 'shared.parquet'
    source.write_bytes(b'original')
    expected = hashlib.sha256(b'repaired').hexdigest()
    with pytest.raises(ValueError, match='preserving'):
        download_verified('unused', source, 8, expected)
    destination = tmp_path / 'private.parquet'
    destination.symlink_to(source)
    with pytest.raises(ValueError, match='symlink'):
        download_verified('unused', destination, 8, expected)
    assert source.read_bytes() == b'original'


def test_read_only_reference_and_output_scope(tmp_path):
    source = tmp_path / 'shared.parquet'
    source.write_bytes(b'original')
    checked = dict(bytes=8, mtime_ns=source.stat().st_mtime_ns, sha256=hashlib.sha256(b'original').hexdigest())
    root = tmp_path / 'personal'
    root.mkdir()
    destination = root / 'ref.parquet'
    reference_verified_source(source, destination, checked)
    assert destination.is_symlink() and source.read_bytes() == b'original'
    source.write_bytes(b'changed-and-longer')
    with pytest.raises(ValueError, match='changed'):
        reference_verified_source(source, destination, checked)
    with pytest.raises(ValueError, match='inside'):
        checked_directory(tmp_path / 'outside', root)
    redirect = root / 'redirect'
    redirect.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='inside'):
        checked_directory(redirect / 'outside', root)
