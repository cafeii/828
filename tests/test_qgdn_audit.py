import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/qgdn'))
from audit_fineweb import audit_parquet, audit_token_chunk, prepared_audit


def test_parquet_checks_all_rows_and_detects_hash_change_and_truncation(tmp_path):
    path = tmp_path / 'sample.parquet'
    pq.write_table(pa.table({'text': ['alpha', 'beta', 'gamma'], 'id': ['a', 'b', 'c']}),
                   path, row_group_size=2, write_page_checksum=True)
    original = path.read_bytes()
    expected = dict(path='sample/10BT/sample.parquet', size=len(original),
                    lfs=dict(oid=hashlib.sha256(original).hexdigest()))
    result = audit_parquet((path, expected))
    assert result['status'] == 'passed'
    assert result['decoded_rows'] == 3 and result['decoded_row_groups'] == 2
    # A perfectly readable replacement is still not the official source.
    pq.write_table(pa.table({'text': ['changed'], 'id': ['z']}), path)
    assert audit_parquet((path, expected))['status'] == 'failed'
    path.write_bytes(original[:-8])
    result = audit_parquet((path, expected))
    assert result['status'] == 'failed' and len(result['errors']) >= 2


def test_legacy_chunk_checks_document_offsets_vocabulary_and_eos(tmp_path):
    path = tmp_path / 'chunk.bin'
    header = struct.pack('<4I', 2, 16, 22, 26)
    tokens = np.array([3, 4, 2, 5, 2], dtype='<u2')
    path.write_bytes(header + tokens.tobytes())
    chunk = dict(chunk_size=2, chunk_bytes=26, dim=5)
    assert audit_token_chunk((path, chunk))['status'] == 'passed'
    tokens[-1] = 7
    path.write_bytes(header + tokens.tobytes())
    assert audit_token_chunk((path, chunk))['status'] == 'failed'
    path.write_bytes(struct.pack('<4I', 2, 16, 17, 26) + tokens.tobytes())
    assert audit_token_chunk((path, chunk))['status'] == 'failed'


def test_prepared_check_matches_verified_sources_and_document_counts(tmp_path):
    raw = dict(subsets={'10BT': dict(status='passed', files=[
        dict(path='source-a', sha256='hash-a', decoded_rows=2),
        dict(path='source-b', sha256='hash-b', decoded_rows=1)])})
    report = tmp_path / 'raw.json'
    report.write_text(json.dumps(raw))
    manifest = dict(format='qgdn-u16-v1', vocab_size=32000, sources={
        'train': [dict(path='source-a', sha256='hash-a')],
        'val': [dict(path='source-b', sha256='hash-b')]}, splits={})
    for split, ids, count in [('train', [3, 2, 4, 2], 2), ('val', [5, 2], 1)]:
        path = tmp_path / f'{split}.bin'
        payload = np.array(ids, dtype='<u2').tobytes()
        path.write_bytes(payload)
        manifest['splits'][split] = dict(file=path.name, tokens=len(ids), documents=count,
                                         sha256=hashlib.sha256(payload).hexdigest())
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    output = tmp_path / 'out'
    output.mkdir()
    args = SimpleNamespace(raw_report=report, prepared_manifest=path, output=output)
    assert prepared_audit(args)
    manifest['splits']['train']['documents'] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='omitted/duplicated'):
        prepared_audit(args)
