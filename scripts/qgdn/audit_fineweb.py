"""Read-only FineWeb byte/Parquet audits against a pinned official HF manifest."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import struct
import time

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def audit_parquet(task):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    path, expected = Path(task[0]), task[1]
    result = dict(path=str(path), official_path=expected['path'], errors=[])
    errors = result['errors']
    started = time.monotonic()
    try:
        before = path.stat()
        result.update(bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                      expected_bytes=expected['size'], expected_sha256=expected['lfs']['oid'])
        result['sha256'] = sha256(path)
        result['bytes_match'] = before.st_size == expected['size']
        result['sha256_match'] = result['sha256'] == expected['lfs']['oid']
        if not result['bytes_match'] or not result['sha256_match']:
            errors.append('Bytes or SHA-256 differ from pinned upstream')
        f = pq.ParquetFile(path, page_checksum_verification=True)
        schema = f.schema_arrow
        text_type = schema.field('text').type
        if not (pa.types.is_string(text_type) or pa.types.is_large_string(text_type)):
            raise ValueError('text column is not a UTF-8 string')
        result.update(metadata_rows=f.metadata.num_rows, row_groups=f.metadata.num_row_groups,
                      schema=str(schema), created_by=f.metadata.created_by,
                      decoded_rows=0, decoded_row_groups=0, text_nulls=0, text_empty=0)
        for group in range(f.metadata.num_row_groups):
            rows = 0
            try:
                for batch in f.iter_batches(batch_size=1024, row_groups=[group], use_threads=False):
                    batch.validate(full=True)  # Decode/validate every column, including UTF-8.
                    text = batch.column(batch.schema.get_field_index('text'))
                    result['text_nulls'] += text.null_count
                    result['text_empty'] += pc.sum(pc.equal(text, '')).as_py() or 0
                    rows += batch.num_rows
                if rows != f.metadata.row_group(group).num_rows:
                    raise ValueError('Decoded row count differs from row-group metadata')
                result['decoded_row_groups'] += 1
                result['decoded_rows'] += rows
            except Exception as exc:
                errors.append(f'row_group={group}: {type(exc).__name__}: {exc}')
        if result['decoded_rows'] != result['metadata_rows']:
            errors.append('Decoded total row count differs from metadata')
        if result['text_nulls']:
            errors.append('Null text values would invalidate tokenization')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            errors.append('Source changed during audit')
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    result.update(status='failed' if errors else 'passed', seconds=time.monotonic() - started)
    print(json.dumps({k: result.get(k) for k in ('path', 'status', 'sha256_match', 'decoded_rows', 'seconds', 'errors')}), flush=True)
    return result


def audit_token_chunk(task):
    path, chunk = Path(task[0]), task[1]
    result = dict(path=str(path), errors=[])
    try:
        before = path.stat()
        assert before.st_size == chunk['chunk_bytes'], 'Chunk file size differs from index'
        raw = path.read_bytes()
        result['sha256'] = hashlib.sha256(raw).hexdigest()
        items = struct.unpack_from('<I', raw)[0]
        assert items == chunk['chunk_size'], 'Document count differs from index'
        header_bytes = (items + 2) * 4
        offsets = np.frombuffer(raw, dtype='<u4', offset=4, count=items + 1).astype(np.int64)
        assert offsets[0] == header_bytes and offsets[-1] == len(raw), 'Invalid first/last offsets'
        assert np.all(np.diff(offsets) >= 2) and np.all(offsets % 2 == 0), 'Invalid document offsets'
        tokens = np.frombuffer(raw, dtype='<u2', offset=header_bytes)
        assert len(tokens) == chunk['dim'], 'Token count differs from index'
        assert len(tokens) and int(tokens.max()) < 32000, 'Out-of-vocabulary tokens'
        ends = (offsets[1:] - header_bytes) // 2 - 1
        assert np.all(tokens[ends] == 2), 'Document does not end with Llama-2 EOS'
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Chunk changed during audit'
        result.update(documents=items, tokens=len(tokens), token_min=int(tokens.min()), token_max=int(tokens.max()))
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    result['status'] = 'failed' if result['errors'] else 'passed'
    return result


def raw_audit(args):
    upstream = json.loads(args.upstream.read_text())
    result = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
                  repo_id=upstream['repo_id'], revision=upstream['revision'],
                  upstream_manifest_sha256=sha256(args.upstream), subsets={}, leftovers=[])
    # Only read data; each worker uses one CPU. Fork avoids named-semaphore
    # reopening on this cluster, where unrelated cleanup can remove /dev/shm names.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        for subset in args.subsets:
            directory = args.root / 'sample' / subset
            expected = {Path(f['path']).name: f for f in upstream['subsets'][subset]['files']}
            actual = {f.name: f for f in directory.glob('*.parquet')}
            inventory = dict(missing=sorted(expected.keys() - actual.keys()), extra=sorted(actual.keys() - expected.keys()))
            files = list(pool.map(audit_parquet, [(str(actual[name]), expected[name]) for name in sorted(expected.keys() & actual.keys())]))
            summary = dict(inventory=inventory, expected_files=len(expected), actual_files=len(actual), files=files,
                           bytes=sum(f.get('bytes', 0) for f in files), rows=sum(f.get('decoded_rows', 0) for f in files),
                           row_groups=sum(f.get('decoded_row_groups', 0) for f in files))
            summary['status'] = 'passed' if not any(inventory.values()) and all(f['status'] == 'passed' for f in files) else 'failed'
            result['subsets'][subset] = summary
            write_json(args.output / f'{subset}.json', summary)
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.name in actual:
                    continue
                leftover = dict(path=str(path), bytes=path.stat().st_size, excluded_from_training=True)
                if '.parquet.corrupt-' in path.name:
                    original = path.name.split('.parquet.corrupt-')[0] + '.parquet'
                    leftover['diagnostic'] = audit_parquet((str(path), expected[original]))
                result['leftovers'].append(leftover)
        if args.legacy_index:
            index = json.loads(args.legacy_index.read_text())
            chunks = index['chunks']
            names = [c['filename'] for c in chunks]
            actual = {p.name for p in args.legacy_index.parent.glob('*.bin')}
            legacy = dict(index=str(args.legacy_index), index_sha256=sha256(args.legacy_index),
                          missing=sorted(set(names) - actual), extra=sorted(actual - set(names)),
                          duplicate_names=len(names) != len(set(names)),
                          provenance='No official hashes or source manifest; structural verification only')
            supported = index['config']['item_loader'] == 'TokensLoader' and index['config']['data_format'] == ['no_header_numpy:5']
            if not supported:
                raise ValueError('Unrecognized legacy token layout')
            legacy['chunks'] = list(pool.map(audit_token_chunk, [(str(args.legacy_index.parent / c['filename']), c) for c in chunks]))
            legacy['documents'] = sum(c.get('documents', 0) for c in legacy['chunks'])
            legacy['tokens'] = sum(c.get('tokens', 0) for c in legacy['chunks'])
            legacy['documents_match_raw_10BT'] = legacy['documents'] == result['subsets']['10BT']['rows']
            legacy['status'] = 'passed' if not legacy['missing'] and not legacy['extra'] and not legacy['duplicate_names'] and legacy['documents_match_raw_10BT'] and all(c['status'] == 'passed' for c in legacy['chunks']) else 'failed'
            result['legacy'] = legacy
    result['status'] = 'passed' if all(s['status'] == 'passed' for s in result['subsets'].values()) else 'failed'
    result['finished_at'] = datetime.now().astimezone().isoformat()
    write_json(args.output / 'audit.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'revision', 'finished_at')}), flush=True)
    return result['status'] == 'passed'


def prepared_audit(args):
    raw = json.loads(args.raw_report.read_text())
    assert raw['subsets']['10BT']['status'] == 'passed', 'Raw 10BT audit did not pass'
    manifest = json.loads(args.prepared_manifest.read_text())
    assert manifest['format'] == 'qgdn-u16-v1' and manifest['vocab_size'] == 32000
    verified = {f['path']: f for f in raw['subsets']['10BT']['files']}
    seen = set()
    result = dict(status='running', raw_report_sha256=sha256(args.raw_report),
                  prepared_manifest_sha256=sha256(args.prepared_manifest), splits={})
    for split in ('train', 'val'):
        sources = manifest['sources'][split]
        expected_documents = 0
        for source in sources:
            assert source['path'] not in seen, 'Repeated/overlapping input sources'
            checked = verified[source['path']]
            assert source['sha256'] == checked['sha256'], 'Tokenization used different source bytes'
            expected_documents += checked['decoded_rows']
            seen.add(source['path'])
        info = manifest['splits'][split]
        path = args.prepared_manifest.parent / info['file']
        before = path.stat()
        assert before.st_size == info['tokens'] * 2 and info['tokens'] > 0, 'Truncated token output'
        digest = hashlib.sha256()
        high, eos_count = 0, 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
                tokens = np.frombuffer(block, dtype='<u2')
                high = max(high, int(tokens.max()))
                eos_count += int(np.count_nonzero(tokens == 2))
        assert digest.hexdigest() == info['sha256'], 'Token hash differs from manifest'
        assert high < 32000 and eos_count >= expected_documents, 'Invalid token values or missing EOS tokens'
        assert info['documents'] == expected_documents, 'Tokenization omitted/duplicated source documents'
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
        result['splits'][split] = dict(tokens=info['tokens'], documents=expected_documents, sha256=digest.hexdigest(), max_token=high, eos_count=eos_count)
    assert seen == verified.keys(), 'Not all raw source files were tokenized'
    result.update(status='passed', finished_at=datetime.now().astimezone().isoformat())
    write_json(args.output / 'prepared-audit.json', result)
    print(json.dumps(result), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--subsets', nargs='+', choices=['10BT', '100BT'], default=['10BT', '100BT'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--legacy-index', type=Path)
    parser.add_argument('--prepared-manifest', type=Path)
    parser.add_argument('--raw-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > int(os.environ.get('SLURM_CPUS_PER_TASK', args.workers)):
        parser.error('Worker count must stay within allocated CPUs')
    if args.prepared_manifest:
        if not args.raw_report:
            parser.error('--prepared-manifest requires --raw-report')
    elif not args.root or not args.upstream:
        parser.error('Raw auditing requires --root and --upstream')
    args.output.mkdir(parents=True, exist_ok=False)
    ok = prepared_audit(args) if args.prepared_manifest else raw_audit(args)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
