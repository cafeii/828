"""Restore missing canonical names without overwriting a concurrent writer's files.

This is a recovery operation, not a final integrity certificate. A separate
downloader renamed the already verified promoted shards during the full audit.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path

from promote_fineweb import (AUTHORIZATION, ORIGINAL, PERSONAL, PRIVATE, UPSTREAM,
                             UPSTREAM_SHA, now, sha256, snapshot, sync_directory,
                             unchanged, verified, write_json)

JOURNAL = PERSONAL / 'experiments/20260831-201300-fineweb-inplace-100bt-46b57e/outputs/promotion/promotion.json'


def restore_missing(source, target, checked):
    unchanged(source, checked)
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError:
        return dict(mode='existing_path_preserved', target=str(target))
    unchanged(target, checked)
    sync_directory(target.parent)
    return dict(mode='restored_missing_path', target=str(target), published=snapshot(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization', required=True, choices=[AUTHORIZATION])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not 1 <= args.workers <= int(os.environ['SLURM_CPUS_PER_TASK']):
        p.error('Run through Slurm with sufficient CPUs')
    output = args.output.absolute()
    if output.resolve() != output or not output.is_relative_to(PERSONAL / 'experiments'):
        raise ValueError('Output outside authorized personal experiments')
    target = ORIGINAL / 'sample/100BT'
    for directory in (target, PRIVATE, PRIVATE / 'sample/100BT'):
        if directory.resolve(strict=True) != directory:
            raise ValueError('Unexpected redirected directory')
    if sha256(UPSTREAM) != UPSTREAM_SHA:
        raise ValueError('Pinned manifest changed')
    expected = {Path(f['path']).name: f for f in json.loads(UPSTREAM.read_text())['subsets']['100BT']['files']}
    journal = json.loads(JOURNAL.read_text())
    if len(journal['files']) != 32 or any(f['mode'] != 'replaced' for f in journal['files'].values()):
        raise ValueError('Unexpected prior publication journal')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='verifying_recovery_sources', started_at=now(), files={},
                  original_full_integrity_confirmed=False, private_cleanup_performed=False,
                  note='Concurrent lzc/rnn downloader prevents a stable canonical final audit.')
    write_json(output / 'recovery.json', report)
    def check(name):
        f = expected[name]
        source = PRIVATE / 'sample/100BT' / name
        checked = verified(source, f['size'], {f['lfs']['oid']})
        unchanged(source, journal['source_stats'][name])
        return name, checked
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checked = list(pool.map(check, sorted(journal['files'])))
    for name, info in checked:
        report['files'][name] = restore_missing(PRIVATE / 'sample/100BT' / name, target / name, info)
        write_json(output / 'recovery.json', report)
    actual = {f.name: f for f in target.glob('*.parquet')}
    report.update(status='paths_restored_awaiting_exclusive_writer', finished_at=now(),
                  formal_files=len(actual), missing=sorted(expected.keys() - actual.keys()),
                  extra=sorted(actual.keys() - expected.keys()),
                  source_commit=os.environ.get('QGDN_EXPECTED_COMMIT'))
    write_json(output / 'recovery.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'formal_files', 'missing', 'extra')}), flush=True)
    if report['missing'] or report['extra']:
        raise RuntimeError('Canonical inventory changed during recovery; preserve all evidence')


if __name__ == '__main__':
    main()
