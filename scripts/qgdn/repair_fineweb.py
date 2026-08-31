"""Repair audited FineWeb shards into a private overlay; shared sources stay read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def checked_directory(path, allowed_root):
    path = Path(path).resolve()
    allowed_root = Path(allowed_root).resolve(strict=True)
    if path == allowed_root or not path.is_relative_to(allowed_root):
        raise ValueError('Output directory must be strictly inside the personal allowed root')
    return path


def download_verified(url, final, expected_size, expected_sha256, *, attempts=3, timeout=60, backoff=2):
    """A fresh whole-file response per attempt; publish only verified bytes, never overwrite."""
    final = Path(final)
    if final.is_symlink():
        raise ValueError(f'Refusing a symlink download destination: {final}')
    if final.exists():
        if final.stat().st_size != expected_size or sha256(final) != expected_sha256:
            raise ValueError(f'Existing private shard is not verified; preserving it: {final}')
        return dict(status='passed', mode='reused_download', bytes=expected_size, sha256=expected_sha256)
    failures = []
    for attempt in range(1, attempts + 1):
        partial = final.with_name('.' + final.name + '.' + uuid.uuid4().hex + '.download')
        started = last_progress = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent': 'wangzr-qgdn-integrity-repair/1.0',
                'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache'})
            digest, received = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=timeout) as response, partial.open('xb') as stream:
                # No Range or append-based resume: reject unexpected cached partial responses.
                if response.status != 200 or response.headers.get('Content-Range'):
                    raise ValueError(f'Expected a complete HTTP 200 response, got {response.status}')
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != expected_size:
                    raise ValueError(f'HTTP Content-Length differs from pinned manifest: {length}')
                while block := response.read(8 << 20):
                    received += len(block)
                    if received > expected_size:
                        raise ValueError('Download exceeds pinned file size')
                    stream.write(block)
                    digest.update(block)
                    if time.monotonic() - last_progress > 60:
                        print(json.dumps(dict(event='download_progress', file=final.name,
                                              bytes=received, expected_bytes=expected_size)), flush=True)
                        last_progress = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            if received != expected_size or digest.hexdigest() != expected_sha256:
                raise ValueError(f'Download size/SHA-256 mismatch: {received} bytes, {digest.hexdigest()}')
            # link() is atomic and refuses to replace an unexpected existing destination.
            partial.chmod(0o444)
            os.link(partial, final)
            return dict(status='passed', mode='downloaded', bytes=received,
                        sha256=digest.hexdigest(), attempts=attempt,
                        previous_failures=failures, seconds=time.monotonic() - started)
        except Exception as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
            print(json.dumps(dict(event='download_retry', file=final.name,
                                  attempt=attempt, error=failures[-1])), flush=True)
            if attempt < attempts:
                time.sleep(min(backoff * attempt, 30))
        finally:
            # Only remove the unique temporary file created by this attempt.
            partial.unlink(missing_ok=True)
    return dict(status='failed', mode='download_failed', errors=failures)


def reference_verified_source(source, destination, checked):
    source, destination = Path(source), Path(destination)
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (checked['bytes'], checked['mtime_ns']):
        raise ValueError(f'Shared source changed since the previous audit: {source}')
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError('Existing reference points to a different source')
    elif destination.exists():
        raise ValueError('Refusing to replace an existing file with a shared-source reference')
    else:
        destination.symlink_to(source.resolve())
    return dict(status='referenced', mode='shared_source_symlink', source=str(source),
                bytes=checked['bytes'], previous_sha256=checked['sha256'])


def repair(args):
    root = checked_directory(args.destination, args.allowed_root)
    output = checked_directory(args.output, args.allowed_root)
    upstream = json.loads(args.upstream.read_text())
    previous = json.loads(args.raw_report.read_text())
    if upstream['repo_id'] != 'HuggingFaceFW/fineweb' or previous['revision'] != upstream['revision']:
        raise ValueError('Official repository or pinned revision differs from the prior audit')
    if previous['upstream_manifest_sha256'] != sha256(args.upstream):
        raise ValueError('The previous audit used a different upstream manifest')
    expected = {Path(f['path']).name: f for f in upstream['subsets']['100BT']['files']}
    checked = {Path(f['path']).name: f for f in previous['subsets']['100BT']['files']}
    if expected.keys() != checked.keys():
        raise ValueError('Prior audit does not cover every official 100BT source file')
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    lock_path = root / '.repair.lock'
    if lock_path.is_symlink():
        raise ValueError('Repair lock must not be a symlink')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'READY.json').exists():
            raise ValueError('This destination already has a completed repair; do not mutate it')
        plan = dict(revision=upstream['revision'], upstream_sha256=sha256(args.upstream),
                    raw_report_sha256=sha256(args.raw_report), destination=str(root))
        plan_file = root / '.repair-plan.json'
        if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
            raise ValueError('An existing private directory belongs to a different repair plan')
        write_json(plan_file, plan)
        subset = root / 'sample/100BT'
        subset.mkdir(parents=True, exist_ok=True)
        if subset.resolve() != subset:
            raise ValueError('Dataset directory must not redirect through symlinks')
        extra = {p.name for p in subset.glob('*.parquet')} - expected.keys()
        if extra:
            raise ValueError(f'Unexpected formal shards in the private directory: {sorted(extra)}')
        report = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
                      destination=str(root), revision=upstream['revision'], files={},
                      shared_sources_modified=False, plan=plan)
        write_json(output / 'repair.json', report)
        futures = {}
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            for name in sorted(expected):
                official, prior = expected[name], checked[name]
                if prior['expected_sha256'] != official['lfs']['oid'] or prior['expected_bytes'] != official['size']:
                    raise ValueError('Prior file reference differs from the official pinned manifest')
                if prior['status'] == 'passed':
                    report['files'][name] = reference_verified_source(prior['path'], subset / name, prior)
                else:
                    url = f'https://huggingface.co/datasets/{upstream["repo_id"]}/resolve/{upstream["revision"]}/{official["path"]}?download=true'
                    futures[pool.submit(download_verified, url, subset / name, official['size'],
                                        official['lfs']['oid'], attempts=args.attempts)] = name
            for future in as_completed(futures):
                name = futures[future]
                try:
                    report['files'][name] = future.result()
                except Exception as exc:
                    report['files'][name] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
                print(json.dumps(dict(event='download_done', file=name, **report['files'][name])), flush=True)
                write_json(output / 'repair.json', report)
        if any(f['status'] == 'failed' for f in report['files'].values()):
            report['status'] = 'failed'
        else:
            report['status'] = 'auditing_all_150_shards'
            write_json(output / 'repair.json', report)
            audit_dir = output / 'audit'
            command = [sys.executable, '-u', str(Path(__file__).with_name('audit_fineweb.py')),
                       '--root', str(root), '--upstream', str(args.upstream), '--subsets', '100BT',
                       '--workers', str(args.audit_workers), '--output', str(audit_dir)]
            result = subprocess.run(command)
            report['audit_exit_code'] = result.returncode
            report['status'] = 'passed' if result.returncode == 0 else 'failed'
            report['audit_report'] = str(audit_dir / 'audit.json')
            if result.returncode == 0:
                audit = json.loads((audit_dir / 'audit.json').read_text())
                if audit['status'] != 'passed' or audit['subsets']['100BT']['actual_files'] != len(expected):
                    raise ValueError('Final full audit did not confirm every official shard')
                report['audit_report_sha256'] = sha256(audit_dir / 'audit.json')
        report['finished_at'] = datetime.now().astimezone().isoformat()
        write_json(output / 'repair.json', report)
        if report['status'] == 'passed':
            write_json(root / 'READY.json', dict(status='passed', **plan,
                       source_commit=os.environ.get('QGDN_EXPECTED_COMMIT'),
                       verified_at=report['finished_at'], audit_report=report['audit_report'],
                       audit_report_sha256=report['audit_report_sha256'],
                       repair_report=str(output / 'repair.json'), repair_report_sha256=sha256(output / 'repair.json'),
                       note='118 intact shards reference read-only shared sources; 32 repaired shards are private files.'))
        return report['status'] == 'passed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--raw-report', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--allowed-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download-workers', type=int, default=4)
    p.add_argument('--audit-workers', type=int, default=8)
    p.add_argument('--attempts', type=int, default=3)
    args = p.parse_args()
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    if min(args.download_workers, args.audit_workers, args.attempts) < 1 or max(args.download_workers, args.audit_workers) > cpus:
        p.error('Positive worker counts must stay within allocated CPUs')
    raise SystemExit(0 if repair(args) else 1)


if __name__ == '__main__':
    main()
