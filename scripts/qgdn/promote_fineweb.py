"""One-time, explicitly authorized repair of the original B3 FineWeb 100BT directory.

This does not widen the general experiment workspace policy. It accepts only the
specific shared directory, private repair and pinned manifests authorized here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from repair_fineweb import sha256, write_json

PERSONAL = Path('/work/projects/memos-b3/code/wangzr')
ORIGINAL = Path('/work/projects/memos-b3/datasets/lzc_rnn/fineweb')
PRIVATE = PERSONAL / 'datasets/fineweb-v1.2-repaired-20260831'
UPSTREAM = PERSONAL / 'inputs/fineweb-integrity-20260831/upstream-v1.2.json'
RAW = PERSONAL / 'experiments/20260831-181432-fineweb-audit-100bt-5f1572/outputs/audit/audit.json'
REVISION = 'e31fdfd3918d4b48e837d69d274e624a067d7091'
UPSTREAM_SHA = '91d89a7a8738ce4440d0d892ff14bf03cbdbaa1b34cb1da42ec8c269b8668f87'
RAW_SHA = '299392c2c47205f02a4e4d2b3c00c03acfb9b9ec3d707549476391218029f46d'
READY_AUDIT_SHA = 'bd58aaf2ab81c299e976e882824db95ed92701b81db435662bb25598dd7dbc91'
READY_REPAIR_SHA = 'e20fd96d3823b56caad888eb48c4c3eac5964226c5d7df654ca5b3401d8a625f'
AUTHORIZATION = 'repair-original-100BT-and-remove-private-overlay'
INTERRUPTED_JOURNAL = PERSONAL / 'experiments/20260831-201300-fineweb-inplace-100bt-46b57e/outputs/promotion/promotion.json'


def now():
    return datetime.now().astimezone().isoformat()


def snapshot(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f'Expected a regular file, not a symlink: {path}')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, mode=stat.S_IMODE(s.st_mode), uid=s.st_uid, gid=s.st_gid)


def unchanged(path, before):
    after = snapshot(path)
    # link/chmod operations legitimately change ctime, link count and mode.
    if any(after[k] != before[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'uid', 'gid')):
        raise ValueError(f'File changed; preserving it: {path}')


def observed(path):
    before = snapshot(path)
    digest = sha256(path)
    unchanged(path, before)
    return dict(**before, sha256=digest)


def verified(path, size, digests):
    before = observed(path)
    if before['bytes'] != size:
        raise ValueError(f'Unexpected size; preserving it: {path}')
    digest = before['sha256']
    if digest not in digests:
        raise ValueError(f'Unexpected contents; preserving it: {path}')
    return before


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def promote_one(source, target, backup_dir, source_stat, target_stat, expected_sha):
    """Publish atomically, keeping an old inode for recovery until the final audit."""
    unchanged(source, source_stat)
    unchanged(target, target_stat)
    if target_stat['sha256'] == expected_sha:
        return dict(mode='already_correct', target=str(target))
    if source_stat['device'] != target_stat['device']:
        raise ValueError('Atomic publication requires the same filesystem')
    if (source_stat['uid'], source_stat['gid']) != (target_stat['uid'], target_stat['gid']):
        raise ValueError('Refusing to change shared shard ownership')
    backup = backup_dir / (target.name + '.bad')
    staging = backup_dir / (target.name + '.new')
    os.link(target, backup, follow_symlinks=False)  # Never overwrite an existing backup.
    unchanged(backup, target_stat)
    sync_directory(backup_dir)
    try:
        os.link(source, staging, follow_symlinks=False)
        unchanged(staging, source_stat)
        staging.chmod(target_stat['mode'])
        unchanged(target, target_stat)
        os.replace(staging, target)  # Canonical filename is never missing or partial.
        sync_directory(target.parent)
    finally:
        staging.unlink(missing_ok=True)
    return dict(mode='replaced', target=str(target), backup=str(backup),
                old_sha256=target_stat['sha256'], new_sha256=expected_sha,
                published=snapshot(target))


def private_inventory(root, target, expected, repaired):
    for directory in (root, root / 'sample', root / 'sample/100BT', target):
        if directory.resolve(strict=True) != directory or not directory.is_dir():
            raise ValueError(f'Directory redirects through a symlink: {directory}')
    if {p.name for p in root.iterdir()} != {'.repair.lock', '.repair-plan.json', 'READY.json', 'sample'}:
        raise ValueError('Unexpected private root contents; preserving everything')
    if {p.name for p in (root / 'sample').iterdir()} != {'100BT'}:
        raise ValueError('Unexpected private subsets; preserving everything')
    subset = root / 'sample/100BT'
    if {p.name for p in subset.iterdir()} != set(expected):
        raise ValueError('Unexpected private shards or partials; preserving everything')
    for name in expected:
        path = subset / name
        if name in repaired:
            snapshot(path)
        elif not path.is_symlink() or path.resolve(strict=True) != target / name:
            raise ValueError(f'Unexpected shared-source reference: {path}')
    for name in ('.repair.lock', '.repair-plan.json', 'READY.json'):
        snapshot(root / name)


def clean_private(root, target, expected, repaired, source_stats, audit):
    if audit['status'] != 'passed' or audit['subsets']['100BT']['status'] != 'passed':
        raise ValueError('Canonical full audit must pass before private cleanup')
    private_inventory(root, target, expected, repaired)
    checked = {Path(f['path']).name: f for f in audit['subsets']['100BT']['files']}
    if checked.keys() != set(expected):
        raise ValueError('Canonical audit inventory differs from official inventory')
    for name in expected:
        result = checked[name]
        current = snapshot(target / name)
        if result['status'] != 'passed' or not result['sha256_match']:
            raise ValueError('Canonical shard audit did not pass')
        if (current['bytes'], current['mtime_ns']) != (result['bytes'], result['mtime_ns']):
            raise ValueError('Canonical shard changed after audit')
    for name in repaired:
        path = root / 'sample/100BT' / name
        unchanged(path, source_stats[name])
        # Already-correct canonical files may have a different inode, but their
        # complete official audit above is sufficient to retain the canonical copy.
    # Inventory and all validation finish before the first unlink. No recursive
    # deletion and no symlink traversal: shared canonical files remain in place.
    for name in sorted(expected):
        (root / 'sample/100BT' / name).unlink()
    (root / 'sample/100BT').rmdir()
    (root / 'sample').rmdir()
    for name in ('.repair.lock', '.repair-plan.json', 'READY.json'):
        (root / name).unlink()
    root.rmdir()


def prior_cleanup_inventory(target, journal, source_stats):
    """Only collect known original backups and renamed references to our good inodes."""
    backup = Path(journal['backup_directory'])
    if backup.parent != target or not backup.name.startswith('.qgdn-repair-backup-'):
        raise ValueError('Previous backup is outside the exact shared dataset directory')
    if backup.resolve(strict=True) != backup:
        raise ValueError('Previous backup directory redirects')
    expected = {name + '.bad' for name in journal['files']}
    if {p.name for p in backup.iterdir()} != expected:
        raise ValueError('Unexpected prior backup files; preserve them')
    cleanup = []
    for name in sorted(journal['files']):
        path = backup / (name + '.bad')
        before = journal['target_stats'][name]
        unchanged(path, before)
        cleanup.append((path, before))
        renamed = target / (name + '.corrupt')
        if renamed.exists() or renamed.is_symlink():
            # These are the very same verified repaired inodes that were renamed
            # by the competing task. Never remove an unrelated historical file.
            unchanged(renamed, source_stats[name])
            cleanup.append((renamed, source_stats[name]))
    return backup, cleanup


def run(args):
    if args.authorization != AUTHORIZATION:
        raise ValueError('This one-time shared-directory exception must be explicit')
    output = args.output.absolute()
    if output.resolve() != output or not output.is_relative_to(PERSONAL / 'experiments'):
        raise ValueError('Reports must remain in the personal experiments directory')
    target = ORIGINAL / 'sample/100BT'
    if target.resolve(strict=True) != target or PRIVATE.resolve(strict=True) != PRIVATE:
        raise ValueError('Authorized directories must not redirect')
    for path, digest in ((UPSTREAM, UPSTREAM_SHA), (RAW, RAW_SHA)):
        if sha256(path) != digest:
            raise ValueError(f'Pinned evidence changed: {path}')
    upstream, raw = json.loads(UPSTREAM.read_text()), json.loads(RAW.read_text())
    ready = json.loads((PRIVATE / 'READY.json').read_text())
    if ready['status'] != 'passed' or ready['destination'] != str(PRIVATE):
        raise ValueError('Private repair is not ready')
    if ready['revision'] != REVISION or upstream['revision'] != REVISION:
        raise ValueError('Wrong official revision')
    for key, digest in (('audit_report', READY_AUDIT_SHA), ('repair_report', READY_REPAIR_SHA)):
        if sha256(Path(ready[key])) != digest or ready[key + '_sha256'] != digest:
            raise ValueError('Completed private repair evidence changed')
    expected = {Path(f['path']).name: f for f in upstream['subsets']['100BT']['files']}
    prior = {Path(f['path']).name: f for f in raw['subsets']['100BT']['files']}
    repaired = {name for name, f in prior.items() if f['status'] == 'failed'}
    if len(expected) != 150 or len(repaired) != 32 or expected.keys() != prior.keys():
        raise ValueError('Unexpected pinned repair inventory')
    private_inventory(PRIVATE, target, expected, repaired)
    if {p.name for p in target.glob('*.parquet')} != set(expected):
        raise ValueError('Canonical inventory changed')
    interrupted = None
    if args.after_concurrent_download:
        interrupted = json.loads(INTERRUPTED_JOURNAL.read_text())
        if interrupted['target'] != str(target) or set(interrupted['files']) != repaired:
            raise ValueError('Interrupted repair journal has a different target or inventory')
        if any(f['mode'] != 'replaced' for f in interrupted['files'].values()):
            raise ValueError('Interrupted repair did not publish all known 32 files')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='preflight', started_at=now(), target=str(target),
                  private_root=str(PRIVATE), revision=REVISION, source_commit=os.environ.get('QGDN_EXPECTED_COMMIT'),
                  user_authorization='直接修复原目录，不要把语料存在我的目录下面',
                  upstream_sha256=UPSTREAM_SHA, original_audit_sha256=RAW_SHA,
                  after_concurrent_download=args.after_concurrent_download,
                  files={}, private_removed=False, backups_removed=False)
    write_json(output / 'promotion.json', report)
    write_json(output / 'prior-private-READY.json', ready)
    # Locks coordinate our own utilities; concurrent third-party writers are also
    # detected by hashes and before/after metadata checks, never silently accepted.
    shared_lock = target / '.qgdn-100bt-repair.lock'
    fd = os.open(shared_lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o664)
    with os.fdopen(fd, 'a+') as lock, (PRIVATE / '.repair.lock').open('r+') as private_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(private_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backup = target / ('.qgdn-repair-backup-' + output.parent.parent.name)
        try:
            def check(name):
                f = expected[name]
                good = verified(PRIVATE / 'sample/100BT' / name, f['size'], {f['lfs']['oid']})
                if interrupted:
                    unchanged(PRIVATE / 'sample/100BT' / name, interrupted['source_stats'][name])
                    # User authorized finishing the repair after the redundant
                    # writer ended. Preserve and journal its newly observed bytes
                    # before any replacement; still refuse subsequent changes.
                    old = observed(target / name)
                else:
                    old = verified(target / name, f['size'], {prior[name]['sha256'], f['lfs']['oid']})
                print(json.dumps(dict(event='preflight_verified', file=name,
                                      canonical_sha256_match=old['sha256'] == f['lfs']['oid'])), flush=True)
                return name, good, old
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                checks = list(pool.map(check, sorted(repaired)))
            source_stats = {name: good for name, good, old in checks}
            target_stats = {name: old for name, good, old in checks}
            prior_backup, prior_cleanup = prior_cleanup_inventory(target, interrupted, source_stats) if interrupted else (None, [])
            report.update(status='publishing', source_stats=source_stats, target_stats=target_stats,
                          backup_directory=str(backup),
                          prior_cleanup=[dict(path=str(path), before=before) for path, before in prior_cleanup])
            write_json(output / 'promotion.json', report)
            backup.mkdir(exist_ok=False)
            for name, good, old in checks:
                report['files'][name] = promote_one(PRIVATE / 'sample/100BT' / name, target / name,
                    backup, good, old, expected[name]['lfs']['oid'])
                write_json(output / 'promotion.json', report)
                print(json.dumps(dict(event='published', file=name)), flush=True)
            report['status'] = 'auditing_original_150_shards'
            write_json(output / 'promotion.json', report)
            command = [sys.executable, '-u', str(Path(__file__).with_name('audit_fineweb.py')),
                       '--root', str(ORIGINAL), '--upstream', str(UPSTREAM), '--subsets', '100BT',
                       '--workers', str(args.workers), '--output', str(output / 'audit')]
            subprocess.run(command, check=True)
            audit_path = output / 'audit/audit.json'
            audit = json.loads(audit_path.read_text())
            report.update(status='cleaning_verified_private_copy', audit_report=str(audit_path),
                          audit_report_sha256=sha256(audit_path))
            write_json(output / 'promotion.json', report)
            for path, before in prior_cleanup:
                unchanged(path, before)
            clean_private(PRIVATE, target, expected, repaired, source_stats, audit)
            report['private_removed'] = not PRIVATE.exists()
            write_json(output / 'promotion.json', report)
            # Delete only our own known-bad temporary backup links after success.
            for name, item in report['files'].items():
                if item['mode'] == 'replaced':
                    unchanged(backup / (name + '.bad'), target_stats[name])
            for name, item in report['files'].items():
                if item['mode'] == 'replaced':
                    (backup / (name + '.bad')).unlink()
            backup.rmdir()
            for path, before in prior_cleanup:
                unchanged(path, before)
                path.unlink()
            if prior_backup:
                prior_backup.rmdir()
            report.update(status='passed', backups_removed=True, finished_at=now(),
                          prior_backups_removed=bool(interrupted),
                          initial_repaired_files=32,
                          additional_replacements_after_concurrent_download=sum(f['mode'] == 'replaced' for f in report['files'].values()) if interrupted else 0,
                          bytes=audit['subsets']['100BT']['bytes'], documents=audit['subsets']['100BT']['rows'],
                          row_groups=audit['subsets']['100BT']['row_groups'], verified_files=len(expected))
            sync_directory(target)
        except BaseException as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=now(),
                          recovery_note='Preserve the journal and any remaining backup/private files. '
                                        'Published canonical shards contain verified official bytes; do not blindly rerun.')
            write_json(output / 'promotion.json', report)
            raise
        write_json(output / 'promotion.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'target', 'private_removed', 'backups_removed', 'verified_files')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization', required=True, choices=[AUTHORIZATION])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--after-concurrent-download', action='store_true',
                   help='Resume the exact interrupted repair after the redundant external writer has ended')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run this data operation through Slurm, not on the login node')
    if not 1 <= args.workers <= int(os.environ['SLURM_CPUS_PER_TASK']):
        p.error('Workers must fit the Slurm CPU allocation')
    run(args)


if __name__ == '__main__':
    main()
