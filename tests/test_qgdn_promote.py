import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/qgdn'))
import promote_fineweb as m
from restore_fineweb_paths import restore_missing


def fixture_files(tmp_path):
    target, source = tmp_path / 'shared.parquet', tmp_path / 'private.parquet'
    target.write_bytes(b'bad-data')
    target.chmod(0o664)
    source.write_bytes(b'new-data')
    source.chmod(0o444)
    backup = tmp_path / 'backup'
    backup.mkdir()
    good_hash = hashlib.sha256(b'new-data').hexdigest()
    bad_hash = hashlib.sha256(b'bad-data').hexdigest()
    good = m.verified(source, 8, {good_hash})
    old = m.verified(target, 8, {bad_hash})
    return target, source, backup, good, old, good_hash


def test_atomic_publication_preserves_open_readers_ownership_and_permissions(tmp_path):
    target, source, backup, good, old, digest = fixture_files(tmp_path)
    with target.open('rb') as reader:
        result = m.promote_one(source, target, backup, good, old, digest)
        assert reader.read() == b'bad-data'
    assert result['mode'] == 'replaced'
    assert target.read_bytes() == source.read_bytes() == b'new-data'
    assert target.stat().st_ino == source.stat().st_ino
    assert m.snapshot(target)['mode'] == 0o664
    assert (backup / 'shared.parquet.bad').read_bytes() == b'bad-data'
    assert not list(backup.glob('*.new'))


def test_unknown_or_changed_shared_data_is_never_overwritten(tmp_path):
    target, source, backup, good, old, digest = fixture_files(tmp_path)
    target.write_bytes(b'colleague-update')
    with pytest.raises(ValueError, match='preserving'):
        m.promote_one(source, target, backup, good, old, digest)
    with pytest.raises(ValueError, match='preserving'):
        m.verified(target, 16, {digest, old['sha256']})
    assert target.read_bytes() == b'colleague-update'
    assert not list(backup.iterdir())


def test_interruption_before_publish_keeps_original_and_repair(tmp_path, monkeypatch):
    target, source, backup, good, old, digest = fixture_files(tmp_path)
    def interrupt(*args):
        raise OSError('simulated rename failure')
    monkeypatch.setattr(m.os, 'replace', interrupt)
    with pytest.raises(OSError, match='simulated'):
        m.promote_one(source, target, backup, good, old, digest)
    assert target.read_bytes() == b'bad-data'
    assert source.read_bytes() == b'new-data'
    assert (backup / 'shared.parquet.bad').read_bytes() == b'bad-data'
    assert not list(backup.glob('*.new'))


def test_symlink_input_is_rejected(tmp_path):
    target, source, backup, good, old, digest = fixture_files(tmp_path)
    target.unlink()
    target.symlink_to(source)
    with pytest.raises(ValueError, match='symlink'):
        m.promote_one(source, target, backup, good, old, digest)
    assert source.read_bytes() == b'new-data'


def test_recovery_never_overwrites_a_concurrent_writers_canonical_file(tmp_path):
    target, source, backup, good, old, digest = fixture_files(tmp_path)
    assert restore_missing(source, target, good)['mode'] == 'existing_path_preserved'
    assert target.read_bytes() == b'bad-data'
    target.unlink()
    assert restore_missing(source, target, good)['mode'] == 'restored_missing_path'
    assert target.read_bytes() == b'new-data'


def test_cleanup_requires_passed_audit_and_does_not_follow_symlinks(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    private = tmp_path / 'private'
    subset = private / 'sample/100BT'
    subset.mkdir(parents=True)
    for name in ['READY.json', '.repair-plan.json', '.repair.lock']:
        (private / name).write_text('{}')
    (shared / 'intact.parquet').write_bytes(b'unchanged')
    (shared / 'repaired.parquet').write_bytes(b'fixed')
    (subset / 'intact.parquet').symlink_to(shared / 'intact.parquet')
    m.os.link(shared / 'repaired.parquet', subset / 'repaired.parquet')
    names = {'intact.parquet', 'repaired.parquet'}
    stats = {'repaired.parquet': m.snapshot(subset / 'repaired.parquet')}
    files = [dict(path=str(shared / n), status='passed', sha256_match=True,
                  bytes=(shared / n).stat().st_size, mtime_ns=(shared / n).stat().st_mtime_ns) for n in names]
    audit = dict(status='failed', subsets={'100BT': dict(status='passed', files=files)})
    with pytest.raises(ValueError, match='audit must pass'):
        m.clean_private(private, shared, names, {'repaired.parquet'}, stats, audit)
    audit['status'] = 'passed'
    unexpected = private / 'do-not-delete.txt'
    unexpected.write_text('preserve')
    with pytest.raises(ValueError, match='Unexpected private root'):
        m.clean_private(private, shared, names, {'repaired.parquet'}, stats, audit)
    assert (subset / 'repaired.parquet').exists()
    unexpected.unlink()
    m.clean_private(private, shared, names, {'repaired.parquet'}, stats, audit)
    assert not private.exists()
    assert (shared / 'intact.parquet').read_bytes() == b'unchanged'
    assert (shared / 'repaired.parquet').read_bytes() == b'fixed'
