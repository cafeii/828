#!/usr/bin/env python3
"""按 manifest 从 ModelScope 下载数据集文件到目标目录（服务器侧执行）。

特性：文件级断点续传（已完成则跳过）、.part 临时文件原子落盘、
失败重试、并行下载、结束时校验文件数与总字节数。

用法: python3 download.py [manifest.json] [目标目录] [并行数]
默认: manifest.json 与脚本同目录, 目标 /work/projects/memos-b3/datasets/lzc_rnn/fineweb
中断后直接重跑同一命令即可续传。
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_TARGET = "/work/projects/memos-b3/datasets/lzc_rnn/fineweb"
URL_TMPL = ("https://modelscope.cn/api/v1/datasets/{repo}/repo"
            "?Revision={rev}&FilePath={path}")
MAX_RETRY = 5
CHUNK = 8 << 20  # 8MB


def make_opener() -> urllib.request.OpenerDirector:
    """modelscope.cn 优先直连；直连不通则回退到环境变量中的代理。"""
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct.open("https://modelscope.cn", timeout=10):
            pass
        print("[net] modelscope.cn 直连可用")
        return direct
    except Exception:
        print("[net] 直连失败，回退到环境变量代理")
        return urllib.request.build_opener()


def download_one(opener, repo: str, rev: str, target: Path, path: str,
                 size: int) -> str:
    out = target / path
    if out.exists() and out.stat().st_size == size:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    url = URL_TMPL.format(repo=repo, rev=rev,
                          path=urllib.parse.quote(path))
    for attempt in range(MAX_RETRY):
        try:
            got = tmp.stat().st_size if tmp.exists() else 0
            req = urllib.request.Request(url)
            if got:
                req.add_header("Range", f"bytes={got}-")
            with opener.open(req, timeout=120) as r:
                if got and r.status != 206:
                    # 服务端不支持续传：从头重来，避免追加导致文件损坏
                    got = 0
                with open(tmp, "ab" if got else "wb") as f:
                    while chunk := r.read(CHUNK):
                        f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"大小不符: {tmp.stat().st_size} != {size}")
            tmp.rename(out)
            return "done"
        except Exception as e:
            if attempt == MAX_RETRY - 1:
                raise RuntimeError(f"{path} 下载失败: {e}") from e
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    script_dir = Path(__file__).parent
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "manifest.json"
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_TARGET)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    total = manifest["total_bytes"]
    print(f"[plan] {len(files)} 个文件, 共 {total / 1e9:.1f}GB, "
          f"目标 {target}, 并行 {workers}")

    opener = make_opener()
    done_bytes = 0
    done_files = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(download_one, opener, manifest["repo"],
                        manifest["revision"], target, f["path"], f["size"]): f
            for f in files
        }
        for fut in as_completed(futs):
            f = futs[fut]
            status = fut.result()  # 异常在此抛出，中止整个任务
            done_files += 1
            done_bytes += f["size"]
            if status != "skip" or done_files % 10 == 0:
                elapsed = time.time() - t0
                speed = done_bytes / 1e9 / max(elapsed, 1)
                print(f"[{done_files}/{len(files)}] {f['path']} {status} "
                      f"({done_bytes / 1e9:.1f}GB, {speed:.2f}GB/s)",
                      flush=True)

    # 最终校验：逐文件核对大小
    bad = [f["path"] for f in files
           if not (target / f["path"]).exists()
           or (target / f["path"]).stat().st_size != f["size"]]
    if bad:
        print(f"[error] {len(bad)} 个文件缺失或大小不符: {bad[:5]}", file=sys.stderr)
        sys.exit(1)
    print(f"[ok] 校验通过: {len(files)} 个文件, {total / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
