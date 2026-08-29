#!/usr/bin/env python3
"""从 ModelScope 枚举数据集文件并生成下载清单（manifest）。

用法: python enumerate.py [subset ...]
  默认枚举 sample/10BT 与 sample/100BT。输出 manifest.json 到本目录。
  中断/重跑无副作用，纯只读 API 调用（失败自动重试）。
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "swift/fineweb"
REVISION = "master"
API = f"https://modelscope.cn/api/v1/datasets/{REPO}/repo/tree"
DEFAULT_SUBSETS = ["sample/10BT", "sample/100BT"]
PAGE_SIZE = 1000
MAX_RETRY = 5


def fetch_tree(root: str) -> list[dict]:
    """递归枚举 root 下的全部文件，返回 [{path, size}, ...]。"""
    files, page = [], 1
    while True:
        url = (f"{API}?Revision={REVISION}&Root={root}"
               f"&Recursive=true&PageSize={PAGE_SIZE}&PageNumber={page}")
        for attempt in range(MAX_RETRY):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = json.load(r)
                break
            except Exception as e:
                if attempt == MAX_RETRY - 1:
                    raise RuntimeError(f"枚举 {root} 第 {page} 页失败: {e}") from e
                time.sleep(2 * (attempt + 1))
        batch = [f for f in data["Data"]["Files"] if f["Type"] == "blob"]
        files += [{"path": f["Path"], "size": f["Size"]} for f in batch]
        if page * PAGE_SIZE >= data["Data"]["TotalCount"]:
            return files
        page += 1


def main() -> None:
    subsets = sys.argv[1:] or DEFAULT_SUBSETS
    out = Path(__file__).parent / "manifest.json"
    files = []
    for sub in subsets:
        got = fetch_tree(sub)
        total = sum(f["size"] for f in got)
        print(f"{sub}: {len(got)} 个文件, {total / 1e9:.1f}GB")
        files += got
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "created": datetime.now(timezone.utc).isoformat(),
        "subsets": subsets,
        "total_files": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "files": files,
    }
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"manifest 已写入 {out}: 共 {len(files)} 个文件, "
          f"{manifest['total_bytes'] / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
