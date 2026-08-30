# 校验 FineWeb parquet 完整性：并行逐 row group 读 payload，输出坏块清单。
# 背景：download.py 断点续传的 Range 拼接可能产生"大小正确但中段损坏"的文件
# （manifest 无哈希，大小校验发现不了），损坏表现为 pyarrow 读该 row group 时
# thrift 反序列化失败。用法：
#   python3 verify.py <parquet目录> [并行数]
# 退出码：0=全部完好，1=存在坏块（stdout 列出 文件:row_group:错误）。

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def verify_file(filepath: str) -> list[str]:
    import pyarrow.parquet as pq

    bad = []
    try:
        parquet = pq.ParquetFile(filepath)
    except Exception as e:  # 元数据都读不了，整个文件报废
        return [f"{filepath}:METADATA:{type(e).__name__}"]
    for rg in range(parquet.num_row_groups):
        try:
            parquet.read_row_group(rg, columns=["text"])
        except Exception as e:
            bad.append(f"{filepath}:{rg}:{type(e).__name__}")
    return bad


def main() -> None:
    root = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    files = sorted(str(p) for p in root.rglob("*.parquet"))
    assert files, f"{root} 下没有parquet文件"
    print(f"[plan] 校验 {len(files)} 个文件, 并行 {workers}", flush=True)

    all_bad = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for filepath, bad in zip(files, pool.map(verify_file, files)):
            status = "OK" if not bad else f"BAD x{len(bad)}"
            print(f"[{status}] {filepath}", flush=True)
            all_bad.extend(bad)

    if all_bad:
        print("\n坏块清单:")
        for line in all_bad:
            print(f"  {line}")
        sys.exit(1)
    print("全部完好")


if __name__ == "__main__":
    main()
