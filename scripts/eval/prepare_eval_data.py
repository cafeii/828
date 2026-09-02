#!/usr/bin/env python
# 评估数据预下载（在登录节点执行：计算节点无外网）。数据落盘 $HF_HOME（各 run 脚本
# 默认 dataset/eval_data/hf_cache），之后评估作业以 HF_HUB_OFFLINE=1 离线读取。
#
# 同一脚本服务两个 harness（lm_eval 同名不能共存，靠运行环境区分）：
#   std（上游 0.4.9，lzc-rnn 已装）：
#     HF_ENDPOINT=https://hf-mirror.com HF_HOME=$PWD/dataset/eval_data/hf_cache \
#       python scripts/eval/prepare_eval_data.py --suite std
#   jrt（fork，不安装、PYTHONPATH 引入）：
#     HF_ENDPOINT=https://hf-mirror.com HF_HOME=$PWD/dataset/eval_data/hf_cache \
#       PYTHONPATH=third_party/prefix-linear-attention/lm-eval-harness \
#       python scripts/eval/prepare_eval_data.py --suite jrt
import argparse
import sys
from pathlib import Path

_WD = Path(__file__).resolve().parents[2]

# 与 run 脚本任务清单保持一致（docs/experiment.md）
STD_TASKS = "wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,social_iqa_fixed,boolq"
JRT_TASKS = "based_swde,based_fda,based_squad,based_drop,based_nq_2048,based_triviaqa"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["std", "jrt"], required=True)
    args = p.parse_args()

    import evaluate  # noqa: F401 先触发其 transformers 模块对象替换副作用（见 run_lm_eval.py）
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForCausalLM

    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = (STD_TASKS if args.suite == "std" else JRT_TASKS).split(",")
    include = str(_WD / "scripts" / "eval" / "tasks_override") if args.suite == "std" else None
    tm = TaskManager(include_path=include)
    print(f"[prep] suite={args.suite} harness={lm_eval.__file__}", flush=True)

    # get_task_dict 构造 task 对象时即按各自 yaml/类定义下载数据集
    task_dict = get_task_dict(tasks, task_manager=tm)
    for name, t in task_dict.items():
        try:
            t.download()
        except Exception as e:  # 部分版本/任务 download 无需二次调用；构造已下载
            print(f"[prep] {name}.download() skipped: {type(e).__name__}: {e}", flush=True)
    print(f"[ok] {len(task_dict)} tasks prepared: {sorted(task_dict)}")


if __name__ == "__main__":
    main()
