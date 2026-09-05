# 训练超参 YAML（--config）解析单测：合并顺序 命令行 > YAML > 内置默认。
# 运行: uv run pytest tests/test_pretrain_config.py -q
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from pretrain import parse_args  # noqa: E402

CONFIG = str(Path(__file__).parent.parent / "scripts" / "train" / "configs" / "lsr300m-main.yaml")


def test_config_file_values():
    args = parse_args(["--config", CONFIG])
    # 与 20260831-2009-lsa300m-main 的 sbatch 逐项一致
    assert args.model_name == "gdn2_lsr_340M"
    assert args.max_tokens == 15000000000
    assert args.global_batch_size == 512 and args.micro_batch_size == 4
    assert args.learning_rate == 4e-4 and args.min_lr_ratio == 0.1
    assert args.weight_decay == 0.1 and args.beta1 == 0.9 and args.beta2 == 0.95
    assert args.grad_clip == 1.0 and args.warmup_tokens is None
    assert args.eval_iters == 50 and args.strategy == "ddp" and args.devices == 8
    assert args.wandb is False


def test_cli_overrides_config():
    args = parse_args(["--config", CONFIG, "--max_tokens", "100000000", "--model_name", "gdn2_gqa_340M"])
    assert args.max_tokens == 100000000
    assert args.model_name == "gdn2_gqa_340M"
    assert args.global_batch_size == 512  # 未覆盖字段保持 YAML 值


def test_cli_only_still_works():
    args = parse_args(["--model_name", "gdn2_gqa_340M", "--exp_name", "x", "--train_data_dir", "/tmp/d"])
    assert args.max_tokens == int(10e9)  # 内置默认
    assert args.strategy == "ddp"
