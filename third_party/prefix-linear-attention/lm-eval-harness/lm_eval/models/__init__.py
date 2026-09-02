from . import huggingface
from . import openai_completions
from . import textsynth
from . import dummy
from . import anthropic_llms
from . import gguf
from . import vllm_causallms
from . import mamba_lm
from . import optimum_lm
from . import neuron_optimum
# PATCH(rnn工作区): based_lm/jrt_lm/local_lm 依赖 based/train/hydra 等本工作区不用的包，
# 守卫以免阻塞 litgpt 模型注册（见 ../patches/PATCHES.md）
try:
    from . import based_lm
except ImportError:
    pass
try:
    from . import jrt_lm
except ImportError:
    pass
try:
    from . import local_lm
except ImportError:
    pass
from . import litgpt_lm  # PATCH(rnn工作区): 自训lit_gpt模型接入，见 ../patches/PATCHES.md
# TODO: implement __all__


import os

try:
    # enabling faster model download
    import hf_transfer

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass
