#!/usr/bin/env bash
set -euo pipefail

python -m pip install -q --upgrade pip
python -m pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install -q --no-deps "trl>=0.12,<0.16" peft accelerate bitsandbytes
python -m pip install -q -r requirements.txt
