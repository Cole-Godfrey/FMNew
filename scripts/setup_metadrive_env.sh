#!/bin/bash

set -euo pipefail

# Installs the packages required for DSRL MetaDrive environments in the
# current Python environment.

python -m pip install "gym>=0.26,<0.27"
python -m pip install dsrl
python -m pip install git+https://github.com/HenryLHH/metadrive_clean.git@main

cat <<'EOF'

MetaDrive dependencies installed.

You can now launch MetaDrive training, for example:
  export XLA_PYTHON_CLIENT_PREALLOCATE=False
  python launcher/examples/train_offline.py \
      --env_id=30 \
      --config=configs/train_config.py:safe_flow_q_cfm_budget

EOF
