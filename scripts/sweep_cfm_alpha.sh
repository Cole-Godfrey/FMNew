#!/bin/bash
# Sweep q_conservative_alpha for SafeFlowQCFM
# alpha in [0.2, 0.5], env 4-7, 2 seeds each
#
# Usage:
#   cd /home/jerry/Desktop/SafeFlowQ
#   bash scripts/sweep_cfm_alpha.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ENV_IDS=(4 6)
CONFIG="configs/train_config.py:safe_flow_q_cfm"
DATE=$(date +%Y-%m-%d)
SEEDS=(42)
ALPHAS=(0.5)

for ENV_ID in "${ENV_IDS[@]}"; do
    ENV_NAME=$(python -c "
from env.env_list import env_list
print(env_list[$ENV_ID])")

    for ALPHA in "${ALPHAS[@]}"; do
        GROUP="${ENV_NAME}_cfm_awr_cfm_qc_N8_minqc"

        for SEED in "${SEEDS[@]}"; do
            EXP="${GROUP}_alpha${ALPHA}_s${SEED}_${DATE}"
            echo ""
            echo "=========================================="
            echo "  EXP  : $EXP"
            echo "  env_id=$ENV_ID  alpha=$ALPHA  seed=$SEED"
            echo "=========================================="
            python launcher/examples/train_offline.py \
                --env_id=$ENV_ID \
                --config=$CONFIG \
                --config.seed=$SEED \
                --config.agent_kwargs.q_conservative_alpha=$ALPHA \
                --experiment_name="$EXP"
        done
    done
done
