#!/bin/bash
# Train SafeFlowQCFMBudget on Goal1 environments
#
# Usage:
#   cd /home/jerry/Desktop/SafeFlowQ
#   bash scripts/train_cfm_budget.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 4  = OfflineCarGoal1Gymnasium-v0
# 12 = OfflinePointGoal1Gymnasium-v0
ENV_IDS=(0 4)
CONFIG="configs/train_config.py:safe_flow_q_cfm_budget"
DATE=$(date +%Y-%m-%d)
SEEDS=($RANDOM)

for ENV_ID in "${ENV_IDS[@]}"; do
    ENV_NAME=$(python -c "
from env.env_list import env_list
print(env_list[$ENV_ID])")
    GROUP="${ENV_NAME}_cfm_budget_awr_cfm_budget_qc_N8_minqc"

    for SEED in "${SEEDS[@]}"; do
        EXP="${GROUP}_s${SEED}_${DATE}"
        echo ""
        echo "=========================================="
        echo "  EXP  : $EXP"
        echo "  env_id=$ENV_ID  seed=$SEED"
        echo "=========================================="
        python launcher/examples/train_offline.py \
            --env_id=$ENV_ID \
            --config=$CONFIG \
            --config.seed=$SEED \
            --experiment_name="$EXP"
    done
done
