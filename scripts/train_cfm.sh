#!/bin/bash
# Train SafeFlowQCFM on OfflineCarGoal1Gymnasium-v0
#
# Usage:
#   cd /home/jerry/Desktop/SafeFlowQ
#   bash scripts/train_cfm.sh

set -e
ENV_IDS=(4 5 6 7 16 17)
CONFIG="configs/train_config.py:safe_flow_q_cfm"
DATE=$(date +%Y-%m-%d)
SEEDS=(42)

for ENV_ID in "${ENV_IDS[@]}"; do
    ENV_NAME=$(python -c "
import sys; sys.path.append('.')
from env.env_list import env_list
print(env_list[$ENV_ID])")
    GROUP="${ENV_NAME}_cfm_awr_cfm_qc_N8_minqc"

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
