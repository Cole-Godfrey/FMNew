#!/bin/bash
# Hyperparameter sweep: SafeFlowQDiffusion on OfflineCarGoal1Gymnasium-v0
# Sweep axes: safety_coef (sc), awr_temperature (awt), bc_coef (bc)
# Each config runs with 2 seeds for robustness verification.
#
# Usage:
#   cd /home/jerry/Desktop/SafeFlowQ
#   bash scripts/sweep_reward.sh
#
# Results saved under: results/OfflineCarGoal1Gymnasium-v0_ddpm_awr_diffusion_qc_N64_minqc/<exp_name>/

set -e
ENV_ID=4
CONFIG="configs/train_config.py:safe_flow_q_diffusion"
DATE=$(date +%Y-%m-%d)
GROUP="OfflineCarGoal1Gymnasium-v0_ddpm_awr_diffusion_qc_N64_minqc"
SEEDS=(42 123)

run() {
    local sc=$1
    local awt=$2
    local bc=$3
    for SEED in "${SEEDS[@]}"; do
        local EXP="${GROUP}_sc${sc}_awt${awt}_bc${bc}_s${SEED}_${DATE}"
        echo ""
        echo "=========================================="
        echo "  EXP  : $EXP"
        echo "  safety_coef=$sc  awr_temperature=$awt  bc_coef=$bc  seed=$SEED"
        echo "=========================================="
        python launcher/examples/train_offline.py \
            --env_id=$ENV_ID \
            --config=$CONFIG \
            --config.seed=$SEED \
            --config.agent_kwargs.safety_coef=$sc \
            --config.agent_kwargs.awr_temperature=$awt \
            --config.agent_kwargs.bc_coef=$bc \
            --experiment_name="$EXP"
    done
}

# ── 固定 awt=5.0, bc=0.5，扫 safety_coef ────────────────────────────────────
# 1. 极低安全惩罚 → 最大化 reward 空间
run 0.5  5.0  0.5

# 2. 低安全惩罚
run 1.0  5.0  0.5

# 3. 基准（当前配置）
run 3.0  5.0  0.5

# 4. 中等安全惩罚
run 5.0  5.0  0.5

# ── 固定 sc=3.0, bc=0.5，扫 awr_temperature ─────────────────────────────────
# 5. 低温度：AWR 权重更平滑，策略更稳定
run 3.0  3.0  0.5

# 6. 高温度：AWR 权重更尖锐，更激进地追 high-advantage 动作
run 3.0  7.0  0.5

# 7. 极高温度：非常激进的 reward 优化
run 3.0  10.0 0.5

# ── 固定 sc=3.0, awt=5.0，扫 bc_coef ────────────────────────────────────────
# 8. 极小 BC：policy 几乎不锚定数据，探索更自由
run 3.0  5.0  0.1

# 9. 小 BC
run 3.0  5.0  0.3

# 10. 还原 BC：更保守，贴近数据分布
run 3.0  5.0  1.0
