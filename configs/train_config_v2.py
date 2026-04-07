from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='SafeFlowQ_v2',
        seed=-1,
        max_steps=1000001,
        eval_episodes=20,
        batch_size=2048,
        log_interval=100,
        eval_interval=500000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
        pr_data='data/point_robot-expert-random-100k.hdf5',
    )

    possible_structures = {
        # ============================================================
        # SafeFlowQ v2: Distributional + Safety-Aware AWR
        # ============================================================
        "safe_flow_q_v2": ConfigDict(
            dict(
                pretrain_steps=1000000,
                agent_kwargs=dict(
                    model_cls="SafeFlowQV2",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    decay_steps=int(2e6),
                    cost_critic_hyperparam=0.9,
                    critic_type="hj",

                    # Flow Q specific (v2: more steps/samples)
                    hidden_dim=256,
                    time_embed_dim=64,
                    ode_steps=8,         # v1=8, more steps for accurate ODE integration
                    q_samples=4,         # v1=2, enough samples for distributional estimation
                    base_dist_low=-3.0,   # v1=-2.0, wider base distribution
                    base_dist_high=3.0,   # v1=2.0
                    lambda_max=100.0,
                    softplus_beta=1.0,

                    # Actor loss (v2: no bc_coef, safety built into AWR weights)
                    awr_temperature=3.0,
                    max_weight=100.0,
                    safety_coef=5.0,      # v1=20.0, reduced because safety_factor handles most of it
                    safety_temp=1.0,      # sigmoid temperature for safety_factor

                    # Distributional params (NEW in v2)
                    cvar_alpha=0.25,      # lower 25% quantile for conservative Q
                    tail_alpha=0.75,      # upper 25% for pessimistic safety

                    # For experiment naming
                    cost_scale=25,
                    sampling_method="flow_q_v2",
                    actor_objective="safety_aware_awr",
                    extract_method="cvar",
                    N=1,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),

        # ============================================================
        # Ablation: conservative CVaR (more risk-averse)
        # ============================================================
        "safe_flow_q_v2_conservative": ConfigDict(
            dict(
                pretrain_steps=1000000,
                agent_kwargs=dict(
                    model_cls="SafeFlowQV2",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    decay_steps=int(2e6),
                    cost_critic_hyperparam=0.9,
                    critic_type="hj",

                    hidden_dim=256,
                    time_embed_dim=64,
                    ode_steps=20,
                    q_samples=32,
                    base_dist_low=-3.0,
                    base_dist_high=3.0,
                    lambda_max=100.0,
                    softplus_beta=1.0,

                    awr_temperature=3.0,
                    max_weight=100.0,
                    safety_coef=10.0,     # higher safety penalty
                    safety_temp=0.5,      # sharper safety cutoff

                    cvar_alpha=0.1,       # more conservative: lower 10% quantile
                    tail_alpha=0.9,       # more pessimistic: upper 10% tail

                    cost_scale=25,
                    sampling_method="flow_q_v2",
                    actor_objective="safety_aware_awr",
                    extract_method="cvar_conservative",
                    N=1,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
