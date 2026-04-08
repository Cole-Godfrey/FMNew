from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='SafeFlowQ',
        seed=-1,
        max_steps=1000001,
        eval_episodes=10,
        batch_size=2048,
        log_interval=100,
        eval_interval=250000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
        pr_data='data/point_robot-expert-random-100k.hdf5',
    )

    possible_structures = {
        "safe_diffusion": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="SafeDiffusion",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    critic_type="hj",
                    cost_ub=150,
                    beta_schedule='vp',
                    actor_objective="feasibility",
                    sampling_method="ddpm",
                    extract_method="minqc",
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "safe_flow_q": ConfigDict(
            dict(
                pretrain_steps=500000,
                agent_kwargs=dict(
                    model_cls="SafeFlowQ",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    decay_steps=int(2e6),
                    cost_critic_hyperparam=0.9,
                    critic_type="hj",
                    # Flow Q specific
                    hidden_dim=256,
                    time_embed_dim=64,
                    num_categories=3,
                    ode_steps=8,
                    q_samples=2,
                    base_dist_low=-5.0,
                    base_dist_high=5.0,
                    lambda_max=10.0,
                    softplus_beta=1.0,
                    # Safety penalty
                    safety_coef=15,
                    # floq-style variance reduction + conservative usage
                    noise_samples=8,           # repeat batch with multiple z0/t per sample
                    q_conservative_alpha=0.5,  # conservative Q: mean - alpha * std
                    y_norm_clip=5.0,           # clip normalized target for stability
                    bc_coef_unsafe=1.0,        # stronger BC in unsafe/boundary region
                    unsafe_flow_weight=1.0,    # weight unsafe samples in flow loss
                    # For experiment naming
                    sampling_method="flow_q",
                    actor_objective="policy_gradient",
                    extract_method="q_flow",
                    N=1,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "safe_flow_q_diffusion": ConfigDict(
            dict(
                pretrain_steps=500000,
                agent_kwargs=dict(
                    model_cls="SafeFlowQDiffusion",
                    cost_limit=10,
                    env_max_steps=1000,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    decay_steps=int(2e6),
                    cost_critic_hyperparam=0.9,
                    critic_type="qc",
                    # Diffusion actor
                    actor_architecture="ln_resnet",
                    actor_hidden_dims=(256, 256, 256),
                    actor_num_blocks=3,
                    actor_dropout_rate=0.1,
                    actor_layer_norm=True,
                    actor_weight_decay=None,
                    actor_tau=0.005,
                    T=5,
                    N=64,
                    M=0,
                    clip_sampler=True,
                    sampling_method="ddpm",
                    beta_schedule="vp",
                    ddpm_temperature=1.0,
                    # Flow Q specific
                    hidden_dim=256,
                    time_embed_dim=64,
                    num_categories=3,
                    ode_steps=8,
                    q_samples=2,
                    base_dist_low=-5.0,
                    base_dist_high=5.0,
                    lambda_max=10.0,
                    softplus_beta=10.0,
                    # Actor loss (AWR + BC + safety)
                    awr_temperature=5.0,
                    max_weight=100.0,
                    bc_coef=1.0,
                    safety_coef=1.0,
                    # Q normalization
                    q_norm_ema=0.99,
                    # floq-style variance reduction + conservative usage
                    noise_samples=8,
                    q_conservative_alpha=0.5,
                    y_norm_clip=5.0,
                    bc_coef_unsafe=3.0,
                    unsafe_flow_weight=1.0,
                    # For experiment naming
                    actor_objective="awr_diffusion",
                    extract_method="minqc",
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "safe_flow_q_cfm": ConfigDict(
            dict(
                pretrain_steps=500000,
                agent_kwargs=dict(
                    model_cls="SafeFlowQCFM",
                    cost_limit=10,
                    env_max_steps=1000,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    decay_steps=int(2e6),
                    cost_critic_hyperparam=0.9,
                    critic_type="qc",
                    # CFM actor
                    actor_hidden_dims=(256, 256, 256),
                    actor_layer_norm=True,
                    actor_tau=0.005,
                    clip_sampler=True,
                    policy_base_std=1.0,
                    # Flow Q specific
                    hidden_dim=256,
                    time_embed_dim=64,
                    num_categories=3,
                    ode_steps=8,
                    q_samples=2,
                    base_dist_low=-5.0,
                    base_dist_high=5.0,
                    lambda_max=10.0,
                    softplus_beta=30.0,
                    # Actor loss (AWR + BC)
                    awr_temperature=3.0,
                    max_weight=100.0,
                    bc_coef=0.5,
                    # Q normalization
                    q_norm_ema=0.99,
                    # floq-style variance reduction + conservative usage
                    noise_samples=8,
                    q_conservative_alpha=0.5,
                    y_norm_clip=5.0,
                    bc_coef_unsafe=0.0,
                    unsafe_flow_weight=1.0,
                    # For experiment naming
                    sampling_method="cfm",
                    actor_objective="awr_cfm",
                    extract_method="minqc",
                    N=8,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "safe_flow_q_cfm_budget": ConfigDict(
            dict(
                pretrain_steps=500000,
                agent_kwargs=dict(
                    model_cls="SafeFlowQCFMBudget",
                    cost_limit=10,
                    env_max_steps=1000,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    decay_steps=int(2e6),
                    cost_critic_hyperparam=0.9,
                    critic_type="qc",
                    # CFM actor
                    actor_hidden_dims=(256, 256, 256),
                    actor_layer_norm=True,
                    actor_tau=0.005,
                    clip_sampler=True,
                    policy_base_std=1.0,
                    # Flow Q specific
                    hidden_dim=256,
                    time_embed_dim=64,
                    num_categories=3,
                    ode_steps=10,
                    q_samples=2,
                    base_dist_low=-5.0,
                    base_dist_high=5.0,
                    lambda_max=10.0,
                    softplus_beta=30.0,
                    # Actor loss (AWR + BC)
                    awr_temperature=3.0,
                    max_weight=100.0,
                    bc_coef=0.5,
                    # Q normalization
                    q_norm_ema=0.99,
                    # noise_samples = u-diversity × (z0,t)-diversity combined
                    noise_samples=10,
                    q_conservative_alpha=0.5,
                    y_norm_clip=5.0,
                    bc_coef_unsafe=0.0,
                    unsafe_flow_weight=1.0,
                    value_expectile=0.9,
                    safety_penalty=5.0,
                    qc_threshold=1.0,
                    # For experiment naming
                    sampling_method="cfm_budget",
                    actor_objective="awr_cfm_budget",
                    extract_method="minqc",
                    N=8,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
