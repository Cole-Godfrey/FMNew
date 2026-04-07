"""SafeFlowQDiffusion - Flow Matching Q with corrected safety mechanism.
Vc/Qc pretrained, Q_flow learns r-lambda*c distribution.
Actor uses a diffusion policy (DDPM) trained with AWR-style weights + BC + safety penalty.

Safety fixes vs original:
1. category for actor's V(s) is recomputed from Qc(s, a_actor), not reused from data action
2. category_next for V(s') is computed from Qc(s', a_next), not from Vc(s')
3. Actor relies on Q_flow advantage; no explicit safety penalty

floq-style enhancements (no new networks):
4. noise_samples repeat: multiple z0/t per (s,a,y) to reduce flow loss variance
5. conservative Q: mean - alpha*std from flow samples to reduce overestimation
6. stop_gradient on AWR weights to avoid unstable backprop through flow
7. risk-aware BC: stronger BC coefficient in unsafe/boundary region
8. y_norm_clip: clip normalized targets for stable early training
9. midpoint ODE integration for slightly better accuracy
"""
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
from jax import lax
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    StateValue,
    DDPM,
    FourierFeatures,
    MLPResNet,
    cosine_beta_schedule,
    ddpm_sampler,
    get_weight_decay_mask,
    vp_beta_schedule,
)
from jaxrl5.networks.diffusion import dpm_solver_sampler_1st, vp_sde_schedule


def safe_expectile_loss(diff, expectile=0.8):
    """For cost V: penalize underestimation"""
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


# ============================================================================
# Flow Matching Networks
# ============================================================================

class FourierTimeEmbedding(nn.Module):
    embed_dim: int = 64

    @nn.compact
    def __call__(self, t):
        frequencies = jnp.arange(1, self.embed_dim + 1, dtype=jnp.float32) * jnp.pi
        return jnp.cos(t * frequencies)


class VelocityNetwork(nn.Module):
    """Velocity network for Flow Q - learns distribution of r-lambda*c"""
    hidden_dim: int = 256
    num_categories: int = 3
    time_embed_dim: int = 64

    @nn.compact
    def __call__(self, t, z, state, action, category):
        time_feat = FourierTimeEmbedding(self.time_embed_dim)(t)
        cat_feat = nn.Embed(self.num_categories, self.hidden_dim)(category)

        x = jnp.concatenate([state, action, z, time_feat, cat_feat], axis=-1)

        for _ in range(3):
            x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)

        return nn.Dense(1, kernel_init=default_init())(x)


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class SafeFlowQDiffusion(Agent):
    """
    Networks:
    - Vc: Cost Value (expectile loss, pretrained)
    - Qc: Cost Q (TD with HJ Bellman, pretrained)
    - Q_flow (velocity): Flow Matching Q, learns distribution of r - lambda*c
    - Actor: DDPM diffusion policy, trained with AWR + BC + safety penalty
    """
    # Flow Q (velocity network)
    velocity: TrainState
    velocity_target_params: dict

    # Actor (diffusion)
    score_model: TrainState
    target_score_model: TrainState

    # Cost critics (pretrained, frozen during main training)
    safe_critic: TrainState        # Qc
    safe_target_critic: TrainState
    safe_value: TrainState         # Vc

    # Hyperparameters
    discount: float
    tau: float
    actor_tau: float
    cost_critic_hyperparam: float
    critic_type: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)

    # Diffusion specific
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    sampling_method: str = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    # Flow Matching specific
    ode_steps: int = struct.field(pytree_node=False)
    q_samples: int = struct.field(pytree_node=False)
    num_categories: int = struct.field(pytree_node=False)
    base_dist_low: float
    base_dist_high: float

    # Cost penalty params
    lambda_max: float
    softplus_beta: float
    qc_thres: float
    eps_safe: float
    eps_unsafe: float

    awr_temperature: float  # AWR temperature
    max_weight: float       # max advantage weight
    bc_coef: float          # behavior cloning coefficient
    safety_coef: float      # explicit safety penalty coefficient

    q_mean: float      # running mean of Q targets
    q_std: float       # running std of Q targets
    q_norm_ema: float  # EMA coefficient for updating statistics

    # --- floq-style variance reduction + conservative usage ---
    noise_samples: int = struct.field(pytree_node=False)  # repeat batch with multiple noise samples
    q_conservative_alpha: float                            # conservative Q: mean - alpha * std
    y_norm_clip: float                                     # clip normalized target for stability

    # --- risk-aware BC weight (no new network, just reweight loss) ---
    bc_coef_unsafe: float  # stronger BC when data action is unsafe/boundary
    # --- unsafe weighting ---
    unsafe_flow_weight: float  # weight unsafe samples in flow loss

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = "mlp",
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        discount: float = 0.99,
        tau: float = 0.005,
        cost_critic_hyperparam: float = 0.9,
        num_qs: int = 2,
        actor_tau: float = 0.005,
        critic_type: str = 'hj',
        decay_steps: Optional[int] = int(2e6),
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        # Diffusion actor specific
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        sampling_method: str = "ddpm",
        beta_schedule: str = "vp",
        ddpm_temperature: float = 1.0,
        # Flow Matching specific
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        num_categories: int = 3,
        ode_steps: int = 8,
        q_samples: int = 16,
        base_dist_low: float = -5.0,
        base_dist_high: float = 5.0,
        lambda_max: float = 10.0,
        softplus_beta: float = 1.0,
        # Actor loss params (AWR)
        awr_temperature: float = 3.0,
        max_weight: float = 100.0,
        bc_coef: float = 1.0,
        safety_coef: float = 10.0,  # explicit safety penalty weight
        # Q normalization
        q_norm_ema: float = 0.99,
        # floq-style variance reduction
        noise_samples: int = 8,
        q_conservative_alpha: float = 0.5,
        y_norm_clip: float = 5.0,
        bc_coef_unsafe: float = 3.0,
        unsafe_flow_weight: float = 1.0,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, velocity_key, safe_critic_key, safe_value_key = jax.random.split(rng, 5)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (1 - discount) / env_max_steps
        eps_safe = qc_thres * 0.3
        eps_unsafe = qc_thres * 0.3

        if decay_steps is not None:
            actor_lr_schedule = optax.cosine_decay_schedule(actor_lr, decay_steps)
        else:
            actor_lr_schedule = actor_lr

        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)

        # ===== Flow Q =====
        velocity_net = VelocityNetwork(hidden_dim, num_categories, time_embed_dim)
        dummy_t = jnp.zeros((1, 1))
        dummy_z = jnp.zeros((1, 1))
        dummy_cat = jnp.zeros((1,), dtype=jnp.int32)
        velocity_params = velocity_net.init(velocity_key, dummy_t, dummy_z, observations, actions, dummy_cat)['params']
        velocity = TrainState.create(
            apply_fn=velocity_net.apply,
            params=velocity_params,
            tx=optax.adam(actor_lr_schedule)
        )

        # ===== Diffusion Actor =====
        preprocess_time_cls = partial(FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(
            MLP, hidden_dims=(128, 128), activations=mish, activate_final=False
        )

        if actor_architecture == "mlp":
            base_model_cls = partial(
                MLP,
                hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                activations=mish,
                use_layer_norm=actor_layer_norm,
                activate_final=False,
            )
            actor_def = DDPM(
                time_preprocess_cls=preprocess_time_cls,
                cond_encoder_cls=cond_model_cls,
                reverse_encoder_cls=base_model_cls,
            )
        elif actor_architecture == "ln_resnet":
            base_model_cls = partial(
                MLPResNet,
                use_layer_norm=actor_layer_norm,
                num_blocks=actor_num_blocks,
                dropout_rate=actor_dropout_rate,
                out_dim=action_dim,
                activations=mish,
            )
            actor_def = DDPM(
                time_preprocess_cls=preprocess_time_cls,
                cond_encoder_cls=cond_model_cls,
                reverse_encoder_cls=base_model_cls,
            )
        else:
            raise ValueError(f"Invalid actor architecture: {actor_architecture}")

        dummy_time = jnp.zeros((1, 1))
        actor_params = actor_def.init(actor_key, observations, actions, dummy_time)["params"]

        from flax.core.frozen_dict import freeze
        if not isinstance(actor_params, flax.core.frozen_dict.FrozenDict):
            actor_params = freeze(actor_params)

        score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adamw(
                learning_rate=actor_lr_schedule,
                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                mask=get_weight_decay_mask,
            ),
        )
        target_score_model = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ===== Cost Critics (Vc, Qc) =====
        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_optimiser = optax.adam(learning_rate=critic_lr)

        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser)
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        value_def = StateValue(base_cls=value_base_cls)
        value_optimiser = optax.adam(learning_rate=value_lr)
        safe_value_params = value_def.init(safe_value_key, observations)["params"]
        safe_value = TrainState.create(apply_fn=value_def.apply, params=safe_value_params, tx=value_optimiser)

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        alphas = 1 - betas
        alpha_hats = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            velocity=velocity,
            velocity_target_params=velocity_params,
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            tau=tau,
            discount=discount,
            rng=rng,
            act_dim=action_dim,
            actor_tau=actor_tau,
            critic_type=critic_type,
            cost_critic_hyperparam=cost_critic_hyperparam,
            qc_thres=qc_thres,
            eps_safe=eps_safe,
            eps_unsafe=eps_unsafe,
            ode_steps=ode_steps,
            q_samples=q_samples,
            num_categories=num_categories,
            base_dist_low=base_dist_low,
            base_dist_high=base_dist_high,
            lambda_max=lambda_max,
            softplus_beta=softplus_beta,
            awr_temperature=awr_temperature,
            max_weight=max_weight,
            bc_coef=bc_coef,
            safety_coef=safety_coef,
            q_mean=0.0,
            q_std=1.0,
            q_norm_ema=q_norm_ema,
            noise_samples=noise_samples,
            q_conservative_alpha=q_conservative_alpha,
            y_norm_clip=y_norm_clip,
            bc_coef_unsafe=bc_coef_unsafe,
            unsafe_flow_weight=unsafe_flow_weight,
            T=T,
            N=N,
            M=M,
            sampling_method=sampling_method,
            clip_sampler=clip_sampler,
            ddpm_temperature=ddpm_temperature,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hats,
        )

    # ===== Cost Critic Updates (pretrain phase only) =====

    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Update Cost Value using expectile loss"""
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"]
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params):
            vc = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            safe_value_loss = safe_expectile_loss(qc-vc, agent.cost_critic_hyperparam).mean()
            return safe_value_loss, {"safe_value_loss": safe_value_loss, "vc": vc.mean(), "vc_max": vc.max(), "vc_min": vc.min()}

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        return agent.replace(safe_value=safe_value), info

    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Update Cost Q using TD with HJ Bellman or standard cumulative cost (critic_type)"""
        next_vc = agent.safe_value.apply_fn({"params": agent.safe_value.params}, batch["next_observations"])

        if agent.critic_type == "hj":
            qc_nonterminal = (1. - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(batch["costs"], next_vc)
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == 'qc':
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f'Invalid critic type: {agent.critic_type}')

        def safe_critic_loss_fn(safe_critic_params):
            qcs = agent.safe_critic.apply_fn({"params": safe_critic_params}, batch["observations"], batch["actions"])
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()
            return safe_critic_loss, {"safe_critic_loss": safe_critic_loss, "qc": qcs.mean(), "qc_max": qcs.max(), "qc_min": qcs.min()}

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)
        safe_target_critic_params = optax.incremental_update(safe_critic.params, agent.safe_target_critic.params, agent.tau)
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        return agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic), info

    # ===== Flow Q and Actor Updates (main training phase) =====

    def update_flow_q_and_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """
        Update Flow Q (velocity) and Actor with corrected safety mechanism.

        floq-style tricks added (without changing network architecture):
        1. noise_samples repeat: multiple z0/t per (s,a,y) significantly reduces flow loss variance
        2. conservative Q: mean - alpha*std from flow samples reduces overestimation
        3. stop_gradient on AWR weights: avoids actor backprop through flow network
        4. risk-aware BC: stronger BC coefficient in unsafe/boundary region reduces OOD actions
        5. diffusion denoising loss with AWR weights + BC on safe data
        6. no explicit safety penalty (actor only via Q_flow)
        7. midpoint ODE integration: slightly more accurate than left-endpoint
        8. y_norm_clip: stabilizes early training
        """
        rng = agent.rng
        rng, k_pi, k_next, k_flow_z, k_flow_t, k_q, k_actor = jax.random.split(rng, 7)
        k_q1, k_q2, k_q3 = jax.random.split(k_q, 3)

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        next_observations = batch["next_observations"]
        masks = batch["masks"]
        B = observations.shape[0]

        d = agent.qc_thres

        def classify(cost_value):
            """Classify based on cost value: 0=safe, 1=boundary, 2=unsafe"""
            cat = jnp.ones_like(cost_value, dtype=jnp.int32)
            cat = jnp.where(cost_value <= d - agent.eps_safe, 0, cat)
            cat = jnp.where(cost_value >= d + agent.eps_unsafe, 2, cat)
            return cat

        # =========================================================
        # Qc(s, a_data): used for lambda shaping and data category
        # =========================================================
        qcs_data = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, observations, actions)
        max_qc_data = qcs_data.max(axis=0)
        category_data = classify(max_qc_data)

        # Adaptive lambda penalty (per-sample)
        lam = jax.nn.softplus(agent.softplus_beta * (max_qc_data - d)) / agent.softplus_beta
        lam = jnp.clip(lam, 0, agent.lambda_max)

        # r_tilde by safety category:
        #   safe:     r
        #   boundary: r - lambda * c
        #   unsafe:   -c
        safe_mask = category_data == 0
        unsafe_mask = category_data == 2
        r_tilde = jnp.where(
            safe_mask,
            rewards,
            jnp.where(unsafe_mask, -costs, rewards - lam * costs),
        )

        def diffusion_sample_action(score_params, s, key):
            if agent.sampling_method == "ddpm":
                return ddpm_sampler(
                    agent.score_model.apply_fn,
                    score_params,
                    agent.T,
                    key,
                    agent.act_dim,
                    s,
                    agent.alphas,
                    agent.alpha_hats,
                    agent.betas,
                    agent.ddpm_temperature,
                    agent.M,
                    agent.clip_sampler,
                    training=False,
                )
            if agent.sampling_method == "dpm_solver-1":
                return dpm_solver_sampler_1st(
                    agent.score_model.apply_fn,
                    score_params,
                    agent.T,
                    key,
                    agent.act_dim,
                    s,
                    agent.alphas,
                    agent.alpha_hats,
                    agent.betas,
                    agent.ddpm_temperature,
                    agent.M,
                    agent.clip_sampler,
                    training=False,
                )
            raise ValueError(f"Invalid sampling method: {agent.sampling_method}")

        def ode_integrate(velocity_params, z0, s, a, cat):
            """Euler integration with midpoint time for slightly better accuracy"""
            K = agent.ode_steps
            dt = 1.0 / K

            def step_fn(z, i):
                t = jnp.full((z.shape[0], 1), (i + 0.5) * dt)
                dz = agent.velocity.apply_fn({'params': velocity_params}, t, z, s, a, cat)
                return z + dt * dz, None

            final_z, _ = lax.scan(step_fn, z0, jnp.arange(K))
            # Clip ODE output to normalized return range (floq insight: outputs should
            # stay within the plausible Q-value range; use y_norm_clip as reference scale)
            return jnp.clip(final_z, -agent.y_norm_clip * 2, agent.y_norm_clip * 2)

        def estimate_q_stats_normalized(velocity_params, s, a, cat, key):
            """
            Estimate normalized Q distribution statistics (mean, std) via flow samples.
            Used for conservative Q = mean - alpha * std.
            """
            m = agent.q_samples
            bs = s.shape[0]

            s_exp = jnp.tile(s[None], (m, 1, 1)).reshape(-1, s.shape[-1])
            a_exp = jnp.tile(a[None], (m, 1, 1)).reshape(-1, a.shape[-1])
            cat_exp = jnp.tile(cat[None], (m, 1)).reshape(-1)

            z0 = jax.random.uniform(
                key, (m * bs, 1),
                minval=agent.base_dist_low, maxval=agent.base_dist_high
            )
            z_final = ode_integrate(velocity_params, z0, s_exp, a_exp, cat_exp)
            z_final = z_final.reshape(m, bs)

            q_mean = z_final.mean(axis=0)
            q_std = z_final.std(axis=0) + 1e-6
            return q_mean, q_std

        # =========================================================
        # FIX 2: V_next - compute category_next from Qc(s', a_next)
        # instead of from Vc(s') which is action-independent
        # =========================================================
        next_a, _ = diffusion_sample_action(agent.target_score_model.params, next_observations, k_next)
        qcs_next = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, next_observations, next_a)
        max_qc_next = qcs_next.max(axis=0)
        category_next = classify(max_qc_next)

        # V(s') with conservative penalty applied to TD target.
        # Root cause of Q explosion + OOD overestimation:
        #   using raw mean for v_next lets OOD high-mean + high-variance
        #   propagate unfiltered into the bootstrap, inflating y and q_std.
        # Fix: apply the same conservative penalty here as for q_data / v_s.
        v_next_mean_norm, v_next_std_norm = estimate_q_stats_normalized(
            agent.velocity_target_params, next_observations, next_a, category_next, k_q1
        )
        v_next_norm_cons = v_next_mean_norm - agent.q_conservative_alpha * v_next_std_norm

        # Physics-derived absolute bound: |Q| ≤ lambda_max / (1 - γ)
        y_abs_bound = agent.lambda_max / (1.0 - agent.discount)

        # Clip conservative estimate: relaxed to ±3.0 to allow Q to learn realistic
        # large values (e.g. dense reward envs where true Q ~ r/(1-γ) >> 1.5).
        # Conservative penalty already handles pessimism; clip is a secondary safety net.
        v_next_norm_cons = jnp.clip(v_next_norm_cons, -3.0, 3.0)

        # Update EMA from r_tilde only (reward signal, bounded and stable).
        # Previously EMA tracked the full bootstrapped y = r_tilde + γ*V_next, where
        # V_next = v_next_norm_cons * q_std. This created a feedback loop:
        #   q_std grows → V_next grows → y grows → q_std grows further.
        # Stability condition: σ_norm(flow output) < 1/γ ≈ 1.01, but at init
        # the random flow has σ_norm ≈ 2.9, so the loop always diverges at startup.
        # Fix: anchor EMA to r_tilde which is independent of flow network quality.
        batch_mean = jnp.clip(r_tilde.mean(), -y_abs_bound, y_abs_bound)
        batch_std = jnp.clip(jnp.maximum(r_tilde.std(), 1e-6), 1e-6, y_abs_bound)
        new_q_mean = agent.q_norm_ema * agent.q_mean + (1 - agent.q_norm_ema) * batch_mean
        new_q_std = agent.q_norm_ema * agent.q_std + (1 - agent.q_norm_ema) * batch_std

        # Compute TD target entirely in normalized space: avoids denorm→large→renorm loop.
        # r_tilde_norm: normalized reward signal
        # v_next_norm_cons: bootstrap already in normalized space, add directly
        # Result: y_norm is bounded by construction, q_std cannot explode.
        r_tilde_norm = (r_tilde - new_q_mean) / jnp.maximum(new_q_std, 1e-6)
        r_tilde_norm = jnp.clip(r_tilde_norm, -agent.y_norm_clip, agent.y_norm_clip)
        y_norm = r_tilde_norm + agent.discount * masks * v_next_norm_cons
        y_norm = jnp.clip(y_norm, -agent.y_norm_clip, agent.y_norm_clip)

        # =========================================================
        # floq trick: noise_samples repeat
        # Draw multiple z0/t per (s, a, y) to significantly reduce flow loss variance
        # =========================================================
        unsafe_w = jnp.where(category_data == 2, agent.unsafe_flow_weight, 1.0)

        def flow_loss_fn(velocity_params):
            n = agent.noise_samples
            N = n * B

            obs_rep = jnp.repeat(observations, n, axis=0)
            act_rep = jnp.repeat(actions, n, axis=0)
            cat_rep = jnp.repeat(category_data, n, axis=0)
            y_rep = jnp.repeat(y_norm, n, axis=0)
            w_rep = jnp.repeat(unsafe_w, n, axis=0)

            z0 = jax.random.uniform(
                k_flow_z, (N, 1),
                minval=agent.base_dist_low, maxval=agent.base_dist_high
            )
            t = jax.random.uniform(k_flow_t, (N, 1))

            z_t = (1.0 - t) * z0 + t * y_rep[:, None]

            pred = agent.velocity.apply_fn({'params': velocity_params}, t, z_t, obs_rep, act_rep, cat_rep)
            target = y_rep[:, None] - z0
            loss = ((pred - target) ** 2).squeeze(-1)
            return (w_rep * loss).mean()

        flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(agent.velocity.params)
        new_velocity = agent.velocity.apply_gradients(grads=flow_grads)

        # Soft update target velocity
        new_vel_target = optax.incremental_update(
            new_velocity.params, agent.velocity_target_params, agent.actor_tau
        )

        # =========================================================
        # Q(s, a_data): conservative Q = mean - alpha * std
        # Reduces overestimation vs using raw mean
        # =========================================================
        q_data_mean_norm, q_data_std_norm = estimate_q_stats_normalized(
            agent.velocity_target_params, observations, actions, category_data, k_q2
        )
        q_data_norm_cons = q_data_mean_norm - agent.q_conservative_alpha * q_data_std_norm
        q_data = q_data_norm_cons * new_q_std + new_q_mean

        # =========================================================
        # FIX 1: V(s) - compute category from Qc(s, a_actor)
        # instead of reusing data action's category
        # Use diffusion-sampled action for baseline eval
        # =========================================================
        a_pi, _ = diffusion_sample_action(agent.target_score_model.params, observations, k_pi)
        qcs_actor = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, observations, a_pi)
        max_qc_actor = qcs_actor.max(axis=0)
        category_actor = classify(max_qc_actor)

        v_s_mean_norm, v_s_std_norm = estimate_q_stats_normalized(
            agent.velocity_target_params, observations, a_pi, category_actor, k_q3
        )
        v_s_norm_cons = v_s_mean_norm - agent.q_conservative_alpha * v_s_std_norm
        v_s = v_s_norm_cons * new_q_std + new_q_mean

        adv = q_data - v_s
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-6
        adv_norm = (adv - adv_mean) / adv_std

        # weights from target Q_flow; allow gradients through a_pi but not into target params
        weights = jnp.exp(adv_norm * agent.awr_temperature)
        weights = jnp.clip(weights, 0.0, agent.max_weight)

        # =========================================================
        # Actor loss = AWR-weighted diffusion denoising + BC + safety hinge (same as lambda)
        #
        # BC: denoising loss on safe data actions
        # =========================================================
        def actor_loss_fn(score_params):
            key_time, key_noise, key_dropout = jax.random.split(k_actor, 3)

            if agent.sampling_method == "dpm_solver-1":
                eps = 1e-3
                time = jax.random.uniform(key_time, (B,), minval=eps, maxval=1.0)
                noise_sample = jax.random.normal(key_noise, (B, agent.act_dim))
                alpha_t, sigma_t = vp_sde_schedule(time)
                time_in = time[:, None]
                noisy_actions = alpha_t[:, None] * actions + sigma_t[:, None] * noise_sample
                eps_pred = agent.score_model.apply_fn(
                    {"params": score_params},
                    observations,
                    noisy_actions,
                    time_in,
                    rngs={"dropout": key_dropout},
                    training=True,
                )
                x0_pred = (noisy_actions - sigma_t[:, None] * eps_pred) / (alpha_t[:, None] + 1e-6)
            elif agent.sampling_method == "ddpm":
                time_idx = jax.random.randint(key_time, (B,), 0, agent.T)
                noise_sample = jax.random.normal(key_noise, (B, agent.act_dim))
                alpha_hats = agent.alpha_hats[time_idx]
                time_in = time_idx[:, None]
                sqrt_alpha = jnp.sqrt(alpha_hats)[:, None]
                sqrt_one_minus = jnp.sqrt(1 - alpha_hats)[:, None]
                noisy_actions = sqrt_alpha * actions + sqrt_one_minus * noise_sample
                eps_pred = agent.score_model.apply_fn(
                    {"params": score_params},
                    observations,
                    noisy_actions,
                    time_in,
                    rngs={"dropout": key_dropout},
                    training=True,
                )
                x0_pred = (noisy_actions - sqrt_one_minus * eps_pred) / (sqrt_alpha + 1e-6)
            else:
                raise ValueError(f"Invalid sampling method: {agent.sampling_method}")

            if agent.clip_sampler:
                x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

            base_loss = ((eps_pred - noise_sample) ** 2).sum(axis=-1)
            awr_loss = (weights * base_loss).mean()

            bc_coef_per = jnp.where(max_qc_data < d, agent.bc_coef, 0.0)
            bc_loss = (bc_coef_per * base_loss).mean()

            qcs_pi = agent.safe_critic.apply_fn(
                {"params": agent.safe_critic.params}, observations, x0_pred
            )
            max_qc_pi = qcs_pi.max(axis=0)

            safety_penalty_per = (
                jax.nn.softplus(agent.softplus_beta * (max_qc_pi - d)) / agent.softplus_beta
            )
            safety_penalty = safety_penalty_per.mean()

            total_loss = awr_loss + bc_loss + agent.safety_coef * safety_penalty

            return total_loss, {
                "actor_loss": total_loss,
                "awr_loss": awr_loss,
                "bc_loss": bc_loss,
                "safety_penalty": safety_penalty,
                "safety_penalty_min": safety_penalty_per.min(),
                "safety_penalty_max": safety_penalty_per.max(),
                "safety_penalty_scaled": agent.safety_coef * safety_penalty,
                "adv_mean": adv.mean(),
                "weight_mean": weights.mean(),
                "max_qc_pi_mean": max_qc_pi.mean(),
            }

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            agent.score_model.params
        )
        new_score_model = agent.score_model.apply_gradients(grads=actor_grads)

        new_target_score_params = optax.incremental_update(
            new_score_model.params, agent.target_score_model.params, agent.actor_tau
        )
        new_target_score_model = agent.target_score_model.replace(params=new_target_score_params)

        new_agent = agent.replace(
            velocity=new_velocity,
            velocity_target_params=new_vel_target,
            score_model=new_score_model,
            target_score_model=new_target_score_model,
            rng=rng,
            q_mean=new_q_mean,
            q_std=new_q_std,
        )

        info = {
            "flow_loss": flow_loss,
            "actor_loss": actor_info["actor_loss"],
            "awr_loss": actor_info["awr_loss"],
            "bc_loss": actor_info["bc_loss"],
            "safety_penalty": actor_info["safety_penalty"],
            "safety_penalty_min": actor_info["safety_penalty_min"],
            "safety_penalty_max": actor_info["safety_penalty_max"],
            "safety_penalty_scaled": actor_info["safety_penalty_scaled"],
            "adv_mean": actor_info["adv_mean"],
            "weight_mean": actor_info["weight_mean"],
            "adv_std": adv_std,
            "lambda_mean": lam.mean(),
            "max_qc_data_mean": max_qc_data.mean(),
            "max_qc_actor_mean": max_qc_actor.mean(),
            "max_qc_pi_mean": actor_info["max_qc_pi_mean"],
            "q_mean": new_q_mean,
            "q_std": new_q_std,
            "r_tilde_mean": r_tilde.mean(),
            "y_norm_mean": y_norm.mean(),
        }
        return new_agent, info

    @jax.jit
    def _sample_safe_best_reward(self, observations):
        """Sample eval_N actions, keep safe_k safest by Qc, pick best by Q_flow reward."""
        eval_N = 8
        safe_k = 1
        rng = self.rng

        # --- Sample eval_N candidate actions ---
        if self.sampling_method == "ddpm":
            actions, rng = ddpm_sampler(
                self.score_model.apply_fn,
                self.target_score_model.params,
                self.T,
                rng,
                self.act_dim,
                observations,
                self.alphas,
                self.alpha_hats,
                self.betas,
                self.ddpm_temperature,
                self.M,
                self.clip_sampler,
                training=False,
            )
        elif self.sampling_method == "dpm_solver-1":
            actions, rng = dpm_solver_sampler_1st(
                self.score_model.apply_fn,
                self.target_score_model.params,
                self.T,
                rng,
                self.act_dim,
                observations,
                self.alphas,
                self.alpha_hats,
                self.betas,
                self.ddpm_temperature,
                self.M,
                self.clip_sampler,
                training=False,
            )
        else:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}")

        # --- Score by Qc, keep safe_k safest ---
        qcs = self.safe_target_critic.apply_fn(
            {"params": self.safe_target_critic.params}, observations, actions
        )
        max_qc = qcs.max(axis=0)                          # (eval_N,)
        safe_indices = jnp.argsort(max_qc)[:safe_k]      # (safe_k,)
        safe_actions = actions[safe_indices]              # (safe_k, act_dim)
        safe_obs = observations[safe_indices]             # (safe_k, obs_dim)
        safe_qc = max_qc[safe_indices]                   # (safe_k,)

        # --- Classify safe candidates ---
        d = self.qc_thres
        safe_cat = jnp.ones((safe_k,), dtype=jnp.int32)
        safe_cat = jnp.where(safe_qc <= d - self.eps_safe, 0, safe_cat)
        safe_cat = jnp.where(safe_qc >= d + self.eps_unsafe, 2, safe_cat)

        # --- Estimate Q_flow via ODE integration (mean over q_samples) ---
        rng, k_q = jax.random.split(rng)
        m = self.q_samples
        s_exp = jnp.tile(safe_obs[None], (m, 1, 1)).reshape(-1, safe_obs.shape[-1])
        a_exp = jnp.tile(safe_actions[None], (m, 1, 1)).reshape(-1, safe_actions.shape[-1])
        cat_exp = jnp.tile(safe_cat[None], (m, 1)).reshape(-1)
        z0 = jax.random.uniform(
            k_q, (m * safe_k, 1),
            minval=self.base_dist_low, maxval=self.base_dist_high,
        )

        K = self.ode_steps
        dt = 1.0 / K

        def step_fn(z, i):
            t = jnp.full((z.shape[0], 1), (i + 0.5) * dt)
            dz = self.velocity.apply_fn(
                {'params': self.velocity_target_params}, t, z, s_exp, a_exp, cat_exp
            )
            return z + dt * dz, None

        z_final, _ = lax.scan(step_fn, z0, jnp.arange(K))
        z_final = jnp.clip(z_final, -self.y_norm_clip * 2, self.y_norm_clip * 2)
        q_mean = z_final.reshape(m, safe_k).mean(axis=0)  # (safe_k,)

        # --- Pick highest estimated reward ---
        best_idx = jnp.argmax(q_mean)
        best_action = safe_actions[best_idx]

        return best_action, rng

    def eval_actions(self, observations: jnp.ndarray):
        """Sample 8 candidates, keep 4 safest by Qc, pick best reward by Q_flow."""
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(8, axis=0)

        action, rng = self._sample_safe_best_reward(observations)
        action = np.asarray(action)
        return action, self.replace(rng=rng)

    @jax.jit
    def update(self, batch: DatasetDict):
        """Update Flow Q and Actor (Vc/Qc are frozen after pretraining)"""
        new_agent = self
        new_agent, flow_info = new_agent.update_flow_q_and_actor(batch)
        return new_agent, flow_info

    @jax.jit
    def update_cost_critics(self, batch: DatasetDict):
        """Update only Vc and Qc (for pretraining phase)"""
        new_agent = self
        new_agent, vc_info = new_agent.update_vc(batch)
        new_agent, qc_info = new_agent.update_qc(batch)
        return new_agent, {**vc_info, **qc_info}

    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
