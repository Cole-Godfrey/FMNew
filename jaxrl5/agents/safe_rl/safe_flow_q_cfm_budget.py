"""SafeFlowQCFMBudget - SafeFlowQCFM with budget-conditioned policy and Q_flow.

Key improvement over SafeFlowQCFM:
- Introduces u ∈ [0, cost_limit]: accumulated cost already spent in the episode (raw, not normalised).
- For each (s, a) transition, samples u_samples values of u ~ Uniform(0, cost_limit).
- Policy and Q_flow are conditioned on (s, u), turning the problem into an augmented-state
  unconstrained MDP where the budget is part of the context.
- Safety classification uses a hard budget check AND a fixed Qc threshold:
    Cat 0 (safe):    u + c < cost_limit  AND  Qc(s,a) < qc_threshold  → r_tilde = r
    Cat 2 (unsafe):  u + c >= cost_limit  OR   Qc(s,a) >= qc_threshold → r_tilde = -safety_penalty * c
    Cat 1 (border):  never (eps = 0)
- At eval time, the actual accumulated episode cost u_t is passed to eval_actions,
  enabling the policy to adapt its behavior as the budget is consumed.
"""
import os
import pickle
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from jax import lax

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue


# ============================================================
# Helpers
# ============================================================

def safe_expectile_loss(diff, expectile=0.8):
    """For cost V: penalize underestimation."""
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def classify_budget(u, c, qc, cost_limit, qc_threshold):
    """Safety classification: both hard budget check and fixed Qc threshold.

    Safe iff:   u + c < cost_limit  AND  qc < qc_threshold
    Unsafe:     u + c >= cost_limit  OR   qc >= qc_threshold

    Args:
        u:            accumulated cost so far (shape: B), raw scale
        c:            single-step cost of current transition (shape: B), raw scale
        qc:           Qc(s, a) estimate (shape: B), raw scale
        cost_limit:   episode budget (scalar), raw scale
        qc_threshold: fixed Qc safety threshold (scalar), raw scale
    Returns:
        category (int32, shape: B): 0=safe, 2=unsafe
    """
    is_unsafe = (u + c >= cost_limit) | (qc >= qc_threshold)
    return jnp.where(is_unsafe, 2, 0).astype(jnp.int32)


def classify_qc_with_budget(qc, qc_threshold):
    """Qc classification for next state using fixed threshold.

    Used when single-step cost at t+1 is unknown (next state bootstrap).
    """
    return jnp.where(qc >= qc_threshold, 2, 0).astype(jnp.int32)


# ============================================================
# Networks
# ============================================================

class FourierTimeEmbedding(nn.Module):
    embed_dim: int = 64

    @nn.compact
    def __call__(self, t):
        frequencies = jnp.arange(1, self.embed_dim + 1, dtype=jnp.float32) * jnp.pi
        return jnp.cos(t * frequencies)


class VelocityNetwork(nn.Module):
    """Flow-Q velocity network conditioned on (s, a, u, cat).

    u: accumulated episode cost so far (raw value, not normalised).
    """
    hidden_dim: int = 256
    num_categories: int = 3
    time_embed_dim: int = 64

    @nn.compact
    def __call__(self, t, z, state, action, u, category):
        time_feat = FourierTimeEmbedding(self.time_embed_dim)(t)
        cat_feat  = nn.Embed(self.num_categories, self.hidden_dim)(category)
        # u: (B, 1), raw accumulated cost
        x = jnp.concatenate([state, action, z, time_feat, cat_feat, u], axis=-1)

        for _ in range(3):
            x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)

        return nn.Dense(1, kernel_init=default_init())(x)


class PolicyFlowNetwork(nn.Module):
    """Conditional flow matching policy conditioned on (s, u).

    u: accumulated episode cost so far (raw value, not normalised).
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256, 256)
    time_embed_dim: int = 64
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, t, z, state, u):
        time_feat = FourierTimeEmbedding(self.time_embed_dim)(t)
        # u: (B, 1), raw accumulated cost
        x = jnp.concatenate([state, z, time_feat, u], axis=-1)

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=default_init())(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.silu(x)

        return nn.Dense(self.action_dim, kernel_init=default_init())(x)


class RewardValueNetwork(nn.Module):
    """IQL-style state value function V(s, u).

    Trained via expectile regression against Q_flow estimates.
    Avoids action sampling for advantage computation.
    """
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, state, u):
        # u: (B, 1), raw accumulated cost
        x = jnp.concatenate([state, u], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(1, kernel_init=default_init())(x).squeeze(-1)


# ============================================================
# Agent
# ============================================================

class SafeFlowQCFMBudget(Agent):
    """
    Networks:
    - Vc: Cost value (expectile loss, pretrained)
    - Qc: Cost Q (HJ Bellman or cumulative cost, pretrained)
    - Q_flow (velocity): flow matching critic over scalar penalized return, conditioned on u
    - Policy flow: conditional flow matching actor over actions, conditioned on u
    """

    # Flow Q
    velocity: TrainState
    velocity_target_params: dict

    # Actor (conditional flow matching policy)
    policy_flow: TrainState
    target_policy_flow: TrainState

    # Cost critics
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState

    # IQL-style reward value V(s, u)
    reward_value: TrainState

    # Hyperparameters
    discount: float
    tau: float
    actor_tau: float
    cost_critic_hyperparam: float
    critic_type: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)

    # Flow-specific
    ode_steps: int = struct.field(pytree_node=False)
    q_samples: int = struct.field(pytree_node=False)
    num_categories: int = struct.field(pytree_node=False)
    base_dist_low: float
    base_dist_high: float
    policy_base_std: float

    # Cost penalty params
    lambda_max: float
    softplus_beta: float
    # cost_limit: raw episode budget (e.g. 10). For qc critic, Qc and batch["costs"]
    # are also in raw scale, so all budget comparisons are consistent.
    cost_limit: float
    qc_threshold: float

    # Actor / weighting
    awr_temperature: float
    max_weight: float
    bc_coef: float
    q_mean: float
    q_std: float
    q_norm_ema: float

    # FloQ-style variance reduction + conservative usage
    noise_samples: int = struct.field(pytree_node=False)
    q_conservative_alpha: float
    y_norm_clip: float

    # Risk-aware BC and unsafe weighting
    bc_coef_unsafe: float
    unsafe_flow_weight: float
    clip_sampler: bool = struct.field(pytree_node=False)

    # IQL value expectile
    value_expectile: float

    # Penalty scale for Cat2 (unsafe) samples in r_tilde
    safety_penalty: float



    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        actor_layer_norm: bool = False,
        discount: float = 0.99,
        tau: float = 0.005,
        cost_critic_hyperparam: float = 0.9,
        num_qs: int = 2,
        actor_tau: float = 0.005,
        critic_type: str = "hj",
        decay_steps: Optional[int] = int(2e6),
        cost_limit: float = 10.0,
        env_max_steps: int = 1000,
        # Flow matching specific
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        num_categories: int = 3,
        ode_steps: int = 8,
        q_samples: int = 16,
        base_dist_low: float = -5.0,
        base_dist_high: float = 5.0,
        policy_base_std: float = 1.0,
        lambda_max: float = 10.0,
        softplus_beta: float = 1.0,
        # Actor loss params (AWR-style weighted CFM)
        awr_temperature: float = 3.0,
        max_weight: float = 100.0,
        bc_coef: float = 1.0,
        # Q normalization
        q_norm_ema: float = 0.99,
        # FloQ-style variance reduction
        noise_samples: int = 8,
        q_conservative_alpha: float = 0.5,
        y_norm_clip: float = 5.0,
        bc_coef_unsafe: float = 3.0,
        unsafe_flow_weight: float = 1.0,
        clip_sampler: bool = True,
        value_expectile: float = 0.7,
        safety_penalty: float = 5.0,
        qc_threshold: float = 1.0,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, velocity_key, safe_critic_key, safe_value_key = jax.random.split(rng, 5)
        actions     = action_space.sample()
        observations = observation_space.sample()
        action_dim  = action_space.shape[0]

        if decay_steps is not None:
            actor_lr_schedule = optax.cosine_decay_schedule(actor_lr, decay_steps)
        else:
            actor_lr_schedule = actor_lr

        observations = jnp.expand_dims(observations, axis=0)
        actions      = jnp.expand_dims(actions,      axis=0)

        # ===== Flow Q (velocity network) =====
        velocity_net = VelocityNetwork(hidden_dim, num_categories, time_embed_dim)
        dummy_t        = jnp.zeros((1, 1))
        dummy_z_scalar = jnp.zeros((1, 1))
        dummy_cat      = jnp.zeros((1,), dtype=jnp.int32)
        dummy_u        = jnp.zeros((1, 1))   # raw u = 0
        velocity_params = velocity_net.init(
            velocity_key, dummy_t, dummy_z_scalar, observations, actions, dummy_u, dummy_cat
        )["params"]
        velocity = TrainState.create(
            apply_fn=velocity_net.apply,
            params=velocity_params,
            tx=optax.adam(actor_lr_schedule),
        )

        # ===== Policy Flow (conditional FM actor) =====
        policy_def = PolicyFlowNetwork(
            action_dim=action_dim,
            hidden_dims=tuple(actor_hidden_dims),
            time_embed_dim=time_embed_dim,
            use_layer_norm=actor_layer_norm,
        )
        dummy_z_action = jnp.zeros((1, action_dim))
        policy_params  = policy_def.init(actor_key, dummy_t, dummy_z_action, observations, dummy_u)["params"]
        policy_flow = TrainState.create(
            apply_fn=policy_def.apply,
            params=policy_params,
            tx=optax.adamw(learning_rate=actor_lr_schedule, weight_decay=0.0),
        )
        target_policy_flow = TrainState.create(
            apply_fn=policy_def.apply,
            params=policy_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # ===== Cost Critics (Vc, Qc) =====
        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls      = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def      = Ensemble(critic_cls, num=num_qs)
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        value_def      = StateValue(base_cls=value_base_cls)
        value_optimiser = optax.adam(learning_rate=value_lr)
        safe_value_params = value_def.init(safe_value_key, observations)["params"]
        safe_value = TrainState.create(
            apply_fn=value_def.apply, params=safe_value_params, tx=value_optimiser
        )

        # ===== Reward Value V(s, u) =====
        reward_value_def    = RewardValueNetwork(hidden_dims=tuple(critic_hidden_dims))
        reward_value_params = reward_value_def.init(safe_value_key, observations, dummy_u)["params"]
        reward_value = TrainState.create(
            apply_fn=reward_value_def.apply,
            params=reward_value_params,
            tx=optax.adam(learning_rate=value_lr),
        )

        return cls(
            actor=None,
            velocity=velocity,
            velocity_target_params=velocity_params,
            policy_flow=policy_flow,
            target_policy_flow=target_policy_flow,
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
            # cost_limit in raw scale; all budget quantities use the same raw scale.
            cost_limit=cost_limit,
            qc_threshold=qc_threshold,
            ode_steps=ode_steps,
            q_samples=q_samples,
            num_categories=num_categories,
            base_dist_low=base_dist_low,
            base_dist_high=base_dist_high,
            policy_base_std=policy_base_std,
            lambda_max=lambda_max,
            softplus_beta=softplus_beta,
            awr_temperature=awr_temperature,
            max_weight=max_weight,
            bc_coef=bc_coef,
            q_mean=0.0,
            q_std=1.0,
            q_norm_ema=q_norm_ema,
            noise_samples=noise_samples,
            q_conservative_alpha=q_conservative_alpha,
            y_norm_clip=y_norm_clip,
            bc_coef_unsafe=bc_coef_unsafe,
            unsafe_flow_weight=unsafe_flow_weight,
            clip_sampler=clip_sampler,
            value_expectile=value_expectile,
            safety_penalty=safety_penalty,
            reward_value=reward_value,
        )

    # ============================================================
    # Cost Critic Updates (pretrain phase only)
    # ============================================================

    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params}, batch["observations"], batch["actions"]
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params):
            vc = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            safe_value_loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()
            return safe_value_loss, {
                "safe_value_loss": safe_value_loss,
                "vc": vc.mean(),
                "vc_max": vc.max(),
                "vc_min": vc.min(),
            }

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value  = agent.safe_value.apply_gradients(grads=grads)
        return agent.replace(safe_value=safe_value), info

    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )

        if agent.critic_type == "hj":
            qc_nonterminal = (1.0 - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(
                batch["costs"], next_vc
            )
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == "qc":
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f"Invalid critic type: {agent.critic_type}")

        def safe_critic_loss_fn(safe_critic_params):
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()
            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        return agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic), info

    # ============================================================
    # Main update: Flow Q + CFM actor (with budget conditioning)
    # ============================================================

    def update_flow_q_and_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        rng, k_pi, k_next, k_flow_z, k_flow_t, k_q, k_actor, k_u = jax.random.split(rng, 8)
        k_q1, k_q2, k_q3 = jax.random.split(k_q, 3)
        k_actor_z, k_actor_t = jax.random.split(k_actor, 2)

        observations      = batch["observations"]
        actions           = batch["actions"]
        rewards           = batch["rewards"]
        costs             = batch["costs"]
        next_observations = batch["next_observations"]
        masks             = batch["masks"]
        B = observations.shape[0]

        # ----------------------------------------------------------
        # Single expansion: noise_samples serves as both u-diversity
        # and (z0, t) diversity — each of the N samples gets a fresh
        # independent draw of (u, z0, t).
        # ----------------------------------------------------------
        n = agent.noise_samples
        N = n * B

        obs_exp      = jnp.repeat(observations,      n, axis=0)   # (N, obs_dim)
        act_exp      = jnp.repeat(actions,           n, axis=0)   # (N, act_dim)
        costs_exp    = jnp.repeat(costs,             n, axis=0)   # (N,)
        rewards_exp  = jnp.repeat(rewards,           n, axis=0)   # (N,)
        masks_exp    = jnp.repeat(masks,             n, axis=0)   # (N,)
        next_obs_exp = jnp.repeat(next_observations, n, axis=0)   # (N, obs_dim)

        # Sample u uniformly over a wider raw budget range to cover over-budget states.
        u_exp = jax.random.uniform(k_u, (N,), minval=5.0, maxval=20.0)
        u_col = u_exp[:, None]   # (N, 1)

        # ----------------------------------------------------------
        # Budget-aware classification
        # Qc(s,a) does not depend on u → compute once on B, then repeat.
        # ----------------------------------------------------------
        qcs_b      = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params}, observations, actions
        )
        max_qc_b   = qcs_b.max(axis=0)                              # (B,)
        max_qc_exp = jnp.repeat(max_qc_b, n, axis=0)               # (N,)
        # Classify: safe iff u+c < cost_limit AND Qc < qc_threshold
        category_exp = classify_budget(
            u_exp, costs_exp, max_qc_exp, agent.cost_limit, agent.qc_threshold
        )

        # λ shaping (unused for Cat1=never, kept for border-zone r_tilde formula)
        lam_exp  = jax.nn.softplus(agent.softplus_beta * (max_qc_exp - agent.qc_threshold)) / agent.softplus_beta
        lam_exp  = jnp.clip(lam_exp, 0, agent.lambda_max)
        safe_mask   = category_exp == 0
        unsafe_mask = category_exp == 2
        r_tilde_exp = jnp.where(
            safe_mask, rewards_exp,
            jnp.where(unsafe_mask, -100* costs_exp, rewards_exp - lam_exp * costs_exp)
        )

        # u_next = clip(u + c, 0, cost_limit) for bootstrap
        u_next_col = jnp.clip(u_exp + costs_exp, 0.0, agent.cost_limit)[:, None]  # (N, 1)

        # V_next via target policy conditioned on u_next
        def policy_sample_action(policy_params, s, u, key):
            bs = s.shape[0]
            z0 = agent.policy_base_std * jax.random.normal(key, (bs, agent.act_dim))
            K  = agent.ode_steps
            dt = 1.0 / K

            def step_fn(z, i):
                t    = jnp.full((bs, 1), (i + 0.5) * dt)
                dz   = agent.policy_flow.apply_fn({"params": policy_params}, t, z, s, u)
                z_next = z + dt * dz
                if agent.clip_sampler:
                    z_next = jnp.clip(z_next, -1.0, 1.0)
                return z_next, None

            action, _ = lax.scan(step_fn, z0, jnp.arange(K))
            return action

        def ode_integrate_q(velocity_params, z0, s, a, u, cat):
            K  = agent.ode_steps
            dt = 1.0 / K

            def step_fn(z, i):
                t  = jnp.full((z.shape[0], 1), (i + 0.5) * dt)
                dz = agent.velocity.apply_fn(
                    {"params": velocity_params}, t, z, s, a, u, cat
                )
                return z + dt * dz, None

            final_z, _ = lax.scan(step_fn, z0, jnp.arange(K))
            return jnp.clip(final_z, -agent.y_norm_clip * 2, agent.y_norm_clip * 2)

        def estimate_q_stats_normalized(velocity_params, s, a, u, cat, key):
            m  = agent.q_samples
            bs = s.shape[0]
            s_rep   = jnp.tile(s[None],   (m, 1, 1)).reshape(-1, s.shape[-1])
            a_rep   = jnp.tile(a[None],   (m, 1, 1)).reshape(-1, a.shape[-1])
            cat_rep = jnp.tile(cat[None], (m, 1)).reshape(-1)
            u_rep   = jnp.tile(u[None],   (m, 1, 1)).reshape(-1, 1)
            z0 = jax.random.uniform(
                key, (m * bs, 1), minval=agent.base_dist_low, maxval=agent.base_dist_high
            )
            z_final = ode_integrate_q(velocity_params, z0, s_rep, a_rep, u_rep, cat_rep)
            z_final = z_final.reshape(m, bs)
            q_mean  = z_final.mean(axis=0)
            q_std   = z_final.std(axis=0) + 1e-6
            return q_mean, q_std

        # Next action conditioned on u_next
        next_a_exp = policy_sample_action(
            agent.target_policy_flow.params, next_obs_exp, u_next_col, k_next
        )

        # Category for next state: single-step cost at t+1 is unknown.
        # Use remaining budget at t+1 (after paying current step cost).
        qcs_next_exp   = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params}, next_obs_exp, next_a_exp
        )
        max_qc_next_exp  = qcs_next_exp.max(axis=0)
        category_next_exp = classify_qc_with_budget(max_qc_next_exp, agent.qc_threshold)

        v_next_mean_norm, v_next_std_norm = estimate_q_stats_normalized(
            agent.velocity_target_params,
            next_obs_exp, next_a_exp, u_next_col, category_next_exp, k_q1
        )
        v_next_norm_cons = v_next_mean_norm - agent.q_conservative_alpha * v_next_std_norm
        v_next_norm_cons = jnp.clip(v_next_norm_cons, -3.0, 3.0)

        # Normalise r_tilde, build Flow-Q target y_norm
        y_abs_bound = agent.lambda_max / (1.0 - agent.discount)
        batch_mean  = jnp.clip(r_tilde_exp.mean(), -y_abs_bound, y_abs_bound)
        batch_std   = jnp.clip(jnp.maximum(r_tilde_exp.std(), 1e-6), 1e-6, y_abs_bound)
        new_q_mean  = agent.q_norm_ema * agent.q_mean + (1 - agent.q_norm_ema) * batch_mean
        new_q_std   = agent.q_norm_ema * agent.q_std  + (1 - agent.q_norm_ema) * batch_std

        r_tilde_norm_exp = (r_tilde_exp - new_q_mean) / jnp.maximum(new_q_std, 1e-6)
        r_tilde_norm_exp = jnp.clip(r_tilde_norm_exp, -agent.y_norm_clip, agent.y_norm_clip)
        y_norm_exp = r_tilde_norm_exp + agent.discount * masks_exp * v_next_norm_cons
        y_norm_exp = jnp.clip(y_norm_exp, -agent.y_norm_clip, agent.y_norm_clip)

        unsafe_w_exp = jnp.where(category_exp == 2, agent.unsafe_flow_weight, 1.0)

        # Flow-Q loss: N samples, each with its own fresh (z0, t)
        def flow_loss_fn(velocity_params):
            z0   = jax.random.uniform(k_flow_z, (N, 1), minval=agent.base_dist_low, maxval=agent.base_dist_high)
            t    = jax.random.uniform(k_flow_t, (N, 1))
            z_t  = (1.0 - t) * z0 + t * y_norm_exp[:, None]
            pred = agent.velocity.apply_fn(
                {"params": velocity_params}, t, z_t, obs_exp, act_exp, u_col, category_exp
            )
            target = y_norm_exp[:, None] - z0
            loss   = ((pred - target) ** 2).squeeze(-1)
            return (unsafe_w_exp * loss).mean()

        flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(agent.velocity.params)
        new_velocity    = agent.velocity.apply_gradients(grads=flow_grads)
        new_vel_target  = optax.incremental_update(
            new_velocity.params, agent.velocity_target_params, agent.actor_tau
        )

        # ----------------------------------------------------------
        # IQL-style advantage: compute on original B batch with one u per (s,a).
        # V(s, u) is learned via expectile regression — no policy action sampling.
        # ----------------------------------------------------------
        u_adv     = jax.random.uniform(k_pi, (B,), minval=5.0, maxval=20.0)
        u_adv_col = u_adv[:, None]   # (B, 1)

        # Reuse max_qc_b already computed above; classify with fixed threshold
        cat_b = classify_budget(u_adv, costs, max_qc_b, agent.cost_limit, agent.qc_threshold)

        # Q(s, a_data, u) from target velocity network
        q_data_mean_norm, q_data_std_norm = estimate_q_stats_normalized(
            agent.velocity_target_params, observations, actions, u_adv_col, cat_b, k_q2
        )
        q_data_norm_cons = q_data_mean_norm - agent.q_conservative_alpha * q_data_std_norm
        q_data = q_data_norm_cons * new_q_std + new_q_mean

        # Update V(s, u) via expectile regression against Q(s, a_data, u)
        def reward_value_loss_fn(rv_params):
            v_pred = agent.reward_value.apply_fn({"params": rv_params}, observations, u_adv_col)
            diff   = lax.stop_gradient(q_data) - v_pred
            weight = jnp.where(diff > 0, agent.value_expectile, 1.0 - agent.value_expectile)
            return (weight * diff ** 2).mean()

        rv_grads   = jax.grad(reward_value_loss_fn)(agent.reward_value.params)
        new_reward_value = agent.reward_value.apply_gradients(grads=rv_grads)

        # V(s, u) from freshly updated network (stop_gradient applied inside loss above)
        v_s = new_reward_value.apply_fn({"params": new_reward_value.params}, observations, u_adv_col)

        adv      = q_data - lax.stop_gradient(v_s)
        adv_mean = adv.mean()
        adv_std  = adv.std() + 1e-6
        adv_norm = (adv - adv_mean) / adv_std
        weights  = jnp.exp(adv_norm * agent.awr_temperature)
        weights  = jnp.clip(weights, 0.0, agent.max_weight)
        # Repeat u_adv n times so actor loss is conditioned on the SAME u as the AWR weights.
        # This aligns policy conditioning with the advantage signal.
        weights_sg    = lax.stop_gradient(jnp.repeat(weights, n, axis=0))
        u_adv_exp     = jnp.repeat(u_adv, n, axis=0)                              # (N,)
        u_adv_exp_col = jnp.repeat(u_adv_col, n, axis=0)                          # (N, 1)
        costs_adv_exp = jnp.repeat(costs, n, axis=0)                              # (N,)
        # Actor-side safe/unsafe gating must match the budget-aware training semantics:
        # a sample is safe only if it passes both the hard remaining-budget check and Qc threshold.
        actor_safe_mask = classify_budget(
            u_adv_exp, costs_adv_exp, max_qc_exp, agent.cost_limit, agent.qc_threshold
        ) == 0
        actor_safe_mask_f = actor_safe_mask.astype(jnp.float32)
        actor_safe_frac   = actor_safe_mask_f.mean()
        actor_unsafe_frac = 1.0 - actor_safe_frac
        cat_safe_mask_f   = (cat_b == 0).astype(jnp.float32)
        cat_unsafe_mask_f = 1.0 - cat_safe_mask_f
        cat_safe_denom    = jnp.maximum(cat_safe_mask_f.sum(), 1.0)
        cat_unsafe_denom  = jnp.maximum(cat_unsafe_mask_f.sum(), 1.0)
        safe_weight_mean  = (cat_safe_mask_f * weights).sum() / cat_safe_denom
        unsafe_weight_mean = (cat_unsafe_mask_f * weights).sum() / cat_unsafe_denom
        q_data_safe_mean   = (cat_safe_mask_f * q_data).sum() / cat_safe_denom
        q_data_unsafe_mean = (cat_unsafe_mask_f * q_data).sum() / cat_unsafe_denom
        v_s_safe_mean      = (cat_safe_mask_f * v_s).sum() / cat_safe_denom
        v_s_unsafe_mean    = (cat_unsafe_mask_f * v_s).sum() / cat_unsafe_denom
        adv_safe_mean      = (cat_safe_mask_f * adv).sum() / cat_safe_denom
        adv_unsafe_mean    = (cat_unsafe_mask_f * adv).sum() / cat_unsafe_denom

        # Actor loss: CFM with AWR weights, conditioned on u_adv (aligned with weights)
        def actor_loss_fn(policy_params):
            z0     = agent.policy_base_std * jax.random.normal(k_actor_z, (N, agent.act_dim))
            t      = jax.random.uniform(k_actor_t, (N, 1))
            z_t    = (1.0 - t) * z0 + t * act_exp
            target = act_exp - z0

            pred           = agent.policy_flow.apply_fn({"params": policy_params}, t, z_t, obs_exp, u_adv_exp_col)
            per_example_fm = ((pred - target) ** 2).sum(axis=-1)

            # AWR only on budget-aware safe region; normalize by the number of safe
            # samples so the signal is not diluted when the safe fraction is small.
            safe_mask_adv = actor_safe_mask.astype(jnp.float32)
            safe_denom    = jnp.maximum(safe_mask_adv.sum(), 1.0)
            awr_loss      = (safe_mask_adv * weights_sg * per_example_fm).sum() / safe_denom
            # BC uses the same budget-aware safe/unsafe split.
            bc_coef_per = jnp.where(actor_safe_mask, agent.bc_coef, agent.bc_coef_unsafe)
            bc_loss     = (bc_coef_per * per_example_fm).mean()
            unsafe_mask_adv = 1.0 - safe_mask_adv
            unsafe_denom    = jnp.maximum(unsafe_mask_adv.sum(), 1.0)
            bc_safe_loss    = (safe_mask_adv * per_example_fm).sum() / safe_denom
            bc_unsafe_loss  = (unsafe_mask_adv * per_example_fm).sum() / unsafe_denom
            total_loss  = awr_loss + bc_loss

            return total_loss, {
                "actor_loss":      total_loss,
                "policy_fm_loss":  per_example_fm.mean(),
                "awr_loss":        awr_loss,
                "bc_loss":         bc_loss,
                "bc_safe_loss":    bc_safe_loss,
                "bc_unsafe_loss":  bc_unsafe_loss,
                "adv_mean":        adv.mean(),
                "weight_mean":     weights.mean(),
            }

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            agent.policy_flow.params
        )
        new_policy_flow = agent.policy_flow.apply_gradients(grads=actor_grads)
        new_target_policy_params = optax.incremental_update(
            new_policy_flow.params, agent.target_policy_flow.params, agent.actor_tau
        )
        new_target_policy_flow = agent.target_policy_flow.replace(params=new_target_policy_params)

        new_agent = agent.replace(
            velocity=new_velocity,
            velocity_target_params=new_vel_target,
            policy_flow=new_policy_flow,
            target_policy_flow=new_target_policy_flow,
            reward_value=new_reward_value,
            rng=rng,
            q_mean=new_q_mean,
            q_std=new_q_std,
        )

        info = {
            "flow_loss":           flow_loss,
            "actor_loss":          actor_info["actor_loss"],
            "policy_fm_loss":      actor_info["policy_fm_loss"],
            "awr_loss":            actor_info["awr_loss"],
            "bc_loss":             actor_info["bc_loss"],
            "adv_mean":            actor_info["adv_mean"],
            "weight_mean":         actor_info["weight_mean"],
            "bc_safe_loss":        actor_info["bc_safe_loss"],
            "bc_unsafe_loss":      actor_info["bc_unsafe_loss"],
            "v_s_mean":            v_s.mean(),
            "v_s_safe_mean":       v_s_safe_mean,
            "v_s_unsafe_mean":     v_s_unsafe_mean,
            "q_data_mean":         q_data.mean(),
            "q_data_safe_mean":    q_data_safe_mean,
            "q_data_unsafe_mean":  q_data_unsafe_mean,
            "adv_safe_mean":       adv_safe_mean,
            "adv_unsafe_mean":     adv_unsafe_mean,
            "adv_std":             adv_std,
            "lambda_mean":         lam_exp.mean(),
            "max_qc_data_mean":    max_qc_exp.mean(),
            "max_qc_data_b_mean":  max_qc_b.mean(),
            "q_mean":              new_q_mean,
            "q_std":               new_q_std,
            "r_tilde_mean":        r_tilde_exp.mean(),
            "y_norm_mean":         y_norm_exp.mean(),
            "cat0_frac":           (category_exp == 0).mean(),
            "cat1_frac":           (category_exp == 1).mean(),
            "cat2_frac":           (category_exp == 2).mean(),
            "actor_safe_frac":     actor_safe_frac,
            "actor_unsafe_frac":   actor_unsafe_frac,
            "safe_weight_mean":    safe_weight_mean,
            "unsafe_weight_mean":  unsafe_weight_mean,
            # Budget diagnostics
            "cost_limit":          agent.cost_limit,
            "qc_threshold":        agent.qc_threshold,
            "u_exp_min":           u_exp.min(),
            "u_exp_max":           u_exp.max(),
            "u_adv_min":           u_adv.min(),
            "u_adv_max":           u_adv.max(),
            "max_qc_exp_mean":     max_qc_exp.mean(),
            "max_qc_exp_max":      max_qc_exp.max(),
        }
        return new_agent, info

    # ============================================================
    # Evaluation: sample 8 candidates, pick safest then best reward
    # ============================================================

    @jax.jit
    def _sample_safe_best_reward(self, observations, u):
        """
        Args:
            observations: (8, obs_dim) — repeated single observation
            u:            scalar jax array — raw accumulated episode cost
        """
        eval_N = observations.shape[0]
        rng    = self.rng
        rng, k_policy, k_q = jax.random.split(rng, 3)

        # Broadcast u to (eval_N, 1)
        u_batch = jnp.full((eval_N, 1), u)

        # ---- sample actions from policy conditioned on u ----
        def policy_sample_action(policy_params, s, u_b, key):
            bs = s.shape[0]
            z0 = self.policy_base_std * jax.random.normal(key, (bs, self.act_dim))
            K  = self.ode_steps
            dt = 1.0 / K

            def step_fn(z, i):
                t      = jnp.full((bs, 1), (i + 0.5) * dt)
                dz     = self.policy_flow.apply_fn({"params": policy_params}, t, z, s, u_b)
                z_next = z + dt * dz
                if self.clip_sampler:
                    z_next = jnp.clip(z_next, -1.0, 1.0)
                return z_next, None

            action, _ = lax.scan(step_fn, z0, jnp.arange(K))
            return action

        actions = policy_sample_action(
            self.target_policy_flow.params, observations, u_batch, k_policy
        )

        # ---- select action by minimum Qc ----
        qcs          = self.safe_target_critic.apply_fn(
            {"params": self.safe_target_critic.params}, observations, actions
        )
        max_qc       = qcs.max(axis=0)
        best_idx    = jnp.argmin(max_qc)
        best_action = actions[best_idx]
        return best_action, rng

    def eval_actions(self, observations: jnp.ndarray, u: float = 0.0):
        """Sample 8 candidates conditioned on current budget u, pick best safe action.

        Args:
            observations: 1-D observation array
            u:            accumulated episode cost so far (raw float, default 0.0)
        """
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(8, axis=0)
        # u is raw accumulated episode cost, same scale as training (no normalization needed).
        u_jax        = jnp.array(u, dtype=jnp.float32)
        action, rng  = self._sample_safe_best_reward(observations, u_jax)
        action       = np.asarray(action)
        return action, self.replace(rng=rng)

    # ============================================================
    # JIT-compiled update entry points
    # ============================================================

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, flow_info = new_agent.update_flow_q_and_actor(batch)
        return new_agent, flow_info

    @jax.jit
    def update_cost_critics(self, batch: DatasetDict):
        new_agent = self
        new_agent, vc_info = new_agent.update_vc(batch)
        new_agent, qc_info = new_agent.update_qc(batch)
        return new_agent, {**vc_info, **qc_info}

    # ============================================================
    # Serialisation
    # ============================================================

    def save(self, modeldir, save_time):
        file_name  = "model" + str(save_time) + ".pickle"
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), "wb"))

    def load(self, model_location):
        pkl_file  = pickle.load(open(model_location, "rb"))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
