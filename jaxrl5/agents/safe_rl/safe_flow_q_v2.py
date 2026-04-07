"""SafeFlowQ v2 - Distributional Flow Matching Q with risk-sensitive safety.

Key improvements over v1:
1. Distributional Q estimation: CVaR for advantage, tail risk for safety
2. Unified actor loss: Safety-Aware AWR (no conflicting BC loss)
3. Continuous safety conditioning: replace discrete {0,1,2} categories with continuous safety level
4. Actor-based lambda: cost penalty reflects current policy, not data policy
5. More ODE steps / Q samples for accurate distribution estimation
6. Batch-based Q normalization instead of lagging EMA
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
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, get_weight_decay_mask


def safe_expectile_loss(diff, expectile=0.8):
    """For cost V: penalize underestimation"""
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


# ============================================================================
# Flow Matching Networks (v2: continuous safety conditioning)
# ============================================================================

class FourierTimeEmbedding(nn.Module):
    embed_dim: int = 64

    @nn.compact
    def __call__(self, t):
        frequencies = jnp.arange(1, self.embed_dim + 1, dtype=jnp.float32) * jnp.pi
        return jnp.cos(t * frequencies)


class VelocityNetworkV2(nn.Module):
    """Velocity network with continuous safety conditioning.

    Instead of discrete category embedding {0,1,2}, takes a continuous
    safety_level in [-1, 1] and projects it through a dense layer.
    This eliminates hard category boundaries and allows gradient flow.
    """
    hidden_dim: int = 256
    time_embed_dim: int = 64

    @nn.compact
    def __call__(self, t, z, state, action, safety_level):
        time_feat = FourierTimeEmbedding(self.time_embed_dim)(t)
        # Continuous safety conditioning: safety_level is (batch, 1) in [-1, 1]
        safety_feat = nn.Dense(self.hidden_dim, kernel_init=default_init())(safety_level)
        safety_feat = nn.silu(safety_feat)

        x = jnp.concatenate([state, action, z, time_feat, safety_feat], axis=-1)

        for _ in range(3):
            x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)

        return nn.Dense(1, kernel_init=default_init())(x)


class ResBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        return x + residual


class Actor(nn.Module):
    """ResNet Actor - outputs mean and log_std for Gaussian policy"""
    hidden_dim: int = 256
    action_dim: int = 2
    num_res_blocks: int = 2

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(state)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        for _ in range(self.num_res_blocks):
            x = ResBlock(self.hidden_dim)(x)

        mean = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        log_std = jnp.clip(nn.Dense(self.action_dim, kernel_init=default_init())(x), -5.0, 2.0)
        return mean, log_std


class SafeFlowQV2(Agent):
    """
    SafeFlowQ v2 - Distributional Flow Matching for Safe Offline RL.

    Networks:
    - Vc: Cost Value (expectile loss, pretrained)
    - Qc: Cost Q (TD with HJ Bellman, pretrained)
    - Q_flow (velocity): Flow Matching Q, learns distribution of r - lambda*c
    - Actor: ResNet Gaussian, trained with Safety-Aware AWR

    Key changes from v1:
    - Distributional Q: CVaR for advantage, tail risk for safety
    - No BC loss; safety built into AWR weights via safety_factor
    - Continuous safety conditioning instead of discrete categories
    - Lambda computed on actor actions (not data actions)
    - Batch normalization for Q targets
    """
    # Flow Q (velocity network)
    velocity: TrainState
    velocity_target_params: dict

    # Actor
    actor: TrainState

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

    # Flow Matching specific
    ode_steps: int = struct.field(pytree_node=False)
    q_samples: int = struct.field(pytree_node=False)
    base_dist_low: float
    base_dist_high: float

    # Cost penalty params
    lambda_max: float
    softplus_beta: float
    qc_thres: float
    eps_safe: float
    eps_unsafe: float

    # Actor loss params
    awr_temperature: float
    max_weight: float
    safety_coef: float       # soft safety penalty coefficient (reduced role)
    safety_temp: float        # temperature for safety_factor sigmoid

    # Distributional params
    cvar_alpha: float         # CVaR quantile for conservative Q (e.g. 0.25)
    tail_alpha: float         # tail quantile for pessimistic safety (e.g. 0.75)

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
        discount: float = 0.99,
        tau: float = 0.005,
        cost_critic_hyperparam: float = 0.9,
        num_qs: int = 2,
        actor_tau: float = 0.005,
        critic_type: str = 'hj',
        decay_steps: Optional[int] = int(2e6),
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        # Flow Matching specific
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        ode_steps: int = 20,
        q_samples: int = 32,
        base_dist_low: float = -3.0,
        base_dist_high: float = 3.0,
        lambda_max: float = 100.0,
        softplus_beta: float = 1.0,
        # Actor loss params
        awr_temperature: float = 3.0,
        max_weight: float = 100.0,
        safety_coef: float = 5.0,
        safety_temp: float = 1.0,
        # Distributional params
        cvar_alpha: float = 0.25,
        tail_alpha: float = 0.75,
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

        # ===== Flow Q (v2: continuous safety conditioning) =====
        velocity_net = VelocityNetworkV2(hidden_dim, time_embed_dim)
        dummy_t = jnp.zeros((1, 1))
        dummy_z = jnp.zeros((1, 1))
        dummy_safety = jnp.zeros((1, 1))  # continuous safety level
        velocity_params = velocity_net.init(
            velocity_key, dummy_t, dummy_z, observations, actions, dummy_safety
        )['params']
        velocity = TrainState.create(
            apply_fn=velocity_net.apply,
            params=velocity_params,
            tx=optax.adam(actor_lr_schedule)
        )

        # ===== Actor =====
        actor_net = Actor(hidden_dim, action_dim)
        actor_params = actor_net.init(actor_key, observations)['params']
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_params,
            tx=optax.adam(actor_lr_schedule)
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

        return cls(
            velocity=velocity,
            velocity_target_params=velocity_params,
            actor=actor,
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
            base_dist_low=base_dist_low,
            base_dist_high=base_dist_high,
            lambda_max=lambda_max,
            softplus_beta=softplus_beta,
            awr_temperature=awr_temperature,
            max_weight=max_weight,
            safety_coef=safety_coef,
            safety_temp=safety_temp,
            cvar_alpha=cvar_alpha,
            tail_alpha=tail_alpha,
        )

    # ===== Helpers =====

    def _compute_safety_level(agent, qc_values):
        """Convert Qc values to continuous safety level in [-1, 1].

        safety_level < 0 means safe, > 0 means unsafe, ~0 is boundary.
        """
        d = agent.qc_thres
        half_width = agent.eps_safe + agent.eps_unsafe
        safety_level = (qc_values - d) / jnp.maximum(half_width, 1e-6)
        return jnp.clip(safety_level, -1.0, 1.0)

    def _sample_action(agent, actor_params, s, key):
        """Sample action from Gaussian policy with tanh squashing."""
        mean, log_std = agent.actor.apply_fn({'params': actor_params}, s)
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, mean.shape)
        return jnp.tanh(mean + std * noise)

    def _ode_integrate(agent, velocity_params, z0, s, a, safety_level):
        """Euler ODE integration for flow matching."""
        K = agent.ode_steps
        dt = 1.0 / K
        def step_fn(z, i):
            t = jnp.full((z.shape[0], 1), i * dt)
            v = agent.velocity.apply_fn(
                {'params': velocity_params}, t, z, s, a, safety_level
            )
            return z + dt * v, None
        final_z, _ = lax.scan(step_fn, z0, jnp.arange(K))
        return final_z

    def _estimate_q_distribution(agent, velocity_params, s, a, safety_level, key):
        """Estimate Q-value distribution by integrating velocity network.

        Returns all samples (not just mean) for distributional estimation.
        Shape: (q_samples, batch_size)
        """
        m = agent.q_samples
        bs = s.shape[0]
        s_exp = jnp.tile(s[None], (m, 1, 1)).reshape(-1, s.shape[-1])
        a_exp = jnp.tile(a[None], (m, 1, 1)).reshape(-1, a.shape[-1])
        sl_exp = jnp.tile(safety_level[None], (m, 1, 1)).reshape(-1, 1)
        z0 = jax.random.uniform(key, (m * bs, 1),
                                minval=agent.base_dist_low,
                                maxval=agent.base_dist_high)
        z_final = agent._ode_integrate(velocity_params, z0, s_exp, a_exp, sl_exp)
        return z_final.reshape(m, bs)  # (q_samples, batch_size)

    def _cvar(agent, samples):
        """CVaR (lower alpha-quantile mean) for conservative Q estimation.

        samples: (q_samples, batch_size)
        Returns: (batch_size,)
        """
        sorted_s = jnp.sort(samples, axis=0)
        k = jnp.maximum(1, jnp.floor(agent.q_samples * agent.cvar_alpha).astype(jnp.int32))
        indices = jnp.arange(agent.q_samples)[:, None]  # (q_samples, 1)
        mask = (indices < k).astype(jnp.float32)         # (q_samples, 1)
        mask = mask / jnp.maximum(mask.sum(axis=0, keepdims=True), 1e-6)
        return (sorted_s * mask).sum(axis=0)  # (q_samples,1) broadcasts with (q_samples, bs)

    def _tail_risk(agent, samples):
        """Upper tail mean for pessimistic safety estimation.

        samples: (q_samples, batch_size)
        Returns: (batch_size,)
        """
        sorted_s = jnp.sort(samples, axis=0)
        k = jnp.maximum(1, jnp.floor(agent.q_samples * agent.tail_alpha).astype(jnp.int32))
        indices = jnp.arange(agent.q_samples)[:, None]  # (q_samples, 1)
        mask = (indices >= k).astype(jnp.float32)        # (q_samples, 1)
        mask = mask / jnp.maximum(mask.sum(axis=0, keepdims=True), 1e-6)
        return (sorted_s * mask).sum(axis=0)

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
            safe_value_loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()
            return safe_value_loss, {
                "safe_value_loss": safe_value_loss,
                "vc": vc.mean(), "vc_max": vc.max(), "vc_min": vc.min()
            }

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        return agent.replace(safe_value=safe_value), info

    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Update Cost Q using TD with HJ Bellman"""
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )

        if agent.critic_type == "hj":
            qc_nonterminal = (1. - agent.discount) * batch["costs"] + \
                agent.discount * jnp.maximum(batch["costs"], next_vc)
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == 'qc':
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f'Invalid critic type: {agent.critic_type}')

        def safe_critic_loss_fn(safe_critic_params):
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()
            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(), "qc_max": qcs.max(), "qc_min": qcs.min()
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        return agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic), info

    # ===== Flow Q and Actor Updates (main training phase) =====

    def update_flow_q_and_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Update Flow Q and Actor with distributional safety-aware training."""
        rng = agent.rng
        rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        next_observations = batch["next_observations"]
        masks = batch["masks"]
        batch_size = observations.shape[0]

        d = agent.qc_thres

        # ================================================================
        # Improvement 4: Lambda based on ACTOR actions (not data actions)
        # Reflects the current policy's safety, not the data policy's.
        # ================================================================
        actor_actions_for_lam = agent._sample_action(agent.actor.params, observations, k1)
        qc_actor_lam = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params}, observations, actor_actions_for_lam
        )
        max_qc_actor_lam = qc_actor_lam.max(axis=0)

        lam = jax.nn.softplus(agent.softplus_beta * (max_qc_actor_lam - d)) / agent.softplus_beta
        lam = jnp.clip(lam, 0, agent.lambda_max)

        r_tilde = rewards - lam * costs

        # ================================================================
        # Improvement 3: Continuous safety level instead of discrete category
        # ================================================================
        # Safety level for data actions
        qc_data = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params}, observations, actions
        )
        max_qc_data = qc_data.max(axis=0)
        safety_level_data = agent._compute_safety_level(max_qc_data)[:, None]  # (bs, 1)

        # Safety level for next state (using actor's next action)
        next_a = agent._sample_action(agent.actor.params, next_observations, k3)
        qc_next_a = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params}, next_observations, next_a
        )
        max_qc_next_a = qc_next_a.max(axis=0)
        safety_level_next = agent._compute_safety_level(max_qc_next_a)[:, None]

        # ================================================================
        # Improvement 1: Distributional V_next using mean (for TD target)
        # TD target uses mean for unbiased value estimation
        # ================================================================
        v_next_samples = agent._estimate_q_distribution(
            agent.velocity_target_params, next_observations, next_a, safety_level_next, k4
        )  # (q_samples, batch_size)
        v_next = v_next_samples.mean(axis=0)  # mean for TD target

        y = r_tilde + agent.discount * masks * v_next

        # ================================================================
        # Improvement 6: Batch-based normalization (no lagging EMA)
        # ================================================================
        y_mean = y.mean()
        y_std = jnp.maximum(y.std(), 1e-6)
        y_norm = (y - y_mean) / y_std

        # ===== Flow Q Loss (conditional flow matching) =====
        def flow_loss_fn(velocity_params):
            z0 = jax.random.uniform(k5, (batch_size, 1),
                                    minval=agent.base_dist_low,
                                    maxval=agent.base_dist_high)
            t = jax.random.uniform(k6, (batch_size, 1))
            z_t = (1 - t) * z0 + t * y_norm[:, None]

            pred = agent.velocity.apply_fn(
                {'params': velocity_params}, t, z_t, observations, actions, safety_level_data
            )
            target = y_norm[:, None] - z0
            return ((pred - target) ** 2).mean()

        flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(agent.velocity.params)
        new_velocity = agent.velocity.apply_gradients(grads=flow_grads)

        # ================================================================
        # Improvement 1: Distributional advantage with CVaR
        # Q(s, a_data): use CVaR for conservative advantage estimation
        # V(s): use CVaR of actor's actions for consistent baseline
        # ================================================================

        # Q(s, a_data) distribution
        q_data_samples = agent._estimate_q_distribution(
            new_velocity.params, observations, actions, safety_level_data, k2
        )
        # Denormalize: samples are in normalized space, convert back
        q_data_samples_denorm = q_data_samples * y_std + y_mean

        # CVaR of Q for data actions (conservative estimate)
        q_data_cvar = agent._cvar(q_data_samples_denorm)

        # V(s) from actor's sampled actions
        rng, v_key1, v_key2 = jax.random.split(rng, 3)
        sampled_actions = agent._sample_action(agent.actor.params, observations, v_key1)
        qc_actor = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params}, observations, sampled_actions
        )
        max_qc_actor = qc_actor.max(axis=0)
        safety_level_actor = agent._compute_safety_level(max_qc_actor)[:, None]

        v_s_samples = agent._estimate_q_distribution(
            new_velocity.params, observations, sampled_actions, safety_level_actor, v_key2
        )
        v_s_samples_denorm = v_s_samples * y_std + y_mean
        v_s_cvar = agent._cvar(v_s_samples_denorm)

        adv = q_data_cvar - v_s_cvar

        # ================================================================
        # Improvement 2: Safety-Aware AWR (no BC loss)
        # Safety factor down-weights unsafe data actions in AWR.
        # AWR itself provides implicit BC effect.
        # ================================================================
        safety_factor = jax.nn.sigmoid(-(max_qc_data - d) / agent.safety_temp)
        # safe actions → factor ≈ 1, unsafe actions → factor ≈ 0

        weights = jnp.exp(adv / agent.awr_temperature) * safety_factor
        weights = jnp.clip(weights, 0, agent.max_weight)

        rng, penalty_key = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            mean, log_std = agent.actor.apply_fn({'params': actor_params}, observations)
            std = jnp.exp(log_std)

            # --- Safety-Aware AWR loss ---
            actions_clipped = jnp.clip(actions, -0.99, 0.99)
            pre_tanh_actions = jnp.arctanh(actions_clipped)

            log_prob = -0.5 * (
                ((pre_tanh_actions - mean) / std) ** 2
                + 2 * log_std + jnp.log(2 * jnp.pi)
            )
            log_prob = log_prob.sum(axis=-1)
            log_prob -= jnp.sum(jnp.log(1 - actions_clipped ** 2 + 1e-6), axis=-1)

            awr_loss = -(weights * log_prob).mean()

            # --- Soft safety penalty (reduced role, gradient flows through actor) ---
            noise = jax.random.normal(penalty_key, mean.shape)
            sampled_action = jnp.tanh(mean + std * noise)
            qc_sampled = agent.safe_critic.apply_fn(
                {"params": agent.safe_critic.params}, observations, sampled_action
            )
            max_qc_sampled = qc_sampled.max(axis=0)
            safety_penalty = jnp.maximum(max_qc_sampled - d, 0.0).mean()

            total_loss = awr_loss + agent.safety_coef * safety_penalty

            return total_loss, {
                'actor_loss': total_loss,
                'awr_loss': awr_loss,
                'safety_penalty': safety_penalty,
                'q_data_cvar': q_data_cvar.mean(),
                'v_s_cvar': v_s_cvar.mean(),
                'log_prob': log_prob.mean(),
                'adv_mean': adv.mean(),
                'adv_std': adv.std(),
                'weight_mean': weights.mean(),
                'safety_factor_mean': safety_factor.mean(),
            }

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(agent.actor.params)
        new_actor = agent.actor.apply_gradients(grads=actor_grads)

        # Soft update target velocity
        new_vel_target = optax.incremental_update(
            new_velocity.params, agent.velocity_target_params, agent.actor_tau
        )

        new_agent = agent.replace(
            velocity=new_velocity,
            velocity_target_params=new_vel_target,
            actor=new_actor,
            rng=rng,
        )

        # Q distribution statistics for logging
        q_data_mean = q_data_samples_denorm.mean()
        q_data_std = q_data_samples_denorm.std(axis=0).mean()

        info = {
            "flow_loss": flow_loss,
            "actor_loss": actor_info['actor_loss'],
            "awr_loss": actor_info['awr_loss'],
            "safety_penalty": actor_info['safety_penalty'],
            "q_data_cvar": actor_info['q_data_cvar'],
            "v_s_cvar": actor_info['v_s_cvar'],
            "q_data_mean": q_data_mean,
            "q_data_std": q_data_std,
            "log_prob": actor_info['log_prob'],
            "adv_mean": actor_info['adv_mean'],
            "adv_std": actor_info['adv_std'],
            "weight_mean": actor_info['weight_mean'],
            "safety_factor_mean": actor_info['safety_factor_mean'],
            "lambda_mean": lam.mean(),
            "max_qc_actor_mean": max_qc_actor_lam.mean(),
            "max_qc_data_mean": max_qc_data.mean(),
            "y_target_mean": y.mean(),
            "y_norm_mean": y_norm.mean(),
            "y_std": y_std,
        }
        return new_agent, info

    def eval_actions(self, observations: jnp.ndarray):
        """Sample actions using the actor (stochastic)"""
        rng = self.rng
        rng, key = jax.random.split(rng)

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0)

        mean, log_std = self.actor.apply_fn({'params': self.actor.params}, observations)
        std = jnp.exp(log_std)

        noise = jax.random.normal(key, mean.shape)
        action = jnp.tanh(mean + std * noise)

        action = np.asarray(action).squeeze(axis=0)

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
