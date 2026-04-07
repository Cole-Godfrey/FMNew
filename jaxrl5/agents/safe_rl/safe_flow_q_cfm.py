"""SafeFlowQCFM - SafeFlowQ with a Conditional Flow Matching policy actor.

Vc/Qc are pretrained cost critics.
Q_flow (velocity) learns the normalized distribution of r_tilde + gamma * V_next.
Actor is a conditional flow matching (CFM) policy trained with AWR-style weights + risk-aware BC.

Main changes from SafeFlowQDiffusion:
1. Replace diffusion actor (DDPM / score model) with conditional flow matching policy.
2. Replace denoising loss with action-space FM loss on linear interpolation paths.
3. Replace diffusion sampling with ODE integration from base noise to action.
4. Remove explicit actor-side safety penalty; safety enters via Q_flow shaping + Qc-aware BC.

Kept unchanged:
- Vc / Qc pretraining and updates
- Flow Q target construction and conservative evaluation
- Safety categorization / lambda shaping
- AWR weighting from conservative Q_flow advantage
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


def safe_expectile_loss(diff, expectile=0.8):
    """For cost V: penalize underestimation."""
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class FourierTimeEmbedding(nn.Module):
    embed_dim: int = 64

    @nn.compact
    def __call__(self, t):
        frequencies = jnp.arange(1, self.embed_dim + 1, dtype=jnp.float32) * jnp.pi
        return jnp.cos(t * frequencies)


class VelocityNetwork(nn.Module):
    """Flow-Q velocity network: learns scalar return / penalized-return distribution."""
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


class PolicyFlowNetwork(nn.Module):
    """Conditional flow matching policy network.

    Learns v_pi(t, z_t, s) in action space, where z_t interpolates between base noise z0
    and a dataset action a.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256, 256)
    time_embed_dim: int = 64
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, t, z, state):
        time_feat = FourierTimeEmbedding(self.time_embed_dim)(t)
        x = jnp.concatenate([state, z, time_feat], axis=-1)

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=default_init())(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.silu(x)

        return nn.Dense(self.action_dim, kernel_init=default_init())(x)


class SafeFlowQCFM(Agent):
    """
    Networks:
    - Vc: Cost value (expectile loss, pretrained)
    - Qc: Cost Q (HJ Bellman or cumulative cost, pretrained)
    - Q_flow (velocity): flow matching critic over scalar penalized return
    - Policy flow: conditional flow matching actor over actions
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
    qc_thres: float
    eps_safe: float
    eps_unsafe: float

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
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, velocity_key, safe_critic_key, safe_value_key = jax.random.split(rng, 5)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount ** env_max_steps) / (1 - discount) / env_max_steps
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
        dummy_z_scalar = jnp.zeros((1, 1))
        dummy_cat = jnp.zeros((1,), dtype=jnp.int32)
        velocity_params = velocity_net.init(
            velocity_key, dummy_t, dummy_z_scalar, observations, actions, dummy_cat
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
        policy_params = policy_def.init(actor_key, dummy_t, dummy_z_action, observations)["params"]
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
            qc_thres=qc_thres,
            eps_safe=eps_safe,
            eps_unsafe=eps_unsafe,
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
        )

    # ===== Cost Critic Updates (pretrain phase only) =====
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
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        return agent.replace(safe_value=safe_value), info

    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn({"params": agent.safe_value.params}, batch["next_observations"])

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

    # ===== Main update: Flow Q + CFM actor =====
    def update_flow_q_and_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        rng, k_pi, k_next, k_flow_z, k_flow_t, k_q, k_actor = jax.random.split(rng, 7)
        k_q1, k_q2, k_q3 = jax.random.split(k_q, 3)
        k_actor_z, k_actor_t = jax.random.split(k_actor, 2)

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        next_observations = batch["next_observations"]
        masks = batch["masks"]
        B = observations.shape[0]
        d = agent.qc_thres

        def classify(cost_value):
            cat = jnp.ones_like(cost_value, dtype=jnp.int32)
            cat = jnp.where(cost_value <= d - agent.eps_safe, 0, cat)
            cat = jnp.where(cost_value >= d + agent.eps_unsafe, 2, cat)
            return cat

        def ode_integrate_q(velocity_params, z0, s, a, cat):
            K = agent.ode_steps
            dt = 1.0 / K

            def step_fn(z, i):
                t = jnp.full((z.shape[0], 1), (i + 0.5) * dt)
                dz = agent.velocity.apply_fn({"params": velocity_params}, t, z, s, a, cat)
                return z + dt * dz, None

            final_z, _ = lax.scan(step_fn, z0, jnp.arange(K))
            return jnp.clip(final_z, -agent.y_norm_clip * 2, agent.y_norm_clip * 2)

        def policy_sample_action(policy_params, s, key):
            bs = s.shape[0]
            z0 = agent.policy_base_std * jax.random.normal(key, (bs, agent.act_dim))
            K = agent.ode_steps
            dt = 1.0 / K

            def step_fn(z, i):
                t = jnp.full((bs, 1), (i + 0.5) * dt)
                dz = agent.policy_flow.apply_fn({"params": policy_params}, t, z, s)
                z_next = z + dt * dz
                if agent.clip_sampler:
                    z_next = jnp.clip(z_next, -1.0, 1.0)
                return z_next, None

            action, _ = lax.scan(step_fn, z0, jnp.arange(K))
            return action

        def estimate_q_stats_normalized(velocity_params, s, a, cat, key):
            m = agent.q_samples
            bs = s.shape[0]
            s_exp = jnp.tile(s[None], (m, 1, 1)).reshape(-1, s.shape[-1])
            a_exp = jnp.tile(a[None], (m, 1, 1)).reshape(-1, a.shape[-1])
            cat_exp = jnp.tile(cat[None], (m, 1)).reshape(-1)
            z0 = jax.random.uniform(
                key, (m * bs, 1), minval=agent.base_dist_low, maxval=agent.base_dist_high
            )
            z_final = ode_integrate_q(velocity_params, z0, s_exp, a_exp, cat_exp)
            z_final = z_final.reshape(m, bs)
            q_mean = z_final.mean(axis=0)
            q_std = z_final.std(axis=0) + 1e-6
            return q_mean, q_std

        # Qc(s, a_data): used for lambda shaping and data category
        qcs_data = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, observations, actions)
        max_qc_data = qcs_data.max(axis=0)
        category_data = classify(max_qc_data)

        lam = jax.nn.softplus(agent.softplus_beta * (max_qc_data - d)) / agent.softplus_beta
        lam = jnp.clip(lam, 0, agent.lambda_max)

        safe_mask = category_data == 0
        unsafe_mask = category_data == 2
        r_tilde = jnp.where(safe_mask, rewards, jnp.where(unsafe_mask, -costs, rewards - lam * costs))

        # V_next: category_next from Qc(s', a_next) under current target policy
        next_a = policy_sample_action(agent.target_policy_flow.params, next_observations, k_next)
        qcs_next = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, next_observations, next_a)
        max_qc_next = qcs_next.max(axis=0)
        category_next = classify(max_qc_next)

        v_next_mean_norm, v_next_std_norm = estimate_q_stats_normalized(
            agent.velocity_target_params, next_observations, next_a, category_next, k_q1
        )
        v_next_norm_cons = v_next_mean_norm - agent.q_conservative_alpha * v_next_std_norm
        v_next_norm_cons = jnp.clip(v_next_norm_cons, -3.0, 3.0)

        y_abs_bound = agent.lambda_max / (1.0 - agent.discount)
        batch_mean = jnp.clip(r_tilde.mean(), -y_abs_bound, y_abs_bound)
        batch_std = jnp.clip(jnp.maximum(r_tilde.std(), 1e-6), 1e-6, y_abs_bound)
        new_q_mean = agent.q_norm_ema * agent.q_mean + (1 - agent.q_norm_ema) * batch_mean
        new_q_std = agent.q_norm_ema * agent.q_std + (1 - agent.q_norm_ema) * batch_std

        r_tilde_norm = (r_tilde - new_q_mean) / jnp.maximum(new_q_std, 1e-6)
        r_tilde_norm = jnp.clip(r_tilde_norm, -agent.y_norm_clip, agent.y_norm_clip)
        y_norm = r_tilde_norm + agent.discount * masks * v_next_norm_cons
        y_norm = jnp.clip(y_norm, -agent.y_norm_clip, agent.y_norm_clip)

        unsafe_w = jnp.where(category_data == 2, agent.unsafe_flow_weight, 1.0)

        def flow_loss_fn(velocity_params):
            n = agent.noise_samples
            N = n * B
            obs_rep = jnp.repeat(observations, n, axis=0)
            act_rep = jnp.repeat(actions, n, axis=0)
            cat_rep = jnp.repeat(category_data, n, axis=0)
            y_rep = jnp.repeat(y_norm, n, axis=0)
            w_rep = jnp.repeat(unsafe_w, n, axis=0)

            z0 = jax.random.uniform(key=k_flow_z, shape=(N, 1), minval=agent.base_dist_low, maxval=agent.base_dist_high)
            t = jax.random.uniform(k_flow_t, (N, 1))
            z_t = (1.0 - t) * z0 + t * y_rep[:, None]
            pred = agent.velocity.apply_fn({"params": velocity_params}, t, z_t, obs_rep, act_rep, cat_rep)
            target = y_rep[:, None] - z0
            loss = ((pred - target) ** 2).squeeze(-1)
            return (w_rep * loss).mean()

        flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(agent.velocity.params)
        new_velocity = agent.velocity.apply_gradients(grads=flow_grads)
        new_vel_target = optax.incremental_update(new_velocity.params, agent.velocity_target_params, agent.actor_tau)

        # Conservative Q(s, a_data)
        q_data_mean_norm, q_data_std_norm = estimate_q_stats_normalized(
            agent.velocity_target_params, observations, actions, category_data, k_q2
        )
        q_data_norm_cons = q_data_mean_norm - agent.q_conservative_alpha * q_data_std_norm
        q_data = q_data_norm_cons * new_q_std + new_q_mean

        # V(s): evaluate current target policy action under Q_flow
        a_pi = policy_sample_action(agent.target_policy_flow.params, observations, k_pi)
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
        weights = jnp.exp(adv_norm * agent.awr_temperature)
        weights = jnp.clip(weights, 0.0, agent.max_weight)
        weights_sg = lax.stop_gradient(weights)

        def actor_loss_fn(policy_params):
            n = agent.noise_samples
            N = n * B

            obs_rep = jnp.repeat(observations, n, axis=0)
            act_rep = jnp.repeat(actions, n, axis=0)
            w_rep = jnp.repeat(weights_sg, n, axis=0)
            max_qc_data_rep = jnp.repeat(max_qc_data, n, axis=0)

            z0 = agent.policy_base_std * jax.random.normal(k_actor_z, (N, agent.act_dim))
            t = jax.random.uniform(k_actor_t, (N, 1))
            z_t = (1.0 - t) * z0 + t * act_rep
            target = act_rep - z0

            pred = agent.policy_flow.apply_fn({"params": policy_params}, t, z_t, obs_rep)
            per_example_fm = ((pred - target) ** 2).sum(axis=-1)

            awr_loss = (w_rep * per_example_fm).mean()
            bc_coef_per = jnp.where(max_qc_data_rep < d, agent.bc_coef, agent.bc_coef_unsafe)
            bc_loss = (bc_coef_per * per_example_fm).mean()
            total_loss = awr_loss + bc_loss

            return total_loss, {
                "actor_loss": total_loss,
                "policy_fm_loss": per_example_fm.mean(),
                "awr_loss": awr_loss,
                "bc_loss": bc_loss,
                "adv_mean": adv.mean(),
                "weight_mean": weights.mean(),
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
            rng=rng,
            q_mean=new_q_mean,
            q_std=new_q_std,
        )

        info = {
            "flow_loss": flow_loss,
            "actor_loss": actor_info["actor_loss"],
            "policy_fm_loss": actor_info["policy_fm_loss"],
            "awr_loss": actor_info["awr_loss"],
            "bc_loss": actor_info["bc_loss"],
            "adv_mean": actor_info["adv_mean"],
            "weight_mean": actor_info["weight_mean"],
            "adv_std": adv_std,
            "lambda_mean": lam.mean(),
            "max_qc_data_mean": max_qc_data.mean(),
            "max_qc_actor_mean": max_qc_actor.mean(),
            "q_mean": new_q_mean,
            "q_std": new_q_std,
            "r_tilde_mean": r_tilde.mean(),
            "y_norm_mean": y_norm.mean(),
        }
        return new_agent, info

    @jax.jit
    def _sample_safe_best_reward(self, observations):
        """Sample eval_N policy actions, keep safest by Qc, pick best by Q_flow."""
        eval_N = observations.shape[0]
        safe_k = 1
        rng = self.rng
        rng, k_policy, k_q = jax.random.split(rng, 3)

        def policy_sample_action(policy_params, s, key):
            bs = s.shape[0]
            z0 = self.policy_base_std * jax.random.normal(key, (bs, self.act_dim))
            K = self.ode_steps
            dt = 1.0 / K

            def step_fn(z, i):
                t = jnp.full((bs, 1), (i + 0.5) * dt)
                dz = self.policy_flow.apply_fn({"params": policy_params}, t, z, s)
                z_next = z + dt * dz
                if self.clip_sampler:
                    z_next = jnp.clip(z_next, -1.0, 1.0)
                return z_next, None

            action, _ = lax.scan(step_fn, z0, jnp.arange(K))
            return action

        actions = policy_sample_action(self.target_policy_flow.params, observations, k_policy)

        qcs = self.safe_target_critic.apply_fn({"params": self.safe_target_critic.params}, observations, actions)
        max_qc = qcs.max(axis=0)
        safe_indices = jnp.argsort(max_qc)[:safe_k]
        safe_actions = actions[safe_indices]
        safe_obs = observations[safe_indices]
        safe_qc = max_qc[safe_indices]

        d = self.qc_thres
        safe_cat = jnp.ones((safe_k,), dtype=jnp.int32)
        safe_cat = jnp.where(safe_qc <= d - self.eps_safe, 0, safe_cat)
        safe_cat = jnp.where(safe_qc >= d + self.eps_unsafe, 2, safe_cat)

        m = self.q_samples
        s_exp = jnp.tile(safe_obs[None], (m, 1, 1)).reshape(-1, safe_obs.shape[-1])
        a_exp = jnp.tile(safe_actions[None], (m, 1, 1)).reshape(-1, safe_actions.shape[-1])
        cat_exp = jnp.tile(safe_cat[None], (m, 1)).reshape(-1)
        z0 = jax.random.uniform(k_q, (m * safe_k, 1), minval=self.base_dist_low, maxval=self.base_dist_high)

        K = self.ode_steps
        dt = 1.0 / K

        def step_fn(z, i):
            t = jnp.full((z.shape[0], 1), (i + 0.5) * dt)
            dz = self.velocity.apply_fn({"params": self.velocity_target_params}, t, z, s_exp, a_exp, cat_exp)
            return z + dt * dz, None

        z_final, _ = lax.scan(step_fn, z0, jnp.arange(K))
        z_final = jnp.clip(z_final, -self.y_norm_clip * 2, self.y_norm_clip * 2)
        q_mean = z_final.reshape(m, safe_k).mean(axis=0)

        best_idx = jnp.argmax(q_mean)
        best_action = safe_actions[best_idx]
        return best_action, rng

    def eval_actions(self, observations: jnp.ndarray):
        """Sample 8 candidates, keep safest by Qc, pick best reward by Q_flow."""
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(8, axis=0)
        action, rng = self._sample_safe_best_reward(observations)
        action = np.asarray(action)
        return action, self.replace(rng=rng)

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

    def save(self, modeldir, save_time):
        file_name = "model" + str(save_time) + ".pickle"
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), "wb"))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, "rb"))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
