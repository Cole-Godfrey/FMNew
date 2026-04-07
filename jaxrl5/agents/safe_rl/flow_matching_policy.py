




























# """SafeFlowQAllFlowMatching

# Keep Flow-Q architecture unchanged, while switching:
# 1) safety critics pretraining (Qc/Vc) to flow-matching critics
# 2) actor training/sampling to flow-matching policy

# Design notes:
# - Inspired by FloQ-style iterative flow critics (velocity + ODE integration).
# - Q_flow branch remains category-conditional and unchanged in spirit.
# - Safety critics are scalar flow critics:
#     * Qc_flow: conditioned on (s, a)
#     * Vc_flow: conditioned on s
# """
# import os
# from functools import partial
# from typing import Dict, Optional, Sequence, Tuple, Union

# import flax
# import flax.linen as nn
# import gymnasium as gym
# import jax
# import jax.numpy as jnp
# import numpy as np
# import optax
# import pickle
# from flax import struct
# from flax.core.frozen_dict import freeze
# from flax.training.train_state import TrainState
# from jax import lax

# from jaxrl5.agents.agent import Agent
# from jaxrl5.data.dataset import DatasetDict
# from jaxrl5.networks import (
#     FlowMatching,
#     FourierFeatures,
#     MLP,
#     MLPResNet,
#     flow_matching_sampler,
#     get_weight_decay_mask,
# )
# from jaxrl5.networks.flow_matching import flow_matching_loss


# def default_init(scale=1.0):
#     return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


# def mish(x):
#     return x * jnp.tanh(nn.softplus(x))


# def safe_expectile_loss(diff, expectile=0.8):
#     """IQL-style expectile loss (for cost value, underestimation is penalized more)."""
#     weight = jnp.where(diff < 0, expectile, (1.0 - expectile))
#     return weight * (diff ** 2)


# class FourierTimeEmbedding(nn.Module):
#     embed_dim: int = 64

#     @nn.compact
#     def __call__(self, t):
#         frequencies = jnp.arange(1, self.embed_dim + 1, dtype=jnp.float32) * jnp.pi
#         return jnp.cos(t * frequencies)


# class VelocityNetwork(nn.Module):
#     """Q_flow velocity network (unchanged style)."""
#     hidden_dim: int = 256
#     num_categories: int = 3
#     time_embed_dim: int = 64

#     @nn.compact
#     def __call__(self, t, z, state, action, category):
#         time_feat = FourierTimeEmbedding(self.time_embed_dim)(t)
#         cat_feat = nn.Embed(self.num_categories, self.hidden_dim)(category)
#         x = jnp.concatenate([state, action, z, time_feat, cat_feat], axis=-1)

#         for _ in range(3):
#             x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
#             x = nn.LayerNorm()(x)
#             x = nn.silu(x)

#         return nn.Dense(1, kernel_init=default_init())(x)


# class ScalarFlowVelocity(nn.Module):
#     """Generic scalar flow critic velocity: v(t, z, cond) -> dz/dt."""
#     hidden_dim: int = 256
#     time_embed_dim: int = 64

#     @nn.compact
#     def __call__(self, t, z, cond):
#         time_feat = FourierTimeEmbedding(self.time_embed_dim)(t)
#         x = jnp.concatenate([cond, z, time_feat], axis=-1)

#         for _ in range(3):
#             x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
#             x = nn.LayerNorm()(x)
#             x = nn.silu(x)

#         return nn.Dense(1, kernel_init=default_init())(x)


# class OneStepFlowActor(nn.Module):
#     """FloQ-style one-step flow actor: maps (obs, noise) -> action."""
#     hidden_dims: Sequence[int]
#     action_dim: int
#     layer_norm: bool = False

#     @nn.compact
#     def __call__(self, observations, noises, training: bool = False):
#         x = jnp.concatenate([observations, noises], axis=-1)
#         x = MLP(
#             hidden_dims=tuple(list(self.hidden_dims) + [self.action_dim]),
#             activations=mish,
#             use_layer_norm=self.layer_norm,
#             activate_final=False,
#         )(x, training=training)
#         return jnp.clip(x, -1.0, 1.0)


# class SafeFlowQAllFlowMatching(Agent):
#     # Q_flow
#     velocity: TrainState
#     velocity_target_params: dict

#     # Actor (flow matching policy)
#     actor: TrainState
#     actor_bc_flow: TrainState
#     target_actor_params: dict

#     # Safety critics (flow matching)
#     safe_q_flow: TrainState
#     safe_q_target_params: dict
#     safe_v_flow: TrainState
#     safe_v_target_params: dict

#     # Generic
#     discount: float
#     tau: float
#     actor_tau: float
#     critic_type: str = struct.field(pytree_node=False)
#     act_dim: int = struct.field(pytree_node=False)

#     # Flow/Q settings
#     ode_steps: int = struct.field(pytree_node=False)
#     q_samples: int = struct.field(pytree_node=False)
#     num_categories: int = struct.field(pytree_node=False)
#     base_dist_low: float
#     base_dist_high: float
#     critic_noise_samples: int = struct.field(pytree_node=False)

#     # Safety threshold
#     lambda_max: float
#     softplus_beta: float
#     qc_thres: float
#     eps_safe: float
#     eps_unsafe: float

#     # Actor/Q losses
#     awr_temperature: float
#     max_weight: float
#     bc_coef: float
#     bc_coef_unsafe: float

#     q_mean: float
#     q_std: float
#     q_norm_ema: float
#     noise_samples: int = struct.field(pytree_node=False)
#     q_conservative_alpha: float
#     y_norm_clip: float
#     unsafe_flow_weight: float

#     # Flow actor sampling
#     actor_flow_steps: int = struct.field(pytree_node=False)
#     actor_temperature: float
#     clip_sampler: bool = struct.field(pytree_node=False)
#     safe_value_clip_max: float
#     cost_scale: float
#     qc_log_restore_scale: float

#     @classmethod
#     def create(
#         cls,
#         seed: int,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Box,
#         actor_architecture: str = "mlp",
#         actor_lr: Union[float, optax.Schedule] = 3e-4,
#         critic_lr: float = 3e-4,
#         critic_hidden_dims: Sequence[int] = (256, 256),
#         actor_hidden_dims: Sequence[int] = (256, 256, 256),
#         actor_num_blocks: int = 2,
#         actor_weight_decay: Optional[float] = None,
#         actor_dropout_rate: Optional[float] = None,
#         actor_layer_norm: bool = False,
#         discount: float = 0.99,
#         tau: float = 0.005,
#         actor_tau: float = 0.005,
#         critic_type: str = "hj",
#         decay_steps: Optional[int] = int(2e6),
#         cost_limit: float = 10.0,
#         env_max_steps: int = 1000,
#         # Flow settings
#         hidden_dim: int = 256,
#         time_embed_dim: int = 64,
#         num_categories: int = 3,
#         ode_steps: int = 8,
#         q_samples: int = 16,
#         base_dist_low: float = -5.0,
#         base_dist_high: float = 5.0,
#         critic_noise_samples: int = 8,
#         lambda_max: float = 10.0,
#         softplus_beta: float = 1.0,
#         # Actor losses
#         awr_temperature: float = 3.0,
#         max_weight: float = 100.0,
#         bc_coef: float = 1.0,
#         bc_coef_unsafe: float = 3.0,
#         # Q normalization
#         q_norm_ema: float = 0.99,
#         # floq-style
#         noise_samples: int = 8,
#         q_conservative_alpha: float = 0.5,
#         y_norm_clip: float = 5.0,
#         unsafe_flow_weight: float = 1.0,
#         # flow actor
#         actor_flow_steps: int = 10,
#         actor_temperature: float = 1.0,
#         clip_sampler: bool = True,
#         safe_value_clip_max: Optional[float] = None,
#         cost_scale: float = 1.0,
#     ):
#         rng = jax.random.PRNGKey(seed)
#         rng, actor_key, qf_key, sq_key, sv_key = jax.random.split(rng, 5)
#         actions = action_space.sample()
#         observations = observation_space.sample()
#         action_dim = action_space.shape[0]

#         qc_thres = cost_limit * (1 - discount**env_max_steps) / (1 - discount) / env_max_steps
#         eps_safe = qc_thres * 0.3
#         eps_unsafe = qc_thres * 0.3
#         qc_log_restore_scale = cost_limit / jnp.maximum(qc_thres, 1e-6)
#         # Upper bound for cumulative cost critics (Vc/Qc).
#         # If not provided, use discounted horizon cumulative bound from cost_limit.
#         if safe_value_clip_max is None:
#             safe_value_clip_max = cost_limit * (1 - discount**env_max_steps) / (1 - discount)

#         if decay_steps is not None:
#             lr_schedule = optax.cosine_decay_schedule(actor_lr, decay_steps)
#         else:
#             lr_schedule = actor_lr

#         observations = jnp.expand_dims(observations, axis=0)
#         actions = jnp.expand_dims(actions, axis=0)

#         # Q_flow velocity
#         q_flow_net = VelocityNetwork(hidden_dim, num_categories, time_embed_dim)
#         dummy_t = jnp.zeros((1, 1))
#         dummy_z = jnp.zeros((1, 1))
#         dummy_cat = jnp.zeros((1,), dtype=jnp.int32)
#         q_flow_params = q_flow_net.init(qf_key, dummy_t, dummy_z, observations, actions, dummy_cat)["params"]
#         velocity = TrainState.create(
#             apply_fn=q_flow_net.apply,
#             params=q_flow_params,
#             tx=optax.adam(lr_schedule),
#         )

#         # Safety flow critics
#         sq_net = ScalarFlowVelocity(hidden_dim=critic_hidden_dims[-1], time_embed_dim=time_embed_dim)
#         sv_net = ScalarFlowVelocity(hidden_dim=critic_hidden_dims[-1], time_embed_dim=time_embed_dim)
#         sq_cond = jnp.concatenate([observations, actions], axis=-1)
#         sv_cond = observations
#         sq_params = sq_net.init(sq_key, dummy_t, dummy_z, sq_cond)["params"]
#         sv_params = sv_net.init(sv_key, dummy_t, dummy_z, sv_cond)["params"]
#         safe_q_flow = TrainState.create(apply_fn=sq_net.apply, params=sq_params, tx=optax.adam(critic_lr))
#         safe_v_flow = TrainState.create(apply_fn=sv_net.apply, params=sv_params, tx=optax.adam(critic_lr))

#         # FloQ-style actors:
#         # 1) actor_bc_flow: multi-step flow policy trained with flow-matching loss
#         # 2) actor (one-step): distilled from actor_bc_flow
#         preprocess_time_cls = partial(FourierFeatures, output_size=time_embed_dim, learnable=True)
#         cond_model_cls = partial(MLP, hidden_dims=(128, 128), activations=mish, activate_final=False)
#         if actor_architecture == "mlp":
#             base_model_cls = partial(
#                 MLP,
#                 hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
#                 activations=mish,
#                 use_layer_norm=actor_layer_norm,
#                 activate_final=False,
#             )
#         elif actor_architecture == "ln_resnet":
#             base_model_cls = partial(
#                 MLPResNet,
#                 use_layer_norm=actor_layer_norm,
#                 num_blocks=actor_num_blocks,
#                 dropout_rate=actor_dropout_rate,
#                 out_dim=action_dim,
#                 activations=mish,
#             )
#         else:
#             raise ValueError(f"Invalid actor architecture: {actor_architecture}")

#         actor_def = FlowMatching(
#             time_preprocess_cls=preprocess_time_cls,
#             cond_encoder_cls=cond_model_cls,
#             reverse_encoder_cls=base_model_cls,
#         )
#         actor_bc_params = actor_def.init(actor_key, observations, actions, dummy_t)["params"]
#         if not isinstance(actor_bc_params, flax.core.frozen_dict.FrozenDict):
#             actor_bc_params = freeze(actor_bc_params)
#         actor_bc_flow = TrainState.create(
#             apply_fn=actor_def.apply,
#             params=actor_bc_params,
#             tx=optax.adamw(
#                 learning_rate=lr_schedule,
#                 weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
#                 mask=get_weight_decay_mask,
#             ),
#         )

#         one_step_key = jax.random.fold_in(actor_key, 1)
#         one_step_def = OneStepFlowActor(
#             hidden_dims=actor_hidden_dims,
#             action_dim=action_dim,
#             layer_norm=actor_layer_norm,
#         )
#         actor_params = one_step_def.init(one_step_key, observations, actions)["params"]
#         if not isinstance(actor_params, flax.core.frozen_dict.FrozenDict):
#             actor_params = freeze(actor_params)
#         actor = TrainState.create(
#             apply_fn=one_step_def.apply,
#             params=actor_params,
#             tx=optax.adamw(
#                 learning_rate=lr_schedule,
#                 weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
#                 mask=get_weight_decay_mask,
#             ),
#         )

#         return cls(
#             velocity=velocity,
#             velocity_target_params=q_flow_params,
#             actor=actor,
#             actor_bc_flow=actor_bc_flow,
#             target_actor_params=actor_params,
#             safe_q_flow=safe_q_flow,
#             safe_q_target_params=sq_params,
#             safe_v_flow=safe_v_flow,
#             safe_v_target_params=sv_params,
#             discount=discount,
#             tau=tau,
#             actor_tau=actor_tau,
#             critic_type=critic_type,
#             rng=rng,
#             act_dim=action_dim,
#             ode_steps=ode_steps,
#             q_samples=q_samples,
#             num_categories=num_categories,
#             base_dist_low=base_dist_low,
#             base_dist_high=base_dist_high,
#             critic_noise_samples=critic_noise_samples,
#             lambda_max=lambda_max,
#             softplus_beta=softplus_beta,
#             qc_thres=qc_thres,
#             eps_safe=eps_safe,
#             eps_unsafe=eps_unsafe,
#             awr_temperature=awr_temperature,
#             max_weight=max_weight,
#             bc_coef=bc_coef,
#             bc_coef_unsafe=bc_coef_unsafe,
#             q_mean=0.0,
#             q_std=1.0,
#             q_norm_ema=q_norm_ema,
#             noise_samples=noise_samples,
#             q_conservative_alpha=q_conservative_alpha,
#             y_norm_clip=y_norm_clip,
#             unsafe_flow_weight=unsafe_flow_weight,
#             actor_flow_steps=actor_flow_steps,
#             actor_temperature=actor_temperature,
#             clip_sampler=clip_sampler,
#             safe_value_clip_max=safe_value_clip_max,
#             cost_scale=cost_scale,
#             qc_log_restore_scale=qc_log_restore_scale,
#         )

#     # ---------- Shared helpers ----------
#     def _classify(self, cost_value):
#         d = self.qc_thres
#         cat = jnp.ones_like(cost_value, dtype=jnp.int32)
#         cat = jnp.where(cost_value <= d - self.eps_safe, 0, cat)
#         cat = jnp.where(cost_value >= d + self.eps_unsafe, 2, cat)
#         return cat

#     def _positive_cost(self, x):
#         """Non-negative cost value mapping on integrated outputs (not on velocity)."""
#         return jax.nn.softplus(self.softplus_beta * x) / self.softplus_beta

#     def _integrate_scalar_flow(self, apply_fn, params, z0, cond, clip_low=None, clip_high=None):
#         K = self.ode_steps
#         dt = 1.0 / K

#         def step_fn(z, i):
#             t = jnp.full((z.shape[0], 1), (i + 0.5) * dt)
#             dz = apply_fn({"params": params}, t, z, cond)
#             return z + dt * dz, None

#         zf, _ = lax.scan(step_fn, z0, jnp.arange(K))
#         if clip_low is not None or clip_high is not None:
#             lo = -jnp.inf if clip_low is None else clip_low
#             hi = jnp.inf if clip_high is None else clip_high
#             zf = jnp.clip(zf, lo, hi)
#         return zf

#     def _estimate_scalar_flow_stats(self, apply_fn, params, cond, key, clip_low=None, clip_high=None):
#         m = self.q_samples
#         bs = cond.shape[0]
#         cond_exp = jnp.tile(cond[None], (m, 1, 1)).reshape(-1, cond.shape[-1])
#         z0 = jax.random.uniform(
#             key, (m * bs, 1), minval=self.base_dist_low, maxval=self.base_dist_high
#         )
#         zf = self._integrate_scalar_flow(
#             apply_fn, params, z0, cond_exp, clip_low=clip_low, clip_high=clip_high
#         ).reshape(m, bs)
#         return zf.mean(axis=0), zf.std(axis=0) + 1e-6

#     def _sample_actor_actions(self, actor_params, observations, key):
#         """Sample from one-step actor (FloQ-style deployment actor)."""
#         key, noise_key = jax.random.split(key)
#         noises = jax.random.normal(noise_key, (observations.shape[0], self.act_dim)) * self.actor_temperature
#         actions = self.actor.apply_fn({"params": actor_params}, observations, noises, training=False)
#         actions = jnp.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
#         if self.clip_sampler:
#             actions = jnp.clip(actions, -1.0, 1.0)
#         return actions, key

#     def _sample_bc_flow_actions(self, actor_bc_params, observations, key):
#         """Sample by integrating actor_bc_flow (teacher actor)."""
#         actions, key = flow_matching_sampler(
#             self.actor_bc_flow.apply_fn,
#             actor_bc_params,
#             self.actor_flow_steps,
#             key,
#             self.act_dim,
#             observations,
#             temperature=self.actor_temperature,
#             clip_sampler=self.clip_sampler,
#         )
#         actions = jnp.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
#         actions = jnp.clip(actions, -1.0, 1.0)
#         return actions, key

#     # ---------- Safety critic pretraining (all flow matching) ----------
#     def update_vc(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
#         # IQL logic:
#         #   Vc <- expectile( Qc_target(s,a_data) - Vc(s) )
#         rng, k_q, k_v = jax.random.split(self.rng, 3)
#         cond_q = jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)
#         qc_mean, _ = self._estimate_scalar_flow_stats(
#             self.safe_q_flow.apply_fn,
#             self.safe_q_target_params,
#             cond_q,
#             k_q,
#             clip_high=self.safe_value_clip_max,
#         )
#         qc_mean = self._positive_cost(qc_mean)

#         def loss_fn(params):
#             vc_raw_mean, _ = self._estimate_scalar_flow_stats(
#                 self.safe_v_flow.apply_fn,
#                 params,
#                 batch["observations"],
#                 k_v,
#                 clip_high=self.safe_value_clip_max,
#             )
#             vc_mean = self._positive_cost(vc_raw_mean)
#             diff = qc_mean - vc_mean
#             value_loss = safe_expectile_loss(diff, expectile=0.8).mean()
#             restore = self.qc_log_restore_scale
#             return value_loss, {
#                 "safe_value_loss": value_loss,
#                 # Main logs: restored to cumulative-budget physical scale.
#                 "vc": (vc_mean * restore).mean(),
#                 "vc_max": (vc_mean * restore).max(),
#                 "vc_min": (vc_mean * restore).min(),
#                 # Training-scale aliases used by optimization.
#                 "vc_scaled": vc_mean.mean(),
#                 "vc_scaled_max": vc_mean.max(),
#                 "vc_scaled_min": vc_mean.min(),
#                 "qc_target_mean": (qc_mean * restore).mean(),
#                 "qc_target_mean_scaled": qc_mean.mean(),
#             }

#         grads, info = jax.grad(loss_fn, has_aux=True)(self.safe_v_flow.params)
#         safe_v_flow = self.safe_v_flow.apply_gradients(grads=grads)
#         safe_v_target = optax.incremental_update(safe_v_flow.params, self.safe_v_target_params, self.tau)
#         return self.replace(safe_v_flow=safe_v_flow, safe_v_target_params=safe_v_target, rng=rng), info

#     def update_qc(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
#         # IQL/HJ logic:
#         #   target_qc <- costs + gamma * Vc(next)  (or HJ variant)
#         #   Qc <- mse( Qc(s,a_data), target_qc )
#         rng, k_v, k_q = jax.random.split(self.rng, 3)
#         v_mean, _ = self._estimate_scalar_flow_stats(
#             self.safe_v_flow.apply_fn,
#             self.safe_v_target_params,
#             batch["next_observations"],
#             k_v,
#             clip_high=self.safe_value_clip_max,
#         )
#         v_mean = self._positive_cost(v_mean)

#         if self.critic_type == "hj":
#             qc_nonterminal = (1.0 - self.discount) * batch["costs"] + self.discount * jnp.maximum(batch["costs"], v_mean)
#             target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1.0 - batch["masks"])
#         elif self.critic_type == "qc":
#             target_qc = batch["costs"] + self.discount * batch["masks"] * v_mean
#         else:
#             raise ValueError(f"Invalid critic_type: {self.critic_type}")

#         def loss_fn(params):
#             cond = jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)
#             qc_mean, _ = self._estimate_scalar_flow_stats(
#                 self.safe_q_flow.apply_fn,
#                 params,
#                 cond,
#                 k_q,
#                 clip_high=self.safe_value_clip_max,
#             )
#             qc_mean = self._positive_cost(qc_mean)
#             critic_loss = ((qc_mean - target_qc) ** 2).mean()
#             restore = self.qc_log_restore_scale
#             return critic_loss, {
#                 "safe_critic_loss": critic_loss,
#                 # Main logs: restored to cumulative-budget physical scale.
#                 "qc": (qc_mean * restore).mean(),
#                 "qc_max": (qc_mean * restore).max(),
#                 "qc_min": (qc_mean * restore).min(),
#                 "target_qc_mean": (target_qc * restore).mean(),
#                 # Training-scale aliases used by optimization.
#                 "qc_scaled": qc_mean.mean(),
#                 "qc_scaled_max": qc_mean.max(),
#                 "qc_scaled_min": qc_mean.min(),
#                 "target_qc_mean_scaled": target_qc.mean(),
#             }

#         grads, info = jax.grad(loss_fn, has_aux=True)(self.safe_q_flow.params)
#         safe_q_flow = self.safe_q_flow.apply_gradients(grads=grads)
#         safe_q_target = optax.incremental_update(safe_q_flow.params, self.safe_q_target_params, self.tau)
#         return self.replace(safe_q_flow=safe_q_flow, safe_q_target_params=safe_q_target, rng=rng), info

#     # ---------- Main update ----------
#     def update_flow_q_and_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
#         rng = self.rng
#         rng, k_pi, k_next, k_flow_z, k_flow_t, k_q, k_actor = jax.random.split(rng, 7)
#         k_q1, k_q2, k_q3, k_q4, k_q5, k_q6 = jax.random.split(k_q, 6)

#         observations = batch["observations"]
#         actions = batch["actions"]
#         rewards = batch["rewards"]
#         costs = batch["costs"]
#         next_observations = batch["next_observations"]
#         masks = batch["masks"]
#         B = observations.shape[0]
#         d = self.qc_thres

#         cond_data = jnp.concatenate([observations, actions], axis=-1)
#         qc_data_mean, _ = self._estimate_scalar_flow_stats(
#             self.safe_q_flow.apply_fn,
#             self.safe_q_target_params,
#             cond_data,
#             k_q1,
#             clip_high=self.safe_value_clip_max,
#         )
#         qc_data_mean = self._positive_cost(qc_data_mean)
#         category_data = self._classify(qc_data_mean)

#         lam = jax.nn.softplus(self.softplus_beta * (qc_data_mean - d)) / self.softplus_beta
#         lam = jnp.clip(lam, 0.0, self.lambda_max)
#         safe_mask = category_data == 0
#         unsafe_mask = category_data == 2
#         r_tilde = jnp.where(safe_mask, rewards, jnp.where(unsafe_mask, -costs, rewards - lam * costs))

#         next_a, _ = self._sample_actor_actions(self.target_actor_params, next_observations, k_next)
#         cond_next = jnp.concatenate([next_observations, next_a], axis=-1)
#         qc_next_mean, _ = self._estimate_scalar_flow_stats(
#             self.safe_q_flow.apply_fn,
#             self.safe_q_target_params,
#             cond_next,
#             k_q2,
#             clip_high=self.safe_value_clip_max,
#         )
#         qc_next_mean = self._positive_cost(qc_next_mean)
#         category_next = self._classify(qc_next_mean)

#         def ode_integrate_qflow(params, z0, s, a, cat):
#             K = self.ode_steps
#             dt = 1.0 / K
#             def step_fn(z, i):
#                 t = jnp.full((z.shape[0], 1), (i + 0.5) * dt)
#                 dz = self.velocity.apply_fn({"params": params}, t, z, s, a, cat)
#                 return z + dt * dz, None
#             zf, _ = lax.scan(step_fn, z0, jnp.arange(K))
#             return jnp.clip(zf, -self.y_norm_clip * 2, self.y_norm_clip * 2)

#         def estimate_qflow_stats(params, s, a, cat, key):
#             m = self.q_samples
#             bs = s.shape[0]
#             s_exp = jnp.tile(s[None], (m, 1, 1)).reshape(-1, s.shape[-1])
#             a_exp = jnp.tile(a[None], (m, 1, 1)).reshape(-1, a.shape[-1])
#             c_exp = jnp.tile(cat[None], (m, 1)).reshape(-1)
#             z0 = jax.random.uniform(key, (m * bs, 1), minval=self.base_dist_low, maxval=self.base_dist_high)
#             zf = ode_integrate_qflow(params, z0, s_exp, a_exp, c_exp).reshape(m, bs)
#             return zf.mean(axis=0), zf.std(axis=0) + 1e-6

#         v_next_mean, v_next_std = estimate_qflow_stats(
#             self.velocity_target_params, next_observations, next_a, category_next, k_q3
#         )
#         v_next_cons = jnp.clip(v_next_mean - self.q_conservative_alpha * v_next_std, -3.0, 3.0)

#         y_abs_bound = self.lambda_max / (1.0 - self.discount)
#         batch_mean = jnp.clip(r_tilde.mean(), -y_abs_bound, y_abs_bound)
#         batch_std = jnp.clip(jnp.maximum(r_tilde.std(), 1e-6), 1e-6, y_abs_bound)
#         new_q_mean = self.q_norm_ema * self.q_mean + (1 - self.q_norm_ema) * batch_mean
#         new_q_std = self.q_norm_ema * self.q_std + (1 - self.q_norm_ema) * batch_std

#         r_tilde_norm = jnp.clip((r_tilde - new_q_mean) / jnp.maximum(new_q_std, 1e-6), -self.y_norm_clip, self.y_norm_clip)
#         y_norm = jnp.clip(r_tilde_norm + self.discount * masks * v_next_cons, -self.y_norm_clip, self.y_norm_clip)

#         unsafe_w = jnp.where(category_data == 2, self.unsafe_flow_weight, 1.0)

#         def flow_loss_fn(params):
#             n = self.noise_samples
#             obs_rep = jnp.repeat(observations, n, axis=0)
#             act_rep = jnp.repeat(actions, n, axis=0)
#             cat_rep = jnp.repeat(category_data, n, axis=0)
#             y_rep = jnp.repeat(y_norm, n, axis=0)
#             w_rep = jnp.repeat(unsafe_w, n, axis=0)
#             z0 = jax.random.uniform(k_flow_z, (obs_rep.shape[0], 1), minval=self.base_dist_low, maxval=self.base_dist_high)
#             t = jax.random.uniform(k_flow_t, (obs_rep.shape[0], 1))
#             z_t = (1.0 - t) * z0 + t * y_rep[:, None]
#             pred = self.velocity.apply_fn({"params": params}, t, z_t, obs_rep, act_rep, cat_rep)
#             target = y_rep[:, None] - z0
#             loss = ((pred - target) ** 2).squeeze(-1)
#             return (w_rep * loss).mean()

#         flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(self.velocity.params)
#         new_velocity = self.velocity.apply_gradients(grads=flow_grads)
#         new_vel_target = optax.incremental_update(new_velocity.params, self.velocity_target_params, self.actor_tau)

#         # Actor AWR-style advantage targets (computed once outside actor grad for speed/stability).
#         q_data_mean, q_data_std = estimate_qflow_stats(
#             self.velocity_target_params, observations, actions, category_data, k_q4
#         )
#         q_data = (q_data_mean - self.q_conservative_alpha * q_data_std) * new_q_std + new_q_mean

#         a_pi, _ = self._sample_actor_actions(self.target_actor_params, observations, k_pi)
#         cond_actor = jnp.concatenate([observations, a_pi], axis=-1)
#         qc_actor_mean, _ = self._estimate_scalar_flow_stats(
#             self.safe_q_flow.apply_fn,
#             self.safe_q_target_params,
#             cond_actor,
#             k_q5,
#             clip_high=self.safe_value_clip_max,
#         )
#         qc_actor_mean = self._positive_cost(qc_actor_mean)
#         category_actor = self._classify(qc_actor_mean)
#         v_s_mean, v_s_std = estimate_qflow_stats(
#             self.velocity_target_params, observations, a_pi, category_actor, k_q6
#         )
#         v_s = (v_s_mean - self.q_conservative_alpha * v_s_std) * new_q_std + new_q_mean
#         adv = q_data - v_s
#         adv_norm = (adv - adv.mean()) / (adv.std() + 1e-6)
#         weights = jax.lax.stop_gradient(
#             jnp.clip(jnp.exp(adv_norm * self.awr_temperature), 0.0, self.max_weight)
#         )

#         # 1) Update BC flow actor with existing flow_matching_loss utility
#         key_bc_noise, key_bc_t, key_bc_do, key_os_noise, key_teacher = jax.random.split(k_actor, 5)
#         bc_noises = jax.random.normal(key_bc_noise, (B, self.act_dim))
#         bc_times = jax.random.uniform(key_bc_t, (B, 1))

#         def actor_bc_loss_fn(params):
#             bc_loss = flow_matching_loss(
#                 apply_fn=self.actor_bc_flow.apply_fn,
#                 params=params,
#                 observations=observations,
#                 actions=actions,
#                 time=bc_times,
#                 noise_sample=bc_noises,
#                 rng=key_bc_do,
#                 training=True,
#             )
#             return bc_loss, {"bc_flow_loss": bc_loss}

#         (bc_flow_loss, bc_info), bc_grads = jax.value_and_grad(actor_bc_loss_fn, has_aux=True)(self.actor_bc_flow.params)
#         new_actor_bc_flow = self.actor_bc_flow.apply_gradients(grads=bc_grads)

#         # 2) Distill one-step actor from BC flow actor + policy gradient from Q_flow.
#         os_noises = jax.random.normal(key_os_noise, (B, self.act_dim))
#         target_flow_actions, _ = self._sample_bc_flow_actions(new_actor_bc_flow.params, observations, key_teacher)

#         def one_step_actor_loss_fn(params):
#             pred_actions = self.actor.apply_fn({"params": params}, observations, os_noises, training=True)
#             pred_actions = jnp.nan_to_num(pred_actions, nan=0.0, posinf=1.0, neginf=-1.0)
#             pred_actions = jnp.clip(pred_actions, -1.0, 1.0)
#             distill_per_sample = jnp.mean((pred_actions - target_flow_actions) ** 2, axis=-1)
#             distill = jnp.mean(weights * distill_per_sample)

#             cond_pred = jnp.concatenate([observations, pred_actions], axis=-1)
#             qc_pred_mean, _ = self._estimate_scalar_flow_stats(
#                 self.safe_q_flow.apply_fn,
#                 self.safe_q_target_params,
#                 cond_pred,
#                 k_q3,
#                 clip_high=self.safe_value_clip_max,
#             )
#             qc_pred_mean = self._positive_cost(qc_pred_mean)
#             cat_pred = self._classify(qc_pred_mean)
#             q_pred_mean, q_pred_std = estimate_qflow_stats(
#                 self.velocity_target_params, observations, pred_actions, cat_pred, k_q2
#             )
#             q_pred = (q_pred_mean - self.q_conservative_alpha * q_pred_std) * new_q_std + new_q_mean
#             q_loss = -q_pred.mean()

#             total = self.bc_coef * distill + q_loss
#             return total, {
#                 "actor_loss": total,
#                 "distill_loss": distill,
#                 "q_loss": q_loss,
#                 "q_pred_mean": q_pred.mean(),
#                 "qc_actor_mean": qc_pred_mean.mean(),
#             }

#         (actor_loss, actor_info), actor_grads = jax.value_and_grad(one_step_actor_loss_fn, has_aux=True)(self.actor.params)
#         new_actor = self.actor.apply_gradients(grads=actor_grads)
#         new_target_actor = optax.incremental_update(new_actor.params, self.target_actor_params, self.actor_tau)

#         new_agent = self.replace(
#             velocity=new_velocity,
#             velocity_target_params=new_vel_target,
#             actor=new_actor,
#             actor_bc_flow=new_actor_bc_flow,
#             target_actor_params=new_target_actor,
#             q_mean=new_q_mean,
#             q_std=new_q_std,
#             rng=rng,
#         )
#         info = {
#             "flow_loss": flow_loss,
#             "actor_loss": actor_info["actor_loss"],
#             "bc_flow_loss": bc_info["bc_flow_loss"],
#             "distill_loss": actor_info["distill_loss"],
#             "q_loss": actor_info["q_loss"],
#             "q_pred_mean": actor_info["q_pred_mean"],
#             "adv_mean": adv.mean(),
#             "adv_std": adv.std(),
#             "lambda_mean": lam.mean(),
#             "qc_data_mean": qc_data_mean.mean(),
#             "qc_actor_mean": qc_actor_mean.mean(),
#             "q_mean": new_q_mean,
#             "q_std": new_q_std,
#             "r_tilde_mean": r_tilde.mean(),
#             "y_norm_mean": y_norm.mean(),
#         }
#         return new_agent, info

#     def eval_actions(self, observations: jnp.ndarray):
#         # Evaluate with a single one-step actor action.
#         rng = self.rng
#         assert len(observations.shape) == 1
#         obs = jax.device_put(observations)
#         obs = jnp.expand_dims(obs, axis=0)
#         actions, rng = self._sample_actor_actions(self.target_actor_params, obs, rng)
#         action = actions[0]
#         action_np = np.asarray(action)
#         if np.isnan(action_np).any():
#             print("[eval_actions] NaN action detected:", action_np)
#         return action_np, self.replace(rng=rng)

#     @jax.jit
#     def update(self, batch: DatasetDict):
#         new_agent, info = self.update_flow_q_and_actor(batch)
#         return new_agent, info

#     @jax.jit
#     def update_cost_critics(self, batch: DatasetDict):
#         new_agent, vc_info = self.update_vc(batch)
#         new_agent, qc_info = new_agent.update_qc(batch)
#         return new_agent, {**vc_info, **qc_info}

#     def save(self, modeldir, save_time):
#         file_name = "model" + str(save_time) + ".pickle"
#         state_dict = flax.serialization.to_state_dict(self)
#         pickle.dump(state_dict, open(os.path.join(modeldir, file_name), "wb"))

#     def load(self, model_location):
#         pkl_file = pickle.load(open(model_location, "rb"))
#         return flax.serialization.from_state_dict(target=self, state=pkl_file)
