"""Flow Matching Policy for continuous control."""
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    embed_dim: int

    @nn.compact
    def __call__(self, t):
        half_dim = self.embed_dim // 2
        embeddings = jnp.log(10000.0) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        return embeddings


class FlowMatching(nn.Module):
    """
    Flow Matching Policy Network

    Similar to DDPM but predicts velocity field instead of noise.
    Args:
        time_preprocess_cls: Time embedding network
        cond_encoder_cls: Observation/condition encoder
        reverse_encoder_cls: Main velocity prediction network
    """
    time_preprocess_cls: Callable
    cond_encoder_cls: Callable
    reverse_encoder_cls: Callable

    @nn.compact
    def __call__(self, observations, actions, time, training: bool = False, **kwargs):
        """
        Args:
            observations: (batch, obs_dim)
            actions: (batch, action_dim) - current action in flow (a_t)
            time: (batch, 1) - time in [0, 1]
            training: bool - whether in training mode

        Returns:
            velocity: (batch, action_dim) - predicted velocity v(a_t, t, obs)
        """
        # Time embedding
        t_ff = self.time_preprocess_cls()(time)

        # Observation encoding
        cond_encoded = self.cond_encoder_cls()(observations)

        # Concatenate: action + time_embed + obs_embed
        x = jnp.concatenate([actions, t_ff, cond_encoded], axis=-1)

        # Predict velocity through main network
        velocity = self.reverse_encoder_cls()(x, training=training)

        return velocity


def flow_matching_sampler(
    apply_fn,
    params,
    num_steps: int,
    rng,
    action_dim: int,
    observations,
    temperature: float = 1.0,
    clip_sampler: bool = True,
):
    """
    Sample actions using ODE integration for flow matching.

    Args:
        apply_fn: Policy network apply function
        params: Policy network parameters
        num_steps: Number of ODE integration steps
        rng: JAX random key
        action_dim: Action dimension
        observations: Batch of observations
        temperature: Sampling temperature (for initial noise)
        clip_sampler: Whether to clip actions to [-1, 1]

    Returns:
        actions: Sampled actions (batch, action_dim)
        rng: Updated random key
    """
    batch_size = observations.shape[0]

    # Initial noise: a_0 ~ N(0, temperature * I)
    rng, noise_key = jax.random.split(rng)
    a_t = jax.random.normal(noise_key, (batch_size, action_dim)) * temperature

    # ODE integration: da/dt = v(a_t, t, obs)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = jnp.ones((batch_size, 1)) * (step * dt)

        # Predict velocity
        velocity = apply_fn(
            {'params': params},
            observations,
            a_t,
            t,
            training=False
        )

        # Euler step: a_{t+dt} = a_t + v(a_t, t) * dt
        a_t = a_t + velocity * dt

        # Optional: clip to action bounds
        if clip_sampler:
            a_t = jnp.clip(a_t, -1.0, 1.0)

    return a_t, rng


def flow_matching_loss(
    apply_fn,
    params,
    observations,
    actions,
    time,
    noise_sample,
    rng,
    training: bool = True
):
    """
    Compute flow matching loss.

    Flow matching objective: match velocity to (a_1 - a_0)
    where a_t = (1-t) * a_0 + t * a_1
          a_0 = noise, a_1 = data

    Args:
        apply_fn: Policy network apply function
        params: Policy network parameters
        observations: (batch, obs_dim)
        actions: (batch, action_dim) - target actions (a_1)
        time: (batch, 1) - sampled time in [0, 1]
        noise_sample: (batch, action_dim) - sampled noise (a_0)
        rng: JAX random key
        training: bool

    Returns:
        loss: scalar loss
    """
    # Interpolate: a_t = (1-t) * a_0 + t * a_1
    t_expanded = time  # Already (batch, 1)
    a_t = (1 - t_expanded) * noise_sample + t_expanded * actions

    # Target velocity: da/dt = a_1 - a_0
    target_velocity = actions - noise_sample

    # Predict velocity
    pred_velocity = apply_fn(
        {'params': params},
        observations,
        a_t,
        time,
        rngs={'dropout': rng},
        training=training
    )

    # MSE loss
    loss = ((pred_velocity - target_velocity) ** 2).sum(axis=-1).mean()

    return loss
