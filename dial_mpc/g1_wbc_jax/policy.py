"""Pure JAX WBC actor inference for G1 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dial_mpc.g1_wbc_jax.constants import ACTION_DIM, OBS_DIM, default_checkpoint_dirs


DEFAULT_HIDDEN_DIMS = (2048, 2048, 1024, 1024, 512, 256, 128)


class WbcActorParams(NamedTuple):
    obs_mean: jnp.ndarray
    obs_std: jnp.ndarray
    layers: tuple[tuple[jnp.ndarray, jnp.ndarray], ...]


@jax.jit
def actor_forward(params: WbcActorParams, obs: jnp.ndarray) -> jnp.ndarray:
    x = (obs - params.obs_mean) / (params.obs_std + 1.0e-2)
    for i, (weight, bias) in enumerate(params.layers):
        x = jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST) + bias
        if i < len(params.layers) - 1:
            x = jax.nn.elu(x)
    return x


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    aliases = default_checkpoint_dirs()
    checkpoint_key = str(checkpoint)
    if checkpoint_key in aliases:
        return _latest_checkpoint(aliases[checkpoint_key])
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return _latest_checkpoint(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_torch_actor(checkpoint: str | Path) -> WbcActorParams:
    """Load a SPIDER/tracking_bfm Torch actor checkpoint into JAX arrays.

    Torch is used only for deserializing the existing checkpoint. The returned
    params are ordinary JAX arrays and can be used by JIT-compiled rollout code.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading .pt WBC checkpoints requires torch.") from exc

    ckpt_path = resolve_checkpoint_path(checkpoint)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = data.get("actor_state_dict", data)
    obs_mean = _tensor_to_jax(state_dict["obs_normalizer._mean"]).reshape(1, OBS_DIM)
    obs_std = _tensor_to_jax(state_dict["obs_normalizer._std"]).reshape(1, OBS_DIM)

    layers = []
    linear_idx = 0
    module_idx = 0
    dims = (OBS_DIM, *DEFAULT_HIDDEN_DIMS, ACTION_DIM)
    while linear_idx < len(dims) - 1:
        w_key = f"mlp.{module_idx}.weight"
        b_key = f"mlp.{module_idx}.bias"
        if w_key not in state_dict or b_key not in state_dict:
            raise ValueError(f"Checkpoint {ckpt_path} is missing {w_key}/{b_key}.")
        weight = _tensor_to_jax(state_dict[w_key])
        bias = _tensor_to_jax(state_dict[b_key])
        expected = (dims[linear_idx + 1], dims[linear_idx])
        if weight.shape != expected:
            raise ValueError(f"{w_key} shape {weight.shape} != expected {expected}.")
        layers.append((weight, bias))
        linear_idx += 1
        module_idx += 2
    return WbcActorParams(obs_mean=obs_mean, obs_std=obs_std, layers=tuple(layers))


def save_actor_npz(params: WbcActorParams, path: str | Path) -> None:
    """Persist converted JAX actor weights without a Torch dependency."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "obs_mean": np.asarray(params.obs_mean),
        "obs_std": np.asarray(params.obs_std),
        "num_layers": np.asarray(len(params.layers), dtype=np.int32),
    }
    for i, (weight, bias) in enumerate(params.layers):
        payload[f"layer_{i}_weight"] = np.asarray(weight)
        payload[f"layer_{i}_bias"] = np.asarray(bias)
    np.savez(out, **payload)


def load_actor_npz(path: str | Path) -> WbcActorParams:
    raw = np.load(Path(path).expanduser())
    num_layers = int(raw["num_layers"].item())
    layers = tuple(
        (
            jnp.asarray(raw[f"layer_{i}_weight"], dtype=jnp.float32),
            jnp.asarray(raw[f"layer_{i}_bias"], dtype=jnp.float32),
        )
        for i in range(num_layers)
    )
    return WbcActorParams(
        obs_mean=jnp.asarray(raw["obs_mean"], dtype=jnp.float32),
        obs_std=jnp.asarray(raw["obs_std"], dtype=jnp.float32),
        layers=layers,
    )


def _tensor_to_jax(value) -> jnp.ndarray:
    return jnp.asarray(value.detach().cpu().numpy(), dtype=jnp.float32)


def _latest_checkpoint(directory: Path) -> Path:
    candidates = sorted(directory.expanduser().glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under {directory}")
    return candidates[-1].resolve()
