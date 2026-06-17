"""Check JAX actor inference against the original Torch checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dial_mpc.g1_wbc_jax.constants import ACTION_DIM, OBS_DIM
from dial_mpc.g1_wbc_jax.policy import load_torch_actor, resolve_checkpoint_path, actor_forward


def _torch_forward(checkpoint: Path, obs: np.ndarray) -> np.ndarray:
    import torch

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = data.get("actor_state_dict", data)
    x = torch.from_numpy(obs.astype(np.float32))
    x = (x - state_dict["obs_normalizer._mean"]) / (state_dict["obs_normalizer._std"] + 1.0e-2)
    module_idx = 0
    linear_idx = 0
    dims = (OBS_DIM, 2048, 2048, 1024, 1024, 512, 256, 128, ACTION_DIM)
    while linear_idx < len(dims) - 1:
        w = state_dict[f"mlp.{module_idx}.weight"]
        b = state_dict[f"mlp.{module_idx}.bias"]
        x = x @ w.T + b
        if linear_idx < len(dims) - 2:
            x = torch.nn.functional.elu(x)
        linear_idx += 1
        module_idx += 2
    return x.detach().cpu().numpy()


def run(checkpoint: str | Path, batch_size: int, seed: int) -> dict[str, float]:
    ckpt_path = resolve_checkpoint_path(checkpoint)
    jax_params = load_torch_actor(ckpt_path)

    import torch

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = data.get("actor_state_dict", data)
    obs_mean = state_dict["obs_normalizer._mean"].detach().cpu().numpy().astype(np.float32)
    obs_std = state_dict["obs_normalizer._std"].detach().cpu().numpy().astype(np.float32)

    rng = np.random.default_rng(seed)
    obs = obs_mean + obs_std * rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32)

    torch_out = _torch_forward(ckpt_path, obs)
    jax_out = np.asarray(actor_forward(jax_params, jnp.asarray(obs, dtype=jnp.float32)))
    jax_out = np.asarray(jax.device_get(jax_out))

    diff = jax_out - torch_out
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(torch_out), 1.0e-6)
    rel_diff = abs_diff / denom
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel_diff.max()),
        "mean_rel": float(rel_diff.mean()),
        "torch_mean": float(torch_out.mean()),
        "jax_mean": float(jax_out.mean()),
        "torch_std": float(torch_out.std()),
        "jax_std": float(jax_out.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="bc")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats = run(args.checkpoint, args.batch_size, args.seed)
    print(
        " ".join(
            f"{key}={value:.6e}" for key, value in stats.items()
        )
    )


if __name__ == "__main__":
    main()
