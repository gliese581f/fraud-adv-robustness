"""Choose the compute device (CPU or GPU) for the study's models.

Input: the environment variable FRAUDADV_DEVICE, when set (for example `cpu`, `cuda` or `cuda:1`). Without it,
the GPU is used when PyTorch sees one, otherwise the CPU.
Output: a torch.device, and `python -m fraudadv.device` prints which device was chosen.

Every result of the paper was computed on CPU, so this file affects no table; it only decides where training
and attacks run. On a CPU-only machine every model takes the same code path whatever this file says.

How each model family uses the device:
  * MLP, FT-Transformer: the network, training batches, logits and gradients go on the device.
  * HGNN, HGT: the network and the graph context go on the device when the context fits in GPU memory
    (otherwise these two families stay on CPU and print that they did).
  * Chronos-2, TimesFM: the frozen backbone goes on the device (it is the expensive part); the small trained
    head stays on CPU.
  * XGBoost: trains with device="cuda"; predictions stay on CPU, because the attacks query small batches.
  * CatBoost: trains on GPU only when asked (`zoo.CatBoost.fit(..., gpu=True)`, or
    `python -m fraudadv.hpo.train_all ... --catboost-gpu`), because CatBoost's GPU mode supports fewer options
    and grows different trees.
  * ivis (TensorFlow) picks a GPU by itself when the TensorFlow build supports one.

Models are always saved with CPU tensors, so a model trained on a GPU loads on a CPU-only machine and the reverse.
The GPU code paths have not been used for any result, so on first GPU use check that a loaded model's logits
reproduce its stored runs/<dataset>/<family>/test_scores_seed0.npy.
"""
from __future__ import annotations

import os


def torch_device():
    """The torch.device to compute on: FRAUDADV_DEVICE if set, else CUDA when available, else CPU."""
    import torch

    d = os.environ.get("FRAUDADV_DEVICE")
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_gpu() -> bool:
    """True when the chosen device is a GPU."""
    return torch_device().type == "cuda"


def describe() -> str:
    """One line naming the chosen device: the GPU's name and memory, or the CPU's thread count."""
    import torch

    dev = torch_device()
    if dev.type == "cuda":
        i = dev.index or 0
        return f"{dev} ({torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB)"
    return f"cpu ({torch.get_num_threads()} threads)"


if __name__ == "__main__":
    print(describe())
