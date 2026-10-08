"""Weight access for the reference, by Hugging Face tensor names, always returned as float32.

Two sources share one interface:
  * SafetensorsWeights reads a checkpoint directory lazily and decodes one expert at a time, so the full model
    never has to be resident (the real checkpoint is 13.8 GB packed, ~84 GB as fp32).
  * DictWeights holds tensors in memory, for tests. It also accepts dense (unquantized) expert weights in Hugging
    Face's layout, which is how a tiny transformers model exports them.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from . import mxfp4


def _layer(i, rest):
    return f"model.layers.{i}.{rest}"


class Weights:
    """Interface. All returned tensors are fp32 and safe to modify."""

    def get(self, name):
        raise NotImplementedError

    def has(self, name):
        raise NotImplementedError

    def get_rows(self, name, index):
        """Rows `index` (a 1-D LongTensor) of a 2-D tensor, e.g. embedding lookups."""
        return self.get(name)[index]

    def expert(self, layer, e, device="cpu"):
        """(w1 [2I, H], b1 [2I], w2 [H, I], b2 [H]) for expert `e`, on `device`. Rows of w1 alternate glu (even)
        and linear (odd). MXFP4 blocks are moved to `device` before decoding, so a GPU decodes them itself."""
        p = _layer(layer, "mlp.experts.")
        raw = lambda name: self._raw_slice(p + name, e).to(device)
        if self.has(p + "gate_up_proj_blocks"):
            w1 = mxfp4.dequant(raw("gate_up_proj_blocks"), raw("gate_up_proj_scales"))
            w2 = mxfp4.dequant(raw("down_proj_blocks"), raw("down_proj_scales"))
        else:
            # Hugging Face dense layout stores [E, in, out]; transpose to [out, in] like the MXFP4 blocks.
            w1 = raw("gate_up_proj").to(torch.float32).t().contiguous()
            w2 = raw("down_proj").to(torch.float32).t().contiguous()
        return w1, raw("gate_up_proj_bias").to(torch.float32), w2, raw("down_proj_bias").to(torch.float32)

    def _raw_slice(self, name, e):
        raise NotImplementedError


class DictWeights(Weights):
    def __init__(self, tensors):
        self.t = dict(tensors)

    def has(self, name):
        return name in self.t

    def get(self, name):
        return self.t[name].to(torch.float32).clone()

    def _raw_slice(self, name, e):
        return self.t[name][e]


class SafetensorsWeights(Weights):
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            self.where = {}
            for f in sorted(self.dir.glob("*.safetensors")):
                with safe_open(f, framework="pt") as h:
                    self.where.update({k: f.name for k in h.keys()})
        self._open = {}

    def _handle(self, name):
        fname = self.where[name]
        if fname not in self._open:
            self._open[fname] = safe_open(self.dir / fname, framework="pt")
        return self._open[fname]

    def has(self, name):
        return name in self.where

    def get(self, name):
        return self._handle(name).get_tensor(name).to(torch.float32)

    def get_rows(self, name, index):
        # Gather rows without materializing the whole table as fp32 (the embedding is 201088 x 2880).
        sl = self._handle(name).get_slice(name)
        uniq, inverse = torch.unique(index, return_inverse=True)
        rows = torch.stack([sl[int(r)] for r in uniq.tolist()])
        return rows.to(torch.float32)[inverse]

    def _raw_slice(self, name, e):
        return self._handle(name).get_slice(name)[e]
