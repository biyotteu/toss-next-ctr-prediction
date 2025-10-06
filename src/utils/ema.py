# src/utils/ema.py
from __future__ import annotations

import math, copy, torch
from typing import Dict, Iterable, Optional

@torch.no_grad()
def _to_fp32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32)

class ModelEMA:
    """
    Strong EMA for CTR models.
    - FP32 shadow params (precision & stability)
    - Warmup decay schedule (linear or cosine)
    - BN buffers: copy or track
    - Safe store/copy_to/restore for eval
    """
    def __init__(
        self,
        model: torch.nn.Module,
        base_decay: float = 0.999,
        warmup_steps: int = 0,
        warmup_type: str = "linear",     # ["linear","cosine","none"]
        update_after_step: int = 0,      # steps to skip before starting EMA
        update_interval: int = 1,        # update every N optimizer steps
        ema_on_buffers: str = "copy",    # ["copy","track"]
        offload_to_cpu: bool = False,    # keep shadows on CPU (VRAM↓, 약간 느림)
        pin_memory: bool = False,        # CPU offload 시 pinned mem
        param_filter: Optional[Iterable[str]] = None,  # update 제외할 파라미터 이름 패턴
    ):
        self.base_decay = float(base_decay)
        self.warmup_steps = int(max(0, warmup_steps))
        self.warmup_type = warmup_type
        self.update_after_step = int(max(0, update_after_step))
        self.update_interval = int(max(1, update_interval))
        self.ema_on_buffers = ema_on_buffers
        self.offload_to_cpu = bool(offload_to_cpu)
        self.pin_memory = bool(pin_memory)
        self.param_filter = set(param_filter or [])
        self.num_updates = 0  # 실제 EMA 업데이트 횟수

        # shadow storage
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}

        # capture named params/buffers (stable ordering)
        self._param_names = []
        self._buffer_names = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._param_names.append(n)
            t = _to_fp32(p)
            if self.offload_to_cpu:
                t = t.cpu().pin_memory() if self.pin_memory else t.cpu()
            self.shadow_params[n] = t.clone()

        for n, b in model.named_buffers():
            self._buffer_names.append(n)
            tb = _to_fp32(b)
            if self.offload_to_cpu:
                tb = tb.cpu().pin_memory() if self.pin_memory else tb.cpu()
            self.shadow_buffers[n] = tb.clone()

        # temp storage for store/restore
        self._saved_params = None
        self._saved_buffers = None

    # --------- decay schedule ---------
    def _decay_at(self, step: int) -> float:
        """Return decay for this update index (0-based)."""
        d = self.base_decay
        if self.warmup_steps <= 0 or self.warmup_type == "none":
            return d
        t = min(1.0, (step + 1) / self.warmup_steps)
        if self.warmup_type == "linear":
            # start fast (decay small) -> finish at base_decay
            # effective (1 - decay) ramps from 1 to (1 - base_decay)
            decay = 1.0 - (1.0 - d) * t
        elif self.warmup_type == "cosine":
            # cosine from 0 -> 1
            c = 0.5 * (1 + math.cos(math.pi * (1 - t)))
            decay = 1.0 - (1.0 - d) * c
        else:
            decay = d
        return float(max(0.0, min(1.0, decay)))

    # --------- main update ---------
    @torch.no_grad()
    def update(self, model: torch.nn.Module, global_step: int):
        """Call this AFTER optimizer.step()."""
        if global_step < self.update_after_step:
            return
        if (global_step - self.update_after_step) % self.update_interval != 0:
            return

        decay = self._decay_at(self.num_updates)
        one_minus = 1.0 - decay

        # params
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.param_filter:
                # skip if explicitly excluded
                continue
            sp = self.shadow_params[n]
            cur = _to_fp32(p)
            if self.offload_to_cpu:
                cur = cur.cpu()
            sp.mul_(decay).add_(cur, alpha=one_minus)

        # buffers
        if self.ema_on_buffers == "copy":
            for n, b in model.named_buffers():
                sb = self.shadow_buffers[n]
                cur = _to_fp32(b)
                if self.offload_to_cpu:
                    cur = cur.cpu()
                sb.copy_(cur)
        else:  # "track"
            for n, b in model.named_buffers():
                sb = self.shadow_buffers[n]
                cur = _to_fp32(b)
                if self.offload_to_cpu:
                    cur = cur.cpu()
                sb.mul_(decay).add_(cur, alpha=one_minus)

        self.num_updates += 1

    # --------- swap-in / swap-out for eval ---------
    @torch.no_grad()
    def store(self, model: torch.nn.Module):
        """Save current model weights/buffers (for later restore)."""
        self._saved_params = {n: p.detach().clone() for n, p in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        """Load EMA weights/buffers into model (dtype/device sync)."""
        # state_dict style for safety (handles prefixes)
        ema_state = {}
        model_state = model.state_dict()
        for n in model_state.keys():
            if n in self.shadow_params:
                t = self.shadow_params[n]
            elif n in self.shadow_buffers:
                t = self.shadow_buffers[n]
            else:
                # not tracked (e.g., buffers like num_batches_tracked) -> keep model's
                continue
            tgt = model_state[n]
            src = t if not self.offload_to_cpu else t.to(tgt.device, non_blocking=True)
            # cast back to model dtype
            ema_state[n] = src.to(dtype=tgt.dtype)
        model.load_state_dict({**model_state, **ema_state}, strict=False)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        """Restore model weights/buffers saved by store()."""
        if self._saved_params is None:
            return
        model.load_state_dict(self._saved_params, strict=True)
        self._saved_params = None

    # --------- serialization ---------
    def state_dict(self) -> Dict:
        return {
            "base_decay": self.base_decay,
            "warmup_steps": self.warmup_steps,
            "warmup_type": self.warmup_type,
            "update_after_step": self.update_after_step,
            "update_interval": self.update_interval,
            "ema_on_buffers": self.ema_on_buffers,
            "offload_to_cpu": self.offload_to_cpu,
            "pin_memory": self.pin_memory,
            "param_filter": list(self.param_filter),
            "num_updates": self.num_updates,
            "shadow_params": {k: v.cpu() if v.is_cuda else v for k, v in self.shadow_params.items()},
            "shadow_buffers": {k: v.cpu() if v.is_cuda else v for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state: Dict):
        self.base_decay = float(state["base_decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.warmup_type = state["warmup_type"]
        self.update_after_step = int(state["update_after_step"])
        self.update_interval = int(state["update_interval"])
        self.ema_on_buffers = state["ema_on_buffers"]
        self.offload_to_cpu = bool(state.get("offload_to_cpu", False))
        self.pin_memory = bool(state.get("pin_memory", False))
        self.param_filter = set(state.get("param_filter", []))
        self.num_updates = int(state.get("num_updates", 0))

        self.shadow_params = {k: v.clone().detach() for k, v in state["shadow_params"].items()}
        self.shadow_buffers = {k: v.clone().detach() for k, v in state["shadow_buffers"].items()}


def build_ema(model, cfg):
    if not cfg.get("ema", {}).get("enabled", False):
        return None
    ec = cfg["ema"]
    ema = ModelEMA(
        model=model,
        base_decay=float(ec.get("decay", 0.999)),
        warmup_steps=int(ec.get("warmup_steps", 0)),
        warmup_type=str(ec.get("warmup_type", "linear")),
        update_after_step=int(ec.get("update_after_step", 0)),
        update_interval=int(ec.get("update_interval", 1)),
        ema_on_buffers=str(ec.get("ema_on_buffers", "copy")),
        offload_to_cpu=bool(ec.get("offload_to_cpu", False)),
        pin_memory=bool(ec.get("pin_memory", False)),
        param_filter=ec.get("param_filter", []),
    )
    return ema
