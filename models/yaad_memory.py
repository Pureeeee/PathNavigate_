import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class YaadMemory(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        lr: float = 0.05,
        huber_delta: float = 1.0,
        forget_alpha: float = 0.999,
        grad_clip: float = 5.0,
        warmup_steps: int = 100,
        threshold_std_scale: float = 1.0,
        reverse_update: bool = False,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = input_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.memory_mlp = nn.Sequential(*layers)

        self.lr = lr
        self.huber_delta = huber_delta
        self.forget_alpha = forget_alpha
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.threshold_std_scale = threshold_std_scale
        self.reverse_update = reverse_update

        self.surprise_threshold = None
        self.surprise_history: list = []
        self.step_count: int = 0

    def compute_and_update(self, feature: torch.Tensor) -> float:

        feature = feature.detach()
        self.step_count += 1

        self.memory_mlp.zero_grad()
        prediction = self.memory_mlp(feature)
        loss = F.huber_loss(prediction, feature, delta=self.huber_delta, reduction="mean")
        loss.backward()

        total_grad_norm_sq = 0.0
        for p in self.memory_mlp.parameters():
            if p.grad is not None:
                total_grad_norm_sq += p.grad.data.norm(2).item() ** 2
        surprise = total_grad_norm_sq ** 0.5

        self.surprise_history.append(surprise)

        if self.surprise_threshold is None and self.step_count >= self.warmup_steps:
            self._calibrate_threshold()

        if self.reverse_update:

            if self.surprise_threshold is not None and surprise <= self.surprise_threshold:
                self._apply_gradient_step()
            elif self.surprise_threshold is None:

                self._apply_gradient_step()

        else:

            if self.surprise_threshold is None or surprise > self.surprise_threshold:
                self._apply_gradient_step()
            else:
                self._apply_forget()

        return surprise

    def compute_surprise(self, feature: torch.Tensor) -> float:

        feature = feature.detach()
        self.memory_mlp.zero_grad()
        prediction = self.memory_mlp(feature)
        loss = F.huber_loss(prediction, feature, delta=self.huber_delta, reduction="mean")
        loss.backward()

        total_grad_norm_sq = 0.0
        for p in self.memory_mlp.parameters():
            if p.grad is not None:
                total_grad_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_grad_norm_sq ** 0.5

    def update_memory(self, feature: torch.Tensor, surprise: float) -> bool:

        if self.surprise_threshold is None or surprise > self.surprise_threshold:
            self._gradient_step_full(feature)
            return True
        else:
            self._apply_forget()
            return False

    def _apply_gradient_step(self):

        with torch.no_grad():
            for p in self.memory_mlp.parameters():
                if p.grad is not None:

                    grad = p.grad.data
                    grad_norm = grad.norm(2)
                    if grad_norm > self.grad_clip:
                        grad = grad * (self.grad_clip / grad_norm)
                    p.data -= self.lr * grad

    def _gradient_step_full(self, feature: torch.Tensor):

        self.memory_mlp.zero_grad()
        feature = feature.detach()
        pred = self.memory_mlp(feature)
        loss = F.huber_loss(pred, feature, delta=self.huber_delta, reduction="mean")
        loss.backward()
        self._apply_gradient_step()

    def _apply_forget(self):

        with torch.no_grad():
            for p in self.memory_mlp.parameters():
                p.data.mul_(self.forget_alpha)

    def _calibrate_threshold(self):

        arr = np.array(self.surprise_history[:self.warmup_steps])
        self.surprise_threshold = float(arr.mean() + self.threshold_std_scale * arr.std())

    def reset(self):

        for layer in self.memory_mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.surprise_threshold = None
        self.surprise_history = []
        self.step_count = 0

    def get_stats(self) -> dict:

        if not self.surprise_history:
            return {"count": 0}
        arr = np.array(self.surprise_history)
        high_count = int((arr > self.surprise_threshold).sum()) if self.surprise_threshold else 0
        return {
            "count": len(arr),
            "threshold": self.surprise_threshold,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "high_surprise_ratio": high_count / len(arr) if len(arr) > 0 else 0,
        }
