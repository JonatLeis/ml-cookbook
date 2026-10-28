"""Five FP32 training steps on transformer model: one transformer block, no dropout."""
from dataclasses import dataclass

import torch
from torch import nn

from experiment import Experiment
from experiments._utils import cpu_generator
from validation import ValidationError


class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.tokens = nn.Embedding(config.vocab_size, width)
        self.positions = nn.Embedding(config.length, width)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config.vocab_size, bias=False)

    def forward(self, tokens):
        batch, length = tokens.shape
        x = self.tokens(tokens) + self.positions(torch.arange(length, device=tokens.device))
        width = x.shape[-1]
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        mixed = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
        x = x + self.projection(mixed.transpose(1, 2).contiguous().reshape(batch, length, width))
        x = x + self.mlp(self.norm2(x))
        return self.output(self.final_norm(x))


@dataclass
class TransformerTraining(Experiment):
    steps: int = 5
    batch_size: int = 2
    length: int = 16
    vocab_size: int = 64
    width: int = 32
    heads: int = 2
    learning_rate: float = 0.001

    def run(self, device, seed):
        if self.steps < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("steps must be positive and width divisible by positive heads")
        generator = cpu_generator(seed)
        sequence = torch.randint(self.vocab_size, (self.batch_size, self.length + 1),
                                 generator=generator)
        tokens, targets = sequence[:, :-1].to(device), sequence[:, 1:].to(device)
        # Explicit seeding also makes direct Python use reproducible.
        torch.manual_seed(seed)
        model = TinyTransformer(self).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,
                                      weight_decay=0.01, foreach=False, fused=False)
        results = {"tokens": tokens.cpu().clone(), "targets": targets.cpu().clone()}
        for name, parameter in model.named_parameters():
            results[f"initial/{name}"] = parameter.detach().cpu().clone()
        for step in range(1, self.steps + 1):
            prefix = f"step_{step:03d}"
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits.reshape(-1, self.vocab_size),
                                               targets.reshape(-1))
            loss.backward()
            results[f"{prefix}/logits"] = logits.detach().cpu().clone()
            results[f"{prefix}/loss"] = loss.detach().cpu().clone()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise ValidationError(f"missing gradient: {name}")
                results[f"{prefix}/gradient/{name}"] = parameter.grad.detach().cpu().clone()
            optimizer.step()
            for name, parameter in model.named_parameters():
                results[f"{prefix}/parameter/{name}"] = parameter.detach().cpu().clone()
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    results[f"{prefix}/optimizer/{name}/{field}"] = (
                        optimizer.state[parameter][field].detach().cpu().clone()
                    )
        return results

    def validate(self, results):
        super().validate(results)
        names = [key.removeprefix("initial/") for key in results if key.startswith("initial/")]
        for step in range(1, self.steps + 1):
            prefix = f"step_{step:03d}"
            previous = "initial" if step == 1 else f"step_{step - 1:03d}/parameter"
            if not any(torch.count_nonzero(results[f"{prefix}/gradient/{name}"]).item()
                       for name in names):
                raise ValidationError(f"{prefix}: all gradients are zero")
            if not any(not torch.equal(results[f"{previous}/{name}"],
                                       results[f"{prefix}/parameter/{name}"]) for name in names):
                raise ValidationError(f"{prefix}: parameters did not change")
            for name in names:
                if results[f"{prefix}/optimizer/{name}/step"].item() != step:
                    raise ValidationError(f"{prefix}: incorrect AdamW step for {name}")


EXPERIMENT = TransformerTraining()
