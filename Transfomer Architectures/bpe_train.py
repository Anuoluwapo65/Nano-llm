import math
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tokenizers import Tokenizer
import matplotlib.pyplot as plt

from bpe_dataset import TranslationDataset
from bpe_tokenizer import load_train_bpe_forward
from train_full import Full_train, ModelConfig


def get_device(verbose: bool = False, seed: int = 42) -> torch.device:
    """Seed the available generators and return the best available device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Using device: {device}")
    return device


@dataclass
class TrainConfig:
    batch_size: int = 256
    gradient_steps: int = 8
    total_steps: int = 10_000
    warmup_steps: int = 1_000
    lr: float = 5e-3
    min_lr: float = 1e-5
    betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.98))
    weight_decay: float = 0.01
    optim_eps: float = 1e-9
    label_smoothing: float = 0.1
    bucket_size: int = 2_048
    debug: bool = False
    device: torch.device = field(default_factory=get_device)
    mixed_precision: bool = True
    eval_steps: int = 100
    max_seq_len: int = 128
    max_len: int = 128
    num_workers: int = 0


class BucketBatchSampler(Sampler[list[int]]):
    """Yield batches whose examples have similar sequence lengths."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size: int = 2_048,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0 or bucket_size <= 0:
            raise ValueError("batch_size and bucket_size must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            bucket_batches = [
                bucket[index : index + self.batch_size]
                for index in range(0, len(bucket), self.batch_size)
            ]
            if self.drop_last and bucket_batches and len(bucket_batches[-1]) < self.batch_size:
                bucket_batches.pop()
            batches.extend(bucket_batches)

        if self.shuffle:
            rng.shuffle(batches)
        self.seed += 1
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


class AdamOptimizer:
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        self.params = [parameter for parameter in params if parameter.requires_grad]
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "m": torch.zeros_like(parameter),
                "v": torch.zeros_like(parameter),
            }
            for parameter in self.params
        }

    def update_lr(self, new_lr: float) -> None:
        self.lr = new_lr

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = self.betas
        for parameter in self.params:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            state = self.state[parameter]
            state["step"] += 1
            state["m"].mul_(beta1).add_(gradient, alpha=1 - beta1)
            state["v"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            step = state["step"]
            m_hat = state["m"] / (1 - beta1**step)
            v_hat = state["v"] / (1 - beta2**step)
            update = m_hat / (torch.sqrt(v_hat) + self.eps)
            if self.weight_decay:
                update = update + self.weight_decay * parameter
            parameter.add_(update, alpha=-self.lr)

    def zero_grad(self) -> None:
        for parameter in self.params:
            if parameter.grad is not None:
                parameter.grad.detach_()
                parameter.grad.zero_()


def get_lr(cur_step: int, warmup_steps: int, model_dim: int) -> float:
    if cur_step <= 0 or warmup_steps <= 0 or model_dim <= 0:
        raise ValueError("cur_step, warmup_steps, and model_dim must be positive")
    return (model_dim**-0.5) * min(
        cur_step**-0.5,
        cur_step * warmup_steps**-1.5,
    )


def cross_entropy(logits, labels, ignore_index=0, label_smoothing=0.0):
    # Create mask for ignore_index
    mask = labels != ignore_index
    num_classes = logits.size(-1)

    if label_smoothing > 0.0:
        smooth_labels = F.one_hot(labels, num_classes).float()
        smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / num_classes
    else:
        smooth_labels = F.one_hot(labels, num_classes).float()

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(smooth_labels * log_probs, dim=-1)
    loss = loss * mask.float()
    loss = loss.sum() / mask.sum()

    return loss


def exp_loss(loss: float | torch.Tensor) -> float:
    return math.exp(float(loss))


def load_tokenizer(path: str = "bpe_bpe_joint.json") -> Tokenizer:
    tokenizer_path = Path(path)
    if tokenizer_path.is_file():
        return Tokenizer.from_file(str(tokenizer_path))
    return load_train_bpe_forward(vocab_size=10_000, save_prefix="bpe")


def build_dataloader(tokenizer: Tokenizer, config: TrainConfig) -> tuple[TranslationDataset, DataLoader]:
    dataset = TranslationDataset(
        file_prefix="Train",
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        is_debug=config.debug,
    )
    collate_fn = partial(
        TranslationDataset.translation_collate,
        pad_id=dataset.pad_id,
        sos_id=dataset.sos_id,
        eos_id=dataset.eos_id,
        max_len=config.max_len,
    )
    lengths = [target.numel() for target in dataset.target_texts]
    sampler = BucketBatchSampler(
        lengths,
        batch_size=config.batch_size,
        bucket_size=config.bucket_size,
        shuffle=True,
    )
    return dataset, DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.device.type == "cuda",
        collate_fn=collate_fn,
    )


def train(config: TrainConfig, tokenizer: Tokenizer) -> Full_train:
    dataset, dataloader = build_dataloader(tokenizer, config)
    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=config.max_seq_len,
    )
    model = Full_train(model_config).to(config.device)
    optimizer = AdamOptimizer(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.optim_eps,
    )
    model.train()
    batches = iter(dataloader)

    for step in range(1, config.total_steps + 1):
        optimizer.zero_grad()
        loss_step = 0.0
        for _ in range(config.gradient_steps):
            batch = next(batches)
            encoder_input_ids = batch["encoder_input_ids"].to(config.device)
            decoder_input_ids = batch["decoder_input_ids"].to(config.device)
            enc_mask = batch["encoder_mask"].to(config.device)
            dec_mask = batch["decoder_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            use_autocast = config.mixed_precision and config.device.type in {"cuda", "cpu"}
            with torch.autocast(
                device_type=config.device.type,
                enabled=use_autocast,
                dtype=torch.bfloat16,
            ):
                log_probs = model(
                    encoder_input_ids,
                    decoder_input_ids,
                    src_pad_mask=enc_mask,
                    tgt_pad_mask=dec_mask,
                )
                loss = cross_entropy(
                    log_probs,
                    labels,
                    ignore_index=dataset.pad_id,
                    label_smoothing=config.label_smoothing,
                )
                (loss / config.gradient_steps).backward()
            loss_step += float(loss.detach())

        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        lr = max(
            config.min_lr,
            get_lr(step, config.warmup_steps, model_config.model_dim),
        )
        optimizer.update_lr(lr)
        optimizer.step()
        print(f"{step}/{config.total_steps} - Loss: {loss_step:.4f} - LR: {lr:.7f}")

    return model


def plot_learning_rate(model_dim: int = 512, warmup_steps: int = 4_000) -> None:
    steps = np.arange(1, 200_001)
    learning_rates = [get_lr(int(step), warmup_steps, model_dim) for step in steps]
    frame = pd.DataFrame({"step": steps, "learning_rate": learning_rates})
    sns.set_theme(style="whitegrid", context="talk")
    axis = sns.lineplot(data=frame, x="step", y="learning_rate", linewidth=2.8)
    axis.axvline(warmup_steps, linestyle="--", linewidth=1.8)
    axis.set_title("Transformer learning-rate scheduler", pad=12)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Learning rate")
    sns.despine()
    axis.get_figure().tight_layout()
    plt.show()


if __name__ == "__main__":
    train_config = TrainConfig()
    tokenizer = load_tokenizer()
    trained_model = train(train_config, tokenizer)
    torch.save(trained_model.state_dict(), "transformer_translation_model.pth")
    print("Model saved to transformer_translation_model.pth")
