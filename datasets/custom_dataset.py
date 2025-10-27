import os
import sys
import torch as T
from torch.utils.data import Dataset
from collections import deque
import numpy as np
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from datasets.wcst import WCST
from misc.consts import *


class CustomWCSTDataset(Dataset):
    def __init__(
        self, context_length: int = 1, total_batches: int = 1000, sample_batch_size: int = 64,
        fixed_context: int | None = None, allow_switch: bool = True,
    ) -> None:
        super().__init__()

        if context_length < 1:
            raise Exception("[!] Context length cannot be less than 1")

        self.context_length = context_length
        self.total_batches = total_batches
        self.sample_batch_size = sample_batch_size
        self.fixed_context = fixed_context
        self.allow_switch = allow_switch

        # Each dataset instance gets its own WCST environment
        self.wcst = WCST(sample_batch_size)

        # Pre-generate samples for this dataset
        self.samples = self._generate_dataset()

    def _generate_dataset(self):
        samples = []
        prev_trails = deque(maxlen=self.context_length - 1)

        batches_since_switch = 0
        switch_threshold = random.randint(MIN_SWITCH_BATCHES, MAX_SWITCH_BATCHES)

        if self.fixed_context is not None:
            self.wcst.set_context(self.fixed_context)
            print(f"[Dataset Init] Fixed context: {self.wcst.category_feature}")

        for _ in range(self.total_batches):
            if self.allow_switch and batches_since_switch >= switch_threshold:
                prev_trails.clear()
                self.wcst.context_switch()
                print(f"[Dataset Init] Context switched -> {self.wcst.category_feature}")
                batches_since_switch = 0
                switch_threshold = random.randint(MIN_SWITCH_BATCHES, MAX_SWITCH_BATCHES)

            context_examples, new_trials = next(self.wcst.gen_batch())

            for i in range(self.sample_batch_size):
                example_context = T.tensor(np.array(context_examples[i]), dtype=T.long).flatten()
                new_trial = T.tensor(new_trials[i], dtype=T.long)

                sequence = T.cat([*prev_trails, example_context, new_trial[:-1]])
                target = new_trial[-1]

                samples.append([sequence, target])

                prev_trail = T.tensor([*example_context[:4], *new_trial, EOS_TOKEN])
                prev_trails.append(prev_trail)

            batches_since_switch += 1

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def add_padding(x: T.Tensor, max_length: int) -> T.Tensor:
    output = T.full((max_length,), PAD_TOKEN, dtype=x.dtype, device=x.device)
    output[:x.size(0)] = x
    return output


def wcst_collate_fn(batch):
    """Pads variable-length input sequences for transformer training."""
    sequences, targets = zip(*batch)

    max_len = max(seq.size(0) for seq in sequences)

    padded = T.stack([add_padding(seq, max_len) for seq in sequences])

    targets = T.tensor(targets, dtype=T.long, device=padded.device)

    return padded, targets


if __name__ == "__main__":
    dataset = CustomWCSTDataset(
        context_length=2, total_batches=10, sample_batch_size=4, allow_switch=True
    )

    for sequence, target in dataset:
        print(sequence, target)