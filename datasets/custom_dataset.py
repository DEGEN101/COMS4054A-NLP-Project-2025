import os
import sys
import torch as T
from torch.utils.data import Dataset
import numpy as np
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)

from datasets.wcst import WCST
from misc.constants import *


class CustomWCSTDataset(Dataset):
    def __init__(
        self, context_length: int = 1, total_batches: int = 1000, sample_batch_size: int = 64,
        fixed_context: int | None = None, allow_switch: bool = True,
    ) -> None:
        super().__init__()
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

        batches_since_switch = 0
        switch_threshold = random.randint(MIN_SWITCH_BATCHES, MAX_SWITCH_BATCHES)

        if self.fixed_context is not None:
            self.wcst.set_context(self.fixed_context)
            print(f"[Dataset Init] Fixed context: {self.wcst.category_feature}")

        for _ in range(self.total_batches):
            if self.allow_switch and batches_since_switch >= switch_threshold:
                self.wcst.context_switch()
                print(f"[Dataset Init] Context switched -> {self.wcst.category_feature}")
                batches_since_switch = 0
                switch_threshold = random.randint(MIN_SWITCH_BATCHES, MAX_SWITCH_BATCHES)

            context_examples, new_trials = next(self.wcst.gen_batch())

            for i in range(self.sample_batch_size):
                example_context = T.tensor(np.array(context_examples[i]), dtype=T.long).flatten()
                new_trial = T.tensor(new_trials[i], dtype=T.long)
                samples.append([T.cat([example_context, new_trial[:-1]]), new_trial[-1]])

            batches_since_switch += 1

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
