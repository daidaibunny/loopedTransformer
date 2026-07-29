"""Project-wide deterministic seed initialization."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> torch.Generator:
	"""Set every required seed and return the DataLoader generator."""
	if seed != 42:
		raise ValueError("The v1.0 project seed must remain 42")
	os.environ["PYTHONHASHSEED"] = "42"
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	generator = torch.Generator()
	generator.manual_seed(seed)
	return generator
