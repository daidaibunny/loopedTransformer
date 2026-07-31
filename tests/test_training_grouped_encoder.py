from types import SimpleNamespace

import torch
from torch import nn

from looped_vl.training.model import _encode_grouped_batches


class _FakeEncoder(nn.Module):
	def forward(self, values: torch.Tensor) -> SimpleNamespace:
		embeddings = torch.stack((values, values + 10), dim=-1)
		slots = embeddings[:, None, :]
		mean = values.float().mean()
		return SimpleNamespace(
			embeddings=embeddings,
			loop_slot_hidden_states=(slots, slots + 1),
			conditioning_eos_hidden_state=embeddings + 20,
			slot_hidden_states=slots,
			diagnostics={
				"fusion_gate": mean,
				"late_fusion_attention_entropy": mean + 1,
				"slot_pairwise_cosine": mean + 2,
				"recurrent_pass_cosine": (mean + 4, mean + 5),
				"recurrent_pass_relative_update": (mean + 6, mean + 7),
			},
		)


def test_grouped_encoder_restores_original_order_and_weights_diagnostics() -> None:
	encoder = _FakeEncoder()
	batches = (
		{"values": torch.tensor([1.0, 4.0])},
		{"values": torch.tensor([2.0, 3.0])},
	)
	indices = ((0, 3), (1, 2))

	output = _encode_grouped_batches(
		encoder=encoder,
		processed_batches=batches,
		original_indices=indices,
		total_rows=4,
	)

	assert output.embeddings[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
	assert output.slot_hidden_states[:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
	assert output.loop_slot_hidden_states[1][:, 0, 0].tolist() == [
		2.0,
		3.0,
		4.0,
		5.0,
	]
	assert output.conditioning_eos_hidden_state[:, 0].tolist() == [
		21.0,
		22.0,
		23.0,
		24.0,
	]
	assert output.diagnostics["fusion_gate"].item() == 2.5
	assert output.diagnostics["recurrent_pass_cosine"][0].item() == 6.5
