from looped_vl.data_check import representative_indices


def test_representative_indices_cover_first_middle_and_last_mixture_blocks() -> None:
	indices = representative_indices(1_000_000)

	assert indices == [0, 10, 17, 500_000, 500_010, 500_017, 999_980, 999_990, 999_997]
