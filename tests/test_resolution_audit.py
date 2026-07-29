from looped_vl.resolution_audit import summarize_resolutions


def test_summarize_resolutions_reports_source_and_token_distributions() -> None:
	records = [
		{"source": "coco", "raw_width": 640, "raw_height": 480},
		{"source": "gqa_balanced", "raw_width": 500, "raw_height": 375},
		{"source": "clevr", "raw_width": 480, "raw_height": 320},
	]

	result = summarize_resolutions(records, min_pixels=4096, max_pixels=1_843_200)

	assert result["overall"]["count"] == 3
	assert result["by_source"]["coco"]["count"] == 1
	assert result["by_source"]["gqa_balanced"]["count"] == 1
	assert result["by_source"]["clevr"]["count"] == 1
	assert result["overall"]["raw_pixels"]["minimum"] == 153_600
	assert result["overall"]["raw_pixels"]["maximum"] == 307_200
	assert result["overall"]["visual_tokens"]["minimum"] == 150
	assert result["overall"]["visual_tokens"]["maximum"] == 300
	assert result["overall"]["unique_raw_resolutions"] == 3
	assert result["overall"]["unique_processed_resolutions"] == 3
