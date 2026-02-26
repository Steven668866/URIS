PYTHON ?= python3

.PHONY: run run-legacy test doctor benchmark

run:
	streamlit run app.py

run-legacy:
	streamlit run legacy/legacy_video_reasoning_app.py

test:
	pytest tests/test_platform_config.py tests/test_scenario_engine.py tests/test_state_init.py -q

doctor:
	$(PYTHON) scripts/project_doctor.py

benchmark:
	$(PYTHON) scripts/benchmark_interaction.py --iterations 100
