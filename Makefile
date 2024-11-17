# Makefile for code formatting and linting

.PHONY: format lint repro visualize check_commit

format:
	poetry run isort src tests
	poetry run black src tests -l 79

lint:
	poetry run flake8 src tests

check_commit:
	git diff-index --quiet HEAD --

repro: check_commit PIPELINE.md
	poetry run dvc repro
	git commit dvc.lock -m 'run dvc repro' || true

PIPELINE.md: dvc.yaml params.yaml
	poetry run dvc dag --md > PIPELINE.md
	git commit PIPELINE.md -m 'dvc pipeline updated' || true

visualize:
	poetry run streamlit run src/visualize.py

test:	
	poetry run python -m unittest discover tests

run_ui:
	PYTHONPATH=. poetry run gradio src/search_ui.py

