# Makefile for code formatting and linting

.PHONY: format lint repro visualize check_commit run

run:
	PYTHONPATH=. poetry run python -m src.app

format:
	poetry run isort src tests
	poetry run black src tests -l 79

lint:
	poetry run flake8 src tests

check_commit:
	git diff-index --quiet HEAD --

repro: check_commit PIPELINE.md
	PYTHONPATH=. poetry run dvc repro
	git commit dvc.lock -m 'run dvc repro' || true

PIPELINE.md: dvc.yaml params.yaml
	poetry run dvc dag --md > PIPELINE.md
	git commit PIPELINE.md -m 'dvc pipeline updated' || true

visualize:
	poetry run streamlit run src/visualize.py

test:
	PYTHONPATH=$(shell pwd) poetry run pytest tests/test_api_server.py
