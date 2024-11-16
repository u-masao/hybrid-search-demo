# Makefile for code formatting and linting

.PHONY: format lint repro visualize check_commit

format:
	poetry run isort src tests
	poetry run black src tests -l 79

lint:
	poetry run flake8 src tests
	git commit dvc.lock -m 'run dvc repro'

check_commit:
	git diff-index --quiet HEAD --

repro: check_commit
	poetry run dvc repro

visualize:
	poetry run streamlit run src/visualize.py

test:	
	poetry run python -m unittest discover tests

