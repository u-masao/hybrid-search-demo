# Makefile for code formatting and linting

.PHONY: format lint repro visualize

format:
	poetry run isort src tests
	poetry run black src tests -l 79

lint:
	poetry run flake8 src tests

repro:
	dvc repro

visualize:
	poetry run streamlit run src/visualize.py

test:	
	poetry run python -m unittest discover tests

