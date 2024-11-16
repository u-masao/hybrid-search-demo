# Makefile for code formatting and linting

.PHONY: format lint repro

format:
	poetry run isort src tests
	poetry run black src tests -l 79

lint:
	poetry run flake8 src tests

repro:
	dvc repro

test:
	poetry run python -m unittest discover tests

