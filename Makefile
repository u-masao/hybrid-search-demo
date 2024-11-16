# Makefile for code formatting and linting

.PHONY: format lint

format:
	poetry run isort src tests
	poetry run black src tests -l 79

lint:
	poetry run flake8 src tests

test:
	poetry run python -m unittest discover tests

