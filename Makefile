# Makefile for code formatting and linting

.PHONY: format lint

format:
	poetry run isort src
	poetry run black src

lint:
	poetry run flake8 src
