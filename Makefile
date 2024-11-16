# Makefile for code formatting and linting

.PHONY: format lint

format:
	isort src
	black src

lint:
	flake8 src
