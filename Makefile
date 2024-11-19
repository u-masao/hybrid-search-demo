# Makefile for code formatting and linting

.PHONY: format lint repro visualize check_commit run run_api_server

### backend section ###
## elastic, kibana, 
docker_start:
	docker compose start

## embedding api
run_embedding_api:
	poetry run python -m src.embedding_api.api

## search app
run_search_app:
	poetry run python -m src.search_app.main


### batch section ###
repro: check_commit PIPELINE.md
	PYTHONPATH=. poetry run dvc repro
	git commit dvc.lock -m 'run dvc repro' || true


### develop section ###
format:
	poetry run isort src tests
	poetry run black src tests -l 79

lint:
	poetry run flake8 src tests

check_commit:
	git diff-index --quiet HEAD --

PIPELINE.md: dvc.yaml params.yaml
	poetry run dvc dag --md > PIPELINE.md
	git commit PIPELINE.md -m 'dvc pipeline updated' || true

test:
	PYTHONPATH=. poetry run pytest tests


### analyse section ###
visualize:
	poetry run streamlit run src/visualize.py

