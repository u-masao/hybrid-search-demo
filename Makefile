# Makefile for code formatting and linting

.PHONY: docker_start run_embedding_api run_search_app repro check_commit format lint check_commit test visualize

all:

### backend section ###
## run servers
run_servers: docker_start run_embedding_api run_translation_api run_search_app

## elastic, kibana, 
docker_start:
	docker compose start

## embedding api
run_embedding_api:
	poetry run python -m src.embedding_api.api --port 5001 --host 0.0.0.0

## translation api
run_translation_api:
	poetry run python -m src.translation_api.api --port 5002 --host 0.0.0.0

## search app
run_search_app:
	FLASK_APP=src.search_app.main poetry run flask run --debugger --reload


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
	PYTHONPATH=. poetry run pytest -s tests


### analyse section ###
visualize:
	poetry run streamlit run src/data_visualization/visualize.py
