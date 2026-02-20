.PHONY: install setup start stop test lint clean

install:                         ## Check dependencies, create venv, install
	@bash install.sh

setup: install                   ## Run the guided setup wizard
	@. .venv/bin/activate && openlegion setup

start:                           ## Start the runtime
	@. .venv/bin/activate && openlegion start

stop:                            ## Stop all agents
	@. .venv/bin/activate && openlegion stop

test:                            ## Run unit + integration tests
	@. .venv/bin/activate && pytest tests/ \
		--ignore=tests/test_e2e.py \
		--ignore=tests/test_e2e_chat.py \
		--ignore=tests/test_e2e_memory.py \
		--ignore=tests/test_e2e_triggering.py -x

lint:                            ## Lint and format
	@. .venv/bin/activate && ruff check src/ tests/ && ruff format src/ tests/

clean:                           ## Remove venv and caches
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache *.egg-info

help:                            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
