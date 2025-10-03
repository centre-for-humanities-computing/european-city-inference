
install:
	@echo "--- 🚀 Installing project ---"
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv
	bash -c "source .venv/bin/activate"
	uv sync
	uv pip install -e .

jupyterlab:
	uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=european_cities
	uv run --with jupyter jupyter lab --port 4444

pre-commit:
	@echo "--- 🧹 Running pre-commit on all files ---"
	uv run pre-commit install
	uv run pre-commit autoupdate
	uv run pre-commit run --all-files

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						        # running ruff formatting
	uv run ruff check **/*.py --fix						# running ruff linting

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-report=html

run-all-notebooks:
	@echo "--- 📚 Running all notebooks ---"
	cd notebooks/ && uv run python -m nbconvert *.ipynb --to notebook --execute --inplace