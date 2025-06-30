
install:
	@echo "--- 🚀 Installing project ---"
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv
	bash -c "source .venv/bin/activate"
	uv sync

pre-commit:
	@echo "--- 🧹 Running pre-commit on all files ---"
	uv run pre-commit install
	uv run pre-commit autoupdate
	uv run pre-commit run --all-files

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						        # running ruff formatting
	uv run ruff check **/*.py --fix						# running ruff linting