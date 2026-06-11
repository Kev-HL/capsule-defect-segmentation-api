# Makefile for Capsule Defect Detection and Segmentation Project

# Install dependencies
setup:
	pip install -r requirements.txt

# Lint code (format with black and check with flake8))
lint:
	@echo "Linting source code"
	python -m black src/ scripts/
	# .flake8 file is configured with max-line-length = 88 
	# and ignore = E203, W503 for compatibility with black formatting
	python -m flake8 src/ scripts/

# Clean temporary files
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	@echo "Clean complete!"

# Help: show available commands
help:
	@echo "Available make targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d':' -f1 | grep -v '^_' | sort

.PHONY: setup lint clean help