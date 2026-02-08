# Makefile for SLM Fine-tuning Project

.PHONY: help install setup test clean train inference format lint docker-build docker-run

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

setup: ## Run full setup (install + create directories + sample data)
	bash setup.sh

prepare-data: ## Prepare training data from ATP 6-0.5
	python src/prepare_data.py --input data/raw/atp_6_0_5.txt --output data/processed

train: ## Start training with default config
	python src/train.py --config configs/llama_3_2_3b.yaml

inference: ## Run interactive inference (requires MODEL variable)
	python src/inference.py --model $(MODEL) --interactive

test: ## Run tests
	python -m pytest tests/ -v

format: ## Format code with black
	black src/ tests/

lint: ## Lint code with flake8
	flake8 src/ tests/ --max-line-length=100

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .mypy_cache

docker-build: ## Build Docker image
	docker build -t slm-finetuning .

docker-run: ## Run Docker container with GPU
	docker run --gpus all -it -v $(PWD)/models:/workspace/models slm-finetuning

tensorboard: ## Start TensorBoard
	tensorboard --logdir logs --port 6006

.DEFAULT_GOAL := help
