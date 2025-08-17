#!/bin/bash

# Code linting script
# Runs flake8 to check code style and quality

echo "🔍 Linting Python code..."

echo "Running flake8..."
uv run flake8 backend/ --max-line-length=88 --extend-ignore=E203,W503

echo "✅ Linting complete!"