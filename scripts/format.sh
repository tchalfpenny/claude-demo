#!/bin/bash

# Code formatting script
# Runs black and isort to format Python code

echo "🎨 Formatting Python code..."

echo "Running black..."
uv run black backend/ --line-length 88

echo "Running isort..."
uv run isort backend/ --profile black

echo "✅ Code formatting complete!"