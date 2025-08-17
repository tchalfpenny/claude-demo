#!/bin/bash

# Type checking script
# Runs mypy to check type annotations

echo "🔧 Type checking Python code..."

echo "Running mypy..."
uv run mypy backend/ --ignore-missing-imports

echo "✅ Type checking complete!"