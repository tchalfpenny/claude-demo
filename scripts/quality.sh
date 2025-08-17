#!/bin/bash

# Complete quality check script
# Runs formatting, linting, and type checking

echo "🚀 Running complete quality checks..."

# Format code
echo "Step 1: Formatting..."
./scripts/format.sh
echo ""

# Lint code
echo "Step 2: Linting..."
./scripts/lint.sh
echo ""

# Type check
echo "Step 3: Type checking..."
./scripts/typecheck.sh
echo ""

# Run tests
echo "Step 4: Running tests..."
cd backend && uv run pytest -v
cd ..

echo "🎉 All quality checks complete!"