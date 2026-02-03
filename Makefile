.PHONY: help install dev test test-cov lint format type-check security clean build publish docs serve-docs changelog update-changelog release-patch release-minor release-major version-patch version-minor version-major show-version preview-release-notes

.DEFAULT_GOAL := help

help:
	@echo "Development Commands"
	@echo "===================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package"
	@echo "  make dev          Install with dev dependencies + pre-commit"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linters"
	@echo "  make format       Auto-format code"
	@echo "  make type-check   Run type checker"
	@echo "  make security     Run security checks"
	@echo "  make commit       Interactive conventional commit"
	@echo ""
	@echo "Build & Release:"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build package"
	@echo "  make publish      Publish to PyPI"
	@echo "  make changelog    Update CHANGELOG.md"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Build documentation"
	@echo "  make serve-docs   Serve docs locally"
	@echo ""
	@echo "Release:"
	@echo "  make release-patch  Full release: test, bump, changelog, commit, tag, push, GitHub release"
	@echo "  make release-minor  Full release: test, bump, changelog, commit, tag, push, GitHub release"
	@echo "  make release-major  Full release: test, bump, changelog, commit, tag, push, GitHub release"
	@echo "  make show-version   Show current version"
	@echo "  make preview-release-notes  Preview release notes for current version"

install:
	uv sync

dev:
	uv sync --group dev --group docs
	uv run pre-commit install --hook-type commit-msg --hook-type pre-commit

test:
	uv run pytest

test-cov:
	uv run pytest --cov --cov-report=html --cov-report=xml

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

type-check:
	uv run ty check src/

security:
	uv run bandit -r src/
	uv run pip-audit

commit:
	uv run cz commit

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/ site/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build: clean
	uv build

publish: build
	uv run twine check dist/*
	uv run twine upload dist/*

publish-test: build
	uv run twine check dist/*
	uv run twine upload --repository testpypi dist/*

docs:
	uv sync --group docs
	uv run mkdocs build

serve-docs:
	uv sync --group docs
	uv run mkdocs serve

changelog:
	uv run git-cliff -o CHANGELOG.md

update-changelog: changelog

# Version bump only (no commit/tag/push)
version-patch:
	uv run bump-my-version bump patch

version-minor:
	uv run bump-my-version bump minor

version-major:
	uv run bump-my-version bump major

show-version:
	@uv run bump-my-version show current_version

preview-release-notes:
	@VERSION=$$(uv run bump-my-version show current_version); \
	echo "Release notes for v$$VERSION:"; \
	echo "---"; \
	awk "/^## \[$$VERSION\]/{flag=1; next} /^## \[/{if(flag) exit} flag" CHANGELOG.md

# Fully automated release targets
release-patch:
	@$(MAKE) _do-release BUMP=patch

release-minor:
	@$(MAKE) _do-release BUMP=minor

release-major:
	@$(MAKE) _do-release BUMP=major

# Internal release target - do not call directly
_do-release: clean test-cov security lint
	@echo "Starting release process ($(BUMP))..."
	uv run bump-my-version bump $(BUMP)
	@echo "Updating changelog..."
	$(MAKE) update-changelog
	@echo "Extracting release notes..."
	$(MAKE) _extract-release-notes
	@echo "Committing changes..."
	git add -A
	git commit -m "chore(release): bump version to $$(uv run bump-my-version show current_version)"
	@echo "Creating tag..."
	git tag "v$$(uv run bump-my-version show current_version)"
	@echo "Pushing to origin..."
	git push origin main --tags
	@echo "Creating GitHub release..."
	gh release create "v$$(uv run bump-my-version show current_version)" \
		--title "Release $$(uv run bump-my-version show current_version)" \
		--notes-file release_notes.md \
		--latest
	@rm -f release_notes.md
	@echo ""
	@echo "Release v$$(uv run bump-my-version show current_version) complete!"
	@echo "PyPI publication will happen automatically via GitHub Actions."

# Extract release notes for current version from CHANGELOG.md
_extract-release-notes:
	@VERSION=$$(uv run bump-my-version show current_version); \
	awk "/^## \[$$VERSION\]/{flag=1; next} /^## \[/{if(flag) exit} flag" CHANGELOG.md > release_notes.md; \
	if [ ! -s release_notes.md ]; then \
		echo "Release $$VERSION" > release_notes.md; \
	fi

ci-check: lint type-check security test-cov
	@echo "All CI checks passed!"
