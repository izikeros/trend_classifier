.PHONY: clean lint test doc

# main directory with source code
PROJECT_NAME = trend_classifier

# use open for macOS, xdg-open for Linux
UNAME := $(shell uname -s)

ifeq ($(UNAME), Linux)
OPEN := xdg-open
endif
ifeq ($(UNAME), Darwin)
OPEN := open
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint package using pre-commit
lint:
	pre-commit run --all-files

## Run tests using pytest
test:
	pytest tests/

## Create coverage report for package
coverage:
	pytest --cov-report html --cov $(PROJECT_NAME) --verbose

## Return coverage percentage for package
coverage_num:
	@pytest --cov $(PROJECT_NAME)
	@coverage xml
	@coverage report | tail -n 1 | awk -F' ' '{print $$6}'

## Show HTML report for trend_classifier package coverage
coverage_show:
	$(OPEN) htmlcov/index.html

## Generate pdoc HTML documentation for trend_classifier package
doc:
	pdoc --force --html --output-dir ./docs $(PROJECT_NAME)
	mv docs/$(PROJECT_NAME)/* docs
	rmdir docs/$(PROJECT_NAME)

## Generate pdoc HTML documentation for trend_classifier package and open in browser
doc_view:
	pdoc --force --html --output-dir ./docs $(PROJECT_NAME)
	mv docs/$(PROJECT_NAME)/* docs
	rmdir docs/$(PROJECT_NAME)
	$(OPEN) ./docs/index.html

## Generate requirements.txt and requirements-test.txt files
reqs:
	poetry export -f requirements.txt --output requirements.txt
	poetry export -f requirements.txt --with test --output requirements-test.txt

## Upgrade package versions for tox
tox_reqs_update:
	pip-upgrade tox-reqs/all.txt

## Upgrade all (requirements used in tox envs, pyproject dependencies, pre-commit hooks)
update: tox_reqs_update
	pre-commit autoupdate
	pre-commit gc
	poetry update

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
