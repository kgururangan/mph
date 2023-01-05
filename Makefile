.DEFAULT_GOAL := install
src := mph
isort = isort $(src)
black = black $(src)
autoflake = autoflake -ir --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables $(src)

.PHONY: install
install:
	$(CONDA_PREFIX)/bin/pip install -e .
