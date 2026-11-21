install:
	pip install -e ".[dev]"
test:
	pytest
reproduce:
	python examples/reproduce_headline.py
