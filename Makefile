
.phony: init test

init:
	pip install -r requirements.txt

test:
	python -m pytest tests/
