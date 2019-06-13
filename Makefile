.PHONY: default

default: test;

publish:
	rm -rf ./dist/
	python setup.py sdist
	twine upload dist/*

test:
	pytest .

requirements.txt:
	pip-compile requirements.in

