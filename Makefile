.PHONY: test clean docs

# todo: add rebuild all examples

docs: clean
	pip install --quiet -r docs/requirements.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build

clean:
	# clean all temp runs
	rm -rf ./docs/build
	rm -rf ./docs/source/notebooks
	rm -rf ./docs/source/api
