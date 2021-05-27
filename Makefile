.PHONY: ipynb clean docs

# META := $(wildcard **/.meta.yml)
META := $(shell find -name .meta.yml)
IPYNB := $(META:%/.meta.yml=%.ipynb)

ipynb: ${IPYNB}
# 	@echo $<

%.ipynb: %/.meta.yml
	@echo $<
	bash .actions/ipynb-generate.sh $(shell dirname $<)
	bash .actions/ipynb-render.sh $(shell dirname $<)

docs: clean
	pip install --quiet -r docs/requirements.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build

clean:
	# clean all temp runs
	rm -rf ./docs/build
	rm -rf ./docs/source/notebooks
	rm -rf ./docs/source/api
