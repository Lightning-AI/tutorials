.PHONY: ipynb clean docs

# META := $(wildcard **/.meta.yml)
META := $(shell find -regex ".*/.meta.y[a]?ml")
IPYNB := $(META:%/.meta.yml=%.ipynb)
IPYNB := $(IPYNB:%/.meta.yaml=%.ipynb)
export PATH_DATASETS=$(PWD)/.datasets

init:
	@echo $(PATH_DATASETS)
	mkdir -p $(PATH_DATASETS)

ipynb: init ${IPYNB}
# 	@echo $<

%.ipynb: %/.meta.y*ml
	@echo $<
	python .actions/assistant.py augment-script $(shell dirname $<)
	python .actions/assistant.py bash-render $(shell dirname $<) > .actions/_ipynb-render.sh
	bash .actions/_ipynb-render.sh

docs: clean
	pip install --quiet -r docs/requirements.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build

clean:
	rm -rf ./.datasets
	# clean all temp runs
	rm -rf ./docs/build
	rm -rf ./docs/source/notebooks
	rm -rf ./docs/source/api
