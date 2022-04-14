###############################
# GLOBALS
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV); export PYTHONPATH=`pwd`:$${PYTHONPATH}
SHELL = bash

.PHONY: $(CONDA_ENV) clean setup test jupyter

###############################
# COMMANDS
setup: $(CONDA_ENV)
	$(CONDA_ACTIVATE) && R -e "IRkernel::installspec()"
	pre-commit install
	pip install --upgrade --force-reinstall "jax[cpu]"
	pip install -U kaleido

docker_build: Dockerfile
	docker build -t dclasso .

screening:
	$(CONDA_ACTIVATE); nextflow src/screening.nf -params-file results/screening/nf_config.yaml -resume

fdr_control:
	$(CONDA_ACTIVATE); nextflow src/fdr_control.nf -params-file results/fdr_control/nf_config.yaml -resume

benchmark: results/benchmark/config.yaml
	$(CONDA_ACTIVATE); nextflow src/benchmark.nf -params-file results/benchmark/config.yaml -resume

test:
	$(CONDA_ACTIVATE); pytest test

$(CONDA_ENV): environment.yml
	mamba env create --force --prefix $(CONDA_ENV) --file environment.yml

jupyter:
	$(CONDA_ACTIVATE); export PYTHONPATH=`pwd`:$${PYTHONPATH}; jupyter lab --notebook-dir=notebooks/

clean:
	rm -rf env/
