###############################
# GLOBALS
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV); export PYTHONPATH=`pwd`:$${PYTHONPATH}; [[ -z "${DEBUG}" ]] && export FILENAME='nf_config.yaml' || export FILENAME="test.yaml"
PROFILE = -c ./nextflow.config -profile kuma
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

# screening:
# 	$(CONDA_ACTIVATE); nextflow src/screening.nf $(PROFILE) -params-file results/screening/$${FILENAME} -resume

# lambda_control:
# 	$(CONDA_ACTIVATE); nextflow src/lambda_control.nf $(PROFILE) -params-file results/lambda_control/$${FILENAME} -resume

benchmark:
	$(CONDA_ACTIVATE); nextflow src/benchmark.nf $(PROFILE) -params-file results/benchmark/$${FILENAME} -resume

less_benchmark:
	$(CONDA_ACTIVATE); nextflow src/LessJobs_benchmark.nf $(PROFILE) -params-file results/benchmark/$${FILENAME} -resume

real_data:
	$(CONDA_ACTIVATE); nextflow src/real_data_benchmark.nf $(PROFILE) -params-file results/real_data/$${FILENAME} -resume

test:
	$(CONDA_ACTIVATE); pytest test

$(CONDA_ENV): environment.yml
	mamba env create --prefix $(CONDA_ENV) --file environment.yml

jupyter:
	$(CONDA_ACTIVATE); export PYTHONPATH=`pwd`:$${PYTHONPATH}; jupyter lab --notebook-dir=notebooks/


clean:
	rm -rf env/
