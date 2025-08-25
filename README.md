# SmRMR
Code to reproduce the results of the paper *Sparse minimum Redundancy Maximum Relevance for feature selection* by Peter Naylor, Benjamin Poignard, Hector Climente-Gonzales and Makoto Yamada. 

```bibtex
@inproceedings{naylor2025sparse,
  title={Sparse minimum Redundancy Maximum Relevance for feature selection},
  author={Naylor, Peter and Poignard, Benjamin and Climente-Gonz{\'a}lez, H{\'e}ctor and Yamada, Makoto},
  journal={arXiv preprint arXiv:XXXX},
  url={https://arxiv.org/abs/XXXX},
  year={2025}
}
```


If you use our code, please cite the above paper.

## Pre-requisites

This project has three pre-requisites:
- [Nextflow](https://www.nextflow.io/)
- [Anaconda](https://www.anaconda.com/products/individual)
- [mamba](https://github.com/mamba-org/mamba)
- [pre-commit](https://pre-commit.com/)

When the three of them are available, run the following commands:
```
# install the virtual environment in ./env/
make setup
# (optional) if you want to contribute, install pre-commit
pre-commit install
```

## Useful Environnement variables
```bash
DEPLOY = 1 # this variable can be set to run or the local or the long experiment
```

## Reproducible results

### Set up `nextflow.config` file
Nextflow's power is that it allows one to run a scientific pipeline with one single script, this script can optimize your available resources.
If you are on your local computer, there isn't much to change expect to define the number of allowed process in parallel such as `executor.queueSize`.
If you have access to a cluster, you can define a scheduler, such as SGE or Slurm.
Here is an example of such a script for our SGE cluster with Singularity:
```
profiles {
    knockoff {
        process.executor = 'sge' \\scheduler
        process.queue = 'c1normal' \\ cluster queue
        process.memory = '10GB'
        process.container = 'file://./env/container_img.sif' 
        process.containerOptions = '-B /data:/data' \\for mounting our external HD
        executor {
            queueSize = 500
            submitRateLimit = '10 sec'
        }
        singularity {
            enabled = true
            envWhitelist = 'PYTHONPATH'
        }
    }
}
```

### Possible environnement 
To offer a couple of possibilities we have set up a Singularity and conda environnement. 
Nextflow can use either, it can use Singularity containers or it can create a conda environnement on the fly.

``` bash
sudo make docker_build
```
#### Conda
To use the Conda environnement on the fly, you will have to remove the Singularity option and add the Conda ones; for example:

```
profiles {
    knockoff {
        process.executor = 'sge' \\scheduler
        process.queue = 'c1normal' \\ cluster queue
        process.memory = '10GB'
        process.conda = '/path/to/main/folder/.../env/knockoff.yaml'
        executor {
            queueSize = 500
            submitRateLimit = '10 sec'
        }
    }
}
```

## Final commands
Finally, to launch the pipeline on the simulated dataset:
```
make benchmark
```
and to launch the pipeline on real data:
```
make real_data
```
And you should find the results in the appropriate result folder: `results/benchmark` for example.
## Adapt to your project

The configuration file, an example of which can be found `results/benchmark/nf_config.yaml` shows most of the parameter that can be tweaked or added evaluating different models.

We made an python object that runs the proposed model in a `scikit-learn` fashion
```python
dl = smrmr(
    alpha=0.3,
    measure_stat="HSIC",
    kernel="gaussian",
)
train_l, val_l = dl.fit(
    X_train,
    y_train,
    X_val,
    y_val,
    "scad",
    0.1, #lambda
    n1=0.5,
    pen_kwargs=dict(a=3.7, b=3.5),
)
```