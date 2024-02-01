# Geodesic Based Invariant Metric Learning 

This repository contains the implementation of our paper. 

### Requirements
Requirements are noted in the `pyproject.toml` and `poetry.lock` file. 

### Running Experiments

We have several models and subsequent experiments one can run 

#### Models
1. Explicit Equivariant Metric Learning (only rotational symmetries implemented for now)
2. Implicit Equivariant Metric Learning (only rotational symmetries implemented for now)
3. Metric Learning (no prior)
4. Neural ODE for model comparison 

#### Experiments 
Currently, in `main.py` we provide entrypoints to running our experiments. We currently have implementations for the circle and sphere. 

Example:

``` python main.py --sample_sizes=5,10,15 --timesteps=10 --noise=5,10,30  --manifold=circle```

The above code runs an experiment over the list of sample sizes and noise levels on the circle, with no prior information. Resulting plots and figures will be output to `data/`. Please view the  `main.py` file for more entrypoints. 
