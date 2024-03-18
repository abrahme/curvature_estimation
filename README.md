# Geodesic Based Invariant Metric Learning 

This repository contains the implementation of our paper. 

### Requirements
Requirements are noted in the `environment.yml` file and require `conda`. In order to create the environment, first clone the `neural-riemannian-metric` branch of the repository. Then inside the `curvature-estimation` directory, run `conda env create -f environment.yml`. Then, in order to activate the environment, run `conda activate curvature-estimation`. 

#### Experiments 
Currently, in `main.py` we provide entrypoints to running our experiments. We currently have implementations for the circle and sphere. 

Example:

``` python main.py --sample_sizes=5,10,15 --timesteps=10 --noise=5,10,30  --manifold=circle```

The above code runs an experiment over the list of sample sizes and noise levels on the circle, with no prior information. Resulting plots and figures will be output to `data/`. Please view the  `main.py` file for more entrypoints. 
