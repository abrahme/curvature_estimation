import numpy as np
import torch
from models.train import train
from data.toy_examples import create_geodesic_pairs_circle

if __name__ == "__main__":
    basis_x, basis_y = np.meshgrid(np.arange(-1,1, .01), np.arange(-1,1, .01))
    basis = torch.from_numpy(np.stack((basis_x.ravel(), basis_y.ravel()), axis = 1)).to(torch.float64)
    theta = np.linspace(0, np.pi * 2, 1000)
    # basis_on_manifold = np.vstack([np.cos(theta), np.sin(theta)]).T)
    trajectories, start_points, start_velo = create_geodesic_pairs_circle(40, 20, noise =.1)
    trajectories = trajectories.to(torch.float64)
    initial_conditions = (start_points.to(torch.float64), start_velo.to(torch.float64))
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)

    model = train(trajectories, initial_conditions, epochs = 10, regularizer=2.0, n = 2, t = 20, m = m, c = c, 
                  basis = basis, active_dims = active_dims)





    
        