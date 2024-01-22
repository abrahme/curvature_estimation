import numpy as np
import torch
from typing import List
from models.train import train
from models.model_utils import compute_frechet_distance
from data.toy_examples import create_geodesic_pairs_circle
from visualization.visualize import visualize_circle_metric, visualize_training_data, visualize_loss





def circle_metric_with_n(sample_sizes: List[int], noise: float, penalty: float):
    torch.set_default_dtype(torch.float32)
    theta = np.linspace(0, np.pi * 2, 1000)
    basis_on_manifold = np.vstack([np.cos(theta), np.sin(theta)]).T
    
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)

    model_loss = []
    frechet_loss = []
    for num_samps in sample_sizes:
        trajectories, start_points, start_velo = create_geodesic_pairs_circle(num_samps, 5, noise = noise)
        basis = trajectories.reshape((-1, n_dims)) ### only construct basis from whatever points we have 
        initial_conditions = (start_points, start_velo)
        visualize_training_data(basis, start_velo)
        model = train(trajectories, initial_conditions, epochs = 50, regularizer=penalty, n = n_dims,
                       t = 5, m = m, c = c, 
                  basis = basis, active_dims = active_dims)
        with torch.no_grad():
            generated_trajectories, _ = model.forward(initial_conditions)
            predicted_trajectories = torch.permute(generated_trajectories, (1,0,2))
            visualize_circle_metric(model, basis_on_manifold)
            visualize_training_data(predicted_trajectories, train = False)

            frechet_loss.append(compute_frechet_distance(predicted_trajectories, trajectories, model))
            model_loss.append(torch.square(predicted_trajectories - trajectories).sum())
    
    visualize_loss(model_loss, frechet_loss)








if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    basis_x, basis_y = np.meshgrid(np.arange(-1,1, .01), np.arange(-1,1, .01))
    basis = torch.from_numpy(np.stack((basis_x.ravel(), basis_y.ravel()), axis = 1)).to(torch.float32)
    theta = np.linspace(0, np.pi * 2, 1000)
    # basis_on_manifold = np.vstack([np.cos(theta), np.sin(theta)]).T)
    trajectories, start_points, start_velo = create_geodesic_pairs_circle(40, 20, noise =.1)
    initial_conditions = (start_points, start_velo)
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)

    model = train(trajectories, initial_conditions, epochs = 10, regularizer=2.0, n = 2, t = 20, m = m, c = c, 
                  basis = basis, active_dims = active_dims)





    
        