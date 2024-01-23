import numpy as np
import torch
from typing import List
from models.train import train
from data.toy_examples import create_geodesic_pairs_circle
from visualization.visualize import visualize_circle_metric, visualize_training_data, visualize_loss





def circle_metric_with_n(sample_sizes: List[int], noise: float, penalty: float):
    torch.manual_seed(12)
    theta = np.linspace(0, np.pi * 2, 1000)
    basis_on_manifold = torch.from_numpy(np.vstack([np.cos(theta), np.sin(theta)]).T).to(torch.float32)
    
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)

    model_loss = []
    frechet_loss = []
    for num_samps in sample_sizes:
        torch.set_default_dtype(torch.float32)
        trajectories, start_points, start_velo = create_geodesic_pairs_circle(num_samps, 5, noise = noise)
        basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
        initial_conditions = (start_points, start_velo)
        visualize_training_data(basis, start_velo)
        model = train(trajectories, initial_conditions, epochs = 300, regularizer=penalty, n = n_dims,
                       t = 5, m = m, c = c, 
                  basis = basis, active_dims = active_dims)
        with torch.no_grad():
            generated_trajectories, _ = model.forward(initial_conditions)
            predicted_trajectories = torch.permute(generated_trajectories, (1,0,2))
            visualize_circle_metric(model, basis_on_manifold)
            visualize_training_data(torch.reshape(predicted_trajectories, (-1, n_dims)), train = False)

            frechet_loss.append(model.metric_space.metric.dist(torch.reshape(predicted_trajectories, (-1, n_dims)), basis).sum()/num_samps)
            model_loss.append(torch.square(predicted_trajectories - trajectories).sum()/num_samps)
    
    visualize_loss(model_loss, frechet_loss)
    raise ValueError








if __name__ == "__main__":
    circle_metric_with_n(sample_sizes = [5, 50, 500, 1000], noise = .05, penalty = 0 )





    
        