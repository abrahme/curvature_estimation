import numpy as np
import argparse
import torch
from typing import List
from models.train import train
from geomstats.geometry.hypersphere import Hypersphere
from data.toy_examples import create_geodesic_pairs_circle
from visualization.visualize import visualize_circle_metric, visualize_training_data, visualize_loss, visualize_convergence




def plot_convergence(preds: List[np.ndarray], actual: np.ndarray, skip_every: int, n:int, penalty: float):
    num_epochs = len(preds)
    indices = range(0, num_epochs, skip_every)
    for index in indices:
        visualize_convergence(torch.reshape(preds[index],(-1, 2)), actual,epoch_num=index,n=n, penalty=penalty )
    


def circle_metric_with_n(sample_sizes: List[int], noise: float, penalty: float, timesteps:int, keep_preds:bool = False):
    torch.manual_seed(12)
    theta = np.linspace(0, np.pi * 2, 1000)
    basis_on_manifold = torch.from_numpy(np.vstack([np.cos(theta), np.sin(theta)]).T).to(torch.float32)
    xi, yi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    manifold_basis = torch.stack([xi.flatten(), yi.flatten()], axis = -1)
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)

    model_loss = []
    frechet_loss = []
    latent_space = Hypersphere(dim = 1, equip=True)
    for num_samps in sample_sizes:
        torch.set_default_dtype(torch.float32)
        trajectories, start_points, start_velo = create_geodesic_pairs_circle(num_samps, timesteps, noise = noise)
        sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
        initial_conditions = (start_points, start_velo)
        visualize_training_data(sample_basis, num_samps, start_velo)
        model, preds = train(trajectories, initial_conditions, epochs = 300, regularizer=penalty, n = n_dims,
                       t = timesteps, m = m, c = c, 
                  basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds)
        
        with torch.no_grad():
            generated_trajectories, _ = model.forward(initial_conditions)
            predicted_trajectories = torch.permute(generated_trajectories, (1,0,2))
            visualize_circle_metric(model, basis_on_manifold, num_samps, penalty)
            visualize_training_data(torch.reshape(predicted_trajectories, (-1, n_dims)), num_samps, penalty = penalty, train = False)
            geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(predicted_trajectories, (-1, n_dims))), sample_basis).sum()/num_samps
            frechet_loss.append(geodesic_distance)
            model_loss.append(torch.square(predicted_trajectories - trajectories).sum()/num_samps)
        if keep_preds:
            plot_convergence(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty )
    visualize_loss(model_loss, frechet_loss, sample_sizes)
    








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for fitting manifold estimation model")
    # Add command-line arguments
    parser.add_argument('--noise',type=float, help='noise to jitter generated geodesics')
    parser.add_argument('--penalty',type=float, help='how much to penalize prior')
    parser.add_argument('--keep_preds', action="store_true", help='whether or not to plot convergence', default=False)
    parser.add_argument('--timesteps', type=int,  help='length of trajectories')
    parser.add_argument('--sample_sizes',type=lambda x: [int(item) for item in x.split(",")], help='comma separated list of ints')
    args = parser.parse_args()

    circle_metric_with_n(sample_sizes = args.sample_sizes, noise = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps)





    
        