import numpy as np
import argparse
import torch
import pandas as pd
from typing import List
from models.train import train
from geomstats.geometry.hypersphere import Hypersphere
from data.toy_examples import create_geodesic_pairs_circle, create_geodesic_pairs_circle_hemisphere, create_geodesic_pairs_sphere
from visualization.visualize import visualize_circle_metric, visualize_loss, visualize_convergence, visualize_training_data_sphere, visualize_convergence_sphere




def plot_convergence(preds: List[np.ndarray], actual: np.ndarray, skip_every: int, n:int, penalty: float, val: bool = False, noise: int = 0):
    num_epochs = len(preds)
    indices = range(0, num_epochs, skip_every)
    for index in indices:
        visualize_convergence(torch.reshape(preds[index],(-1, 2)), actual,epoch_num=index,n=n, penalty=penalty, val = val, noise=noise)

def plot_convergence_sphere(preds: List[np.ndarray], actual: np.ndarray, skip_every: int, n:int, penalty: float, val: bool = False):
    num_epochs = len(preds)
    indices = range(0, num_epochs, skip_every)
    for index in indices:
        visualize_convergence_sphere(torch.reshape(preds[index],(-1, 2)), actual,epoch_num=index,n=n, penalty=penalty, val = val)

def circle_metric_hemisphere_with_n(sample_sizes: List[int], noise_level: List[float], penalty: float, timesteps:int, keep_preds:bool = False, val:bool = True, loss_type:str = "L2"):
    theta = np.linspace(0, np.pi * 2, 1000)
    basis_on_manifold = torch.from_numpy(np.vstack([np.cos(theta), np.sin(theta)]).T).to(torch.float32)
    xi, yi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    manifold_basis = torch.stack([xi.flatten(), yi.flatten()], axis = -1)
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)

    losses = []

    latent_space = Hypersphere(dim = 1, equip=True)
    for noise in noise_level:
        for num_samps in sample_sizes:
            torch.set_default_dtype(torch.float32)
            trajectories, start_points, start_velo, val_trajectories, val_start_points, val_start_velo = create_geodesic_pairs_circle_hemisphere(num_samps, timesteps, noise = 1/noise)
            sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
            val_sample_basis = torch.reshape(val_trajectories,(-1, n_dims))
            initial_conditions = torch.hstack((start_points, start_velo))
            val_initial_conditions = torch.hstack((val_start_points, val_start_velo))
    
            model, preds = train(trajectories, initial_conditions, epochs = 300, regularizer=penalty, n = n_dims,
                        t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                    basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            
            with torch.no_grad():
                visualize_circle_metric(model, basis_on_manifold, num_samps, penalty, noise = noise)
                

                val_generated_trajectories = model.forward(val_initial_conditions)
                val_predicted_trajectories = torch.permute(val_generated_trajectories, (1,0,2))
                plot_convergence([torch.reshape(val_predicted_trajectories, (-1, n_dims))], val_sample_basis,n = num_samps, penalty = penalty, skip_every = 30, val = val, noise=noise)

                val_geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(val_predicted_trajectories, (-1, n_dims))), val_sample_basis).mean()
                losses.append({"loss_val":val_geodesic_distance, "n": num_samps, "noise": 1/noise, "loss_type": "geodesic"})
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories).mean(),
                              "n": num_samps, "noise":1/noise, "loss_type": "model"})
            if keep_preds:
                plot_convergence(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty, noise=noise )
    pd.DataFrame(losses).to_csv("data/hemisphere_circle_losses.csv", index=False)


def circle_metric_with_n(sample_sizes: List[int], noise_level: float, penalty: float, timesteps:int, keep_preds:bool = False, val:bool = True, loss_type:str = "L2"):
    theta = np.linspace(0, np.pi * 2, 1000)
    basis_on_manifold = torch.from_numpy(np.vstack([np.cos(theta), np.sin(theta)]).T).to(torch.float32)
    xi, yi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    manifold_basis = torch.stack([xi.flatten(), yi.flatten()], axis = -1)
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)

    losses = []
    latent_space = Hypersphere(dim = 1, equip=True)
    for noise in noise_level:
        for num_samps in sample_sizes:
            torch.set_default_dtype(torch.float32)
            trajectories, start_points, start_velo, val_trajectories, val_start_points, val_start_velo = create_geodesic_pairs_circle(num_samps, timesteps, noise = 1/noise)
            sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
            val_sample_basis = torch.reshape(val_trajectories,(-1, n_dims))
            initial_conditions = torch.hstack((start_points, start_velo))
            val_initial_conditions = torch.hstack((val_start_points, val_start_velo))
            model, preds = train(trajectories, initial_conditions, epochs = 300, regularizer=penalty, n = n_dims,
                        t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                    basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            
            with torch.no_grad():
                visualize_circle_metric(model, basis_on_manifold, num_samps, penalty, noise=noise)
                

                val_generated_trajectories = model.forward(val_initial_conditions)
                val_predicted_trajectories = torch.permute(val_generated_trajectories, (1,0,2))
                plot_convergence([torch.reshape(val_predicted_trajectories, (-1, n_dims))], val_sample_basis,n = num_samps, penalty = penalty, skip_every = 10, val = val, noise=noise)
                val_geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(val_predicted_trajectories, (-1, n_dims))), val_sample_basis).mean()
                losses.append({"loss_val":val_geodesic_distance, "n": num_samps, "noise": 1/noise, "loss_type": "geodesic"})
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories).mean(),
                              "n": num_samps, "noise": 1/noise, "loss_type": "model"})     
            if keep_preds:
                plot_convergence(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty, noise=noise )
    pd.DataFrame(losses).to_csv("data/circle_losses.csv", index=False)



def sphere_metric_with_n(sample_sizes: List[int], noise: float, penalty: float, timesteps:int, keep_preds:bool = False, val:bool = True, loss_type:str = "L2"):
    xi, yi, zi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    manifold_basis = torch.stack([xi.flatten(), yi.flatten(), zi.flatten()], axis = -1)
    m = [5,5,5]
    c = 4.0
    active_dims = [0,1,2]
    n_dims = len(active_dims)
    model_loss = []
    frechet_loss = []
    latent_space = Hypersphere(dim = 2, equip=True)
    for num_samps in sample_sizes:
        torch.set_default_dtype(torch.float32)
        trajectories, start_points, start_velo, val_trajectories, val_start_points, val_start_velo = create_geodesic_pairs_sphere(num_samps, timesteps, noise = noise)
        sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
        val_sample_basis = torch.reshape(val_trajectories,(-1, n_dims))
        initial_conditions = torch.hstack((start_points, start_velo))
        val_initial_conditions = torch.hstack((val_start_points, val_start_velo))
        # visualize_training_data_sphere(sample_basis, num_samps)
        model, preds = train(trajectories, initial_conditions, epochs = 300, regularizer=penalty, n = n_dims,
                       t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                  basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
        
        with torch.no_grad():
            generated_trajectories = model.forward(initial_conditions)
            predicted_trajectories = torch.permute(generated_trajectories, (1,0,2))
            visualize_training_data_sphere(torch.reshape(predicted_trajectories, (-1, n_dims)), num_samps, penalty = penalty, train = False)
            

            val_generated_trajectories = model.forward(val_initial_conditions)
            val_predicted_trajectories = torch.permute(val_generated_trajectories, (1,0,2))
            plot_convergence_sphere([torch.reshape(val_predicted_trajectories, (-1, n_dims))], val_sample_basis,n = num_samps, penalty = penalty, skip_every = 10, val = val)

            geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(predicted_trajectories, (-1, n_dims))), sample_basis).sum()/num_samps
            frechet_loss.append(geodesic_distance)
            model_loss.append(torch.square(predicted_trajectories - trajectories).sum()/num_samps)
        if keep_preds:
            plot_convergence_sphere(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty )
    visualize_loss(model_loss, frechet_loss, sample_sizes)








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for fitting manifold estimation model")
    # Add command-line arguments
    parser.add_argument("--manifold", type = str, help = "which type of manifold", default = "sphere")
    parser.add_argument("--loss", type = str, help="which type of loss", default = "L2")
    parser.add_argument('--noise',type=lambda x: [float(item) for item in x.split(",")], help='noise to jitter generated geodesics')
    parser.add_argument('--penalty',type=float, help='how much to penalize prior')
    parser.add_argument('--keep_preds', action="store_true", help='whether or not to plot convergence', default=False)
    parser.add_argument('--val', action="store_true", help='whether or not to compute validation loss', default=True)
    parser.add_argument('--timesteps', type=int,  help='length of trajectories')
    parser.add_argument('--sample_sizes',type=lambda x: [int(item) for item in x.split(",")], help='comma separated list of ints')
    parser.add_argument('--hemisphere', action="store_true", help="fit one part of the manifold", default = False)
    args = parser.parse_args()

    torch.manual_seed(12)
    if args.manifold == "circle":
        if args.hemisphere:
            circle_metric_hemisphere_with_n(sample_sizes = args.sample_sizes, noise_level = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps, loss_type=args.loss)

        else:
            circle_metric_with_n(sample_sizes = args.sample_sizes, noise_level = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps, loss_type=args.loss)
    elif args.manifold == "sphere":
        sphere_metric_with_n(sample_sizes = args.sample_sizes, noise = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps, loss_type=args.loss)





    
        