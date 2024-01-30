import numpy as np
import argparse
from pathlib import Path
import torch
import pandas as pd
from typing import List
from models.train import train, train_symmetric_circle, train_symmetric_sphere
from geomstats.geometry.hypersphere import Hypersphere
from data.toy_examples import create_geodesic_pairs_circle, create_geodesic_pairs_circle_hemisphere, create_geodesic_pairs_sphere, create_geodesic_pairs_sphere_hemisphere
from visualization.visualize import  visualize_convergence, visualize_convergence_sphere




def plot_convergence(preds: List[np.ndarray], actual: np.ndarray, skip_every: int, n:int, penalty: float = 0, val: bool = False, noise: int = 0, hemisphere:bool = False, prior:bool = False):
    num_epochs = len(preds)
    indices = range(0, num_epochs, skip_every)
    for index in indices:
        visualize_convergence(torch.reshape(preds[index],(-1, 2)), actual,epoch_num=index,n=n, penalty=penalty, val = val, noise=noise, hemisphere=hemisphere, prior = prior)

def plot_convergence_sphere(preds: List[np.ndarray], actual: np.ndarray, skip_every: int, n:int, penalty: float = 0, val: bool = False, hemisphere:bool = False, prior:bool = False, noise: int = 0):
    num_epochs = len(preds)
    indices = range(0, num_epochs, skip_every)
    for index in indices:
        visualize_convergence_sphere(torch.reshape(preds[index],(-1, 3)), actual,epoch_num=index,n=n, penalty=penalty, val = val, prior = prior, hemisphere = hemisphere, noise=noise)

def circle_metric_hemisphere_with_n(sample_sizes: List[int], noise_level: List[float], timesteps:int, keep_preds:bool = False, val:bool = True, loss_type:str = "L2",penalty: float = 0,  prior:bool = False):
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)
    # xi, yi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    # manifold_basis = torch.stack([xi.flatten(), yi.flatten()], axis = -1).to(torch.float32)
    losses = []

    latent_space = Hypersphere(dim = 1, equip=True)
    for noise in noise_level:
        for num_samps in sample_sizes:
            torch.set_default_dtype(torch.float32)
            trajectories, start_points, start_velo, val_trajectories, val_start_points, val_start_velo = create_geodesic_pairs_circle_hemisphere(num_samps, timesteps, noise = 1/noise)
            _, _, _, val_trajectories_clean, _, _ = create_geodesic_pairs_circle_hemisphere(num_samps, timesteps, noise = 0)
            sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
            val_sample_basis = torch.reshape(val_trajectories,(-1, n_dims))
            initial_conditions = torch.hstack((start_points, start_velo))
            val_initial_conditions = torch.hstack((val_start_points, val_start_velo))

            if prior:
                if penalty > 0:
                    print("Training normal metric with prior")
                    model, preds = train(input_trajectories = trajectories, initial_conditions=initial_conditions, epochs = 100, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis, active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
                elif penalty == 0:
                    print("Training symmetric metric")
                    model, preds = train_symmetric_circle(input_trajectories = trajectories, initial_conditions=initial_conditions, epochs = 100, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis, active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            else:
                print("Training normal metric without prior")
                model, preds = train(input_trajectories = trajectories, initial_conditions=initial_conditions, epochs = 100, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis, active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            with torch.no_grad():
                
                

                val_generated_trajectories = model.forward(val_initial_conditions)
                val_predicted_trajectories = torch.permute(val_generated_trajectories, (1,0,2))
                plot_convergence([torch.reshape(val_predicted_trajectories, (-1, n_dims))], val_sample_basis,n = num_samps, penalty = penalty, skip_every = 30, val = val, noise=noise, hemisphere=True, prior = prior)

                val_geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(val_predicted_trajectories, (-1, n_dims))), val_sample_basis).mean()
                losses.append({"loss_val":val_geodesic_distance, "n": num_samps, "noise": 1/noise, "loss_type": "geodesic"})
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories).mean().item(),
                              "n": num_samps, "noise":1/noise, "loss_type": "model"})
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories_clean).mean().item(),
                              "n": num_samps, "noise": 1/noise, "loss_type": "model_clean"})   
            if keep_preds:
                plot_convergence(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty, noise=noise, hemisphere=True, prior = prior )
    
    
    prior_path = "normal"
    if prior:
        if penalty > 0:
             prior_path = "prior"
        elif penalty == 0:
            prior_path = "explicit_prior"
    
    directory_path = Path(f"data/losses/hemisphere/circle/{prior_path}")

    # Check if the directory exists
    if not directory_path.exists():
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    pd.DataFrame(losses).to_csv(f"{directory_path}/losses.csv", index=False)


def circle_metric_with_n(sample_sizes: List[int], noise_level: float,timesteps:int, keep_preds:bool = False, val:bool = True, loss_type:str = "L2", penalty: float = 0, prior:bool = False):
    m = [5,5]
    c = 4.0
    active_dims = [0,1]
    n_dims = len(active_dims)
    # xi, yi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    # manifold_basis = torch.stack([xi.flatten(), yi.flatten()], axis = -1).to(torch.float32)

    losses = []
    latent_space = Hypersphere(dim = 1, equip=True)
    for noise in noise_level:
        for num_samps in sample_sizes:
            torch.set_default_dtype(torch.float32)
            trajectories, start_points, start_velo, val_trajectories, val_start_points, val_start_velo = create_geodesic_pairs_circle(num_samps, timesteps, noise = 1/noise)
            _, _, _, val_trajectories_clean, _, _ = create_geodesic_pairs_circle(num_samps, timesteps, noise = 0)
            sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
            val_sample_basis = torch.reshape(val_trajectories,(-1, n_dims))
            initial_conditions = torch.hstack((start_points, start_velo))
            val_initial_conditions = torch.hstack((val_start_points, val_start_velo))
            if prior:
                if penalty > 0:
                    print("Training normal metric with prior")
                    model, preds = train(trajectories, initial_conditions, epochs = 100, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis, active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
                elif penalty == 0:
                    print("Training symmetric metric")
                    model, preds = train_symmetric_circle(trajectories, initial_conditions, epochs = 100, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis, active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            else:
                print("Training normal metric without prior")
                model, preds = train(trajectories, initial_conditions, epochs = 100, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis, active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            
            with torch.no_grad():
                val_generated_trajectories = model.forward(val_initial_conditions)
                val_predicted_trajectories = torch.permute(val_generated_trajectories, (1,0,2))
                plot_convergence([torch.reshape(val_predicted_trajectories, (-1, n_dims))], val_sample_basis,n = num_samps, penalty = penalty, skip_every = 10, val = val, noise=noise, prior = prior)
                val_geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(val_predicted_trajectories, (-1, n_dims))), val_sample_basis).mean()
                losses.append({"loss_val":val_geodesic_distance, "n": num_samps, "noise": 1/noise, "loss_type": "geodesic"})
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories).mean().item(),
                              "n": num_samps, "noise": 1/noise, "loss_type": "model"})  
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories_clean).mean().item(),
                              "n": num_samps, "noise": 1/noise, "loss_type": "model_clean"})   
            if keep_preds:
                plot_convergence(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty, noise=noise, prior = prior )
    
    prior_path = "normal"
    if prior:
        if penalty > 0:
             prior_path = "prior"
        elif penalty == 0:
            prior_path = "explicit_prior"
    
    directory_path = Path(f"data/losses/normal/circle/{prior_path}")

    # Check if the directory exists
    if not directory_path.exists():
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    pd.DataFrame(losses).to_csv(f"{directory_path}/losses.csv", index=False)



def sphere_metric_with_n(sample_sizes: List[int], noise_level: List[float], timesteps:int, keep_preds:bool = False, val:bool = True, loss_type:str = "L2", penalty: float = 0, prior:bool = False):
    xi, yi, zi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    manifold_basis = torch.stack([xi.flatten(), yi.flatten(), zi.flatten()], axis = -1)
    m = [3,3,3]
    c = 4.0
    active_dims = [0,1,2]
    n_dims = len(active_dims)
    losses = []
    latent_space = Hypersphere(dim = 2, equip=True)
    for noise in noise_level:
        for num_samps in sample_sizes:
            torch.set_default_dtype(torch.float32)
            trajectories, start_points, start_velo, val_trajectories, val_start_points, val_start_velo = create_geodesic_pairs_sphere(num_samps, timesteps, noise = 1/noise)
            _, _, _, val_trajectories_clean, _, _ = create_geodesic_pairs_sphere(num_samps, timesteps, noise = 1/noise)
            sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
            val_sample_basis = torch.reshape(val_trajectories,(-1, n_dims))
            initial_conditions = torch.hstack((start_points, start_velo))
            val_initial_conditions = torch.hstack((val_start_points, val_start_velo))
            # visualize_training_data_sphere(sample_basis, num_samps)
            if prior:
                if penalty > 0:
                    print("Training normal metric with prior")
                    model, preds = train(trajectories, initial_conditions, epochs = 200, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
                elif penalty == 0:
                    print("Training symmetric metric")
                    model, preds = train_symmetric_sphere(trajectories, initial_conditions, epochs = 200, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            else:
                print("Training normal metric without prior")
                model, preds = train(trajectories, initial_conditions, epochs = 200, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            
            with torch.no_grad():
                val_generated_trajectories = model.forward(val_initial_conditions)
                val_predicted_trajectories = torch.permute(val_generated_trajectories, (1,0,2))
                plot_convergence_sphere([torch.reshape(val_predicted_trajectories, (-1, n_dims))], val_sample_basis,n = num_samps, penalty = penalty, skip_every = 30, val = True, prior = prior, hemisphere=False, noise = noise)

                val_geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(val_predicted_trajectories, (-1, n_dims))), val_sample_basis).sum()/num_samps
                losses.append({"loss_val":val_geodesic_distance.item(), "n": num_samps, "noise": 1/noise, "loss_type": "geodesic"})
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories).mean().item(),
                                "n": num_samps, "noise": 1/noise, "loss_type": "model"}) 
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories_clean).mean().item(),
                              "n": num_samps, "noise": 1/noise, "loss_type": "model_clean"}) 
            if keep_preds:
                plot_convergence_sphere(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty, prior = prior, hemisphere=False, val=False, noise = noise )
    
        prior_path = "normal"
        if prior:
            if penalty > 0:
                prior_path = "prior"
            elif penalty == 0:
                prior_path = "explicit_prior"
        
        directory_path = Path(f"data/losses/normal/sphere/{prior_path}")

        # Check if the directory exists
        if not directory_path.exists():
            # Create the directory
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
        pd.DataFrame(losses).to_csv(f"{directory_path}/losses.csv", index=False)
        

def sphere_metric_hemisphere_with_n(sample_sizes: List[int], noise_level: List[float], timesteps:int, keep_preds:bool = False, val:bool = True, loss_type:str = "L2", penalty: float = 0, prior:bool = False):
    xi, yi, zi = torch.meshgrid(torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50), torch.linspace(-1.5,1.5,50))
    manifold_basis = torch.stack([xi.flatten(), yi.flatten(), zi.flatten()], axis = -1)
    m = [3,3,3]
    c = 4.0
    active_dims = [0,1,2]
    n_dims = len(active_dims)
    losses = []
    latent_space = Hypersphere(dim = 2, equip=True)
    for noise in noise_level:
        for num_samps in sample_sizes:
            torch.set_default_dtype(torch.float32)
            trajectories, start_points, start_velo, val_trajectories, val_start_points, val_start_velo = create_geodesic_pairs_sphere_hemisphere(num_samps, timesteps, noise = 1/noise)
            _, _, _, val_trajectories_clean, _, _ = create_geodesic_pairs_sphere_hemisphere(num_samps, timesteps, noise = 1/noise)
            sample_basis = torch.reshape(trajectories,(-1, n_dims)) ### only construct basis from whatever points we have 
            val_sample_basis = torch.reshape(val_trajectories,(-1, n_dims))
            initial_conditions = torch.hstack((start_points, start_velo))
            val_initial_conditions = torch.hstack((val_start_points, val_start_velo))
            # visualize_training_data_sphere(sample_basis, num_samps)
            if prior:
                if penalty > 0:
                    print("Training normal metric with prior")
                    model, preds = train(trajectories, initial_conditions, epochs = 200, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
                elif penalty == 0:
                    print("Training symmetric metric")
                    model, preds = train_symmetric_sphere(trajectories, initial_conditions, epochs = 200, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = manifold_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            else:
                print("Training normal metric without prior")
                model, preds = train(trajectories, initial_conditions, epochs = 200, regularizer=penalty, n = n_dims,
                                t = timesteps, m = m, c = c, val_initial_conditions=val_initial_conditions, val_input_trajectories=val_trajectories,
                            basis = sample_basis.to(torch.float32), active_dims = active_dims, return_preds=keep_preds, val=val, loss_type = loss_type)
            
            with torch.no_grad():
                val_generated_trajectories = model.forward(val_initial_conditions)
                val_predicted_trajectories = torch.permute(val_generated_trajectories, (1,0,2))
                plot_convergence_sphere([torch.reshape(val_predicted_trajectories, (-1, n_dims))], val_sample_basis,n = num_samps, penalty = penalty, skip_every = 30, val = True, prior = prior, hemisphere=True, noise = noise)

                val_geodesic_distance = latent_space.metric.dist(latent_space.projection(torch.reshape(val_predicted_trajectories, (-1, n_dims))), val_sample_basis).sum()/num_samps
                losses.append({"loss_val":val_geodesic_distance.item(), "n": num_samps, "noise": 1/noise, "loss_type": "geodesic"})
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories).mean().item(),
                                "n": num_samps, "noise": 1/noise, "loss_type": "model"}) 
                losses.append({"loss_val":torch.square(val_predicted_trajectories - val_trajectories_clean).mean().item(),
                              "n": num_samps, "noise": 1/noise, "loss_type": "model_clean"}) 
            if keep_preds:
                plot_convergence_sphere(preds, sample_basis, skip_every=30, n = num_samps, penalty = penalty, prior = prior, hemisphere=True, val=False, noise = noise )
    
        prior_path = "normal"
        if prior:
            if penalty > 0:
                prior_path = "prior"
            elif penalty == 0:
                prior_path = "explicit_prior"
        
        directory_path = Path(f"data/losses/hemisphere/sphere/{prior_path}")

        # Check if the directory exists
        if not directory_path.exists():
            # Create the directory
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
        pd.DataFrame(losses).to_csv(f"{directory_path}/losses.csv", index=False)







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for fitting manifold estimation model")
    # Add command-line arguments
    parser.add_argument("--manifold", type = str, help = "which type of manifold", default = "sphere")
    parser.add_argument("--loss", type = str, help="which type of loss", default = "L2")
    parser.add_argument('--noise',type=lambda x: [float(item) for item in x.split(",")], help='noise to jitter generated geodesics')
    parser.add_argument('--penalty',type=float, help='how much to penalize prior', default = 0.0)
    parser.add_argument('--prior', action = "store_true", help='how much to penalize prior', default = False)
    parser.add_argument('--keep_preds', action="store_true", help='whether or not to plot convergence', default=False)
    parser.add_argument('--val', action="store_true", help='whether or not to compute validation loss', default=True)
    parser.add_argument('--timesteps', type=int,  help='length of trajectories')
    parser.add_argument('--sample_sizes',type=lambda x: [int(item) for item in x.split(",")], help='comma separated list of ints')
    parser.add_argument('--hemisphere', action="store_true", help="fit one part of the manifold", default = False)
    args = parser.parse_args()

    torch.manual_seed(12)
    if args.manifold == "circle":
        if args.hemisphere:
            circle_metric_hemisphere_with_n(sample_sizes = args.sample_sizes, noise_level = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps, loss_type=args.loss, prior=args.prior)

        else:
            circle_metric_with_n(sample_sizes = args.sample_sizes, noise_level = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps, loss_type=args.loss, prior=args.prior)
    elif args.manifold == "sphere":
        if args.hemisphere:
            sphere_metric_hemisphere_with_n(sample_sizes = args.sample_sizes, noise_level = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps, loss_type=args.loss, prior = args.prior)
        else:
            sphere_metric_with_n(sample_sizes = args.sample_sizes, noise_level = args.noise, penalty = args.penalty, keep_preds=args.keep_preds, timesteps=args.timesteps, loss_type=args.loss, prior = args.prior)





    
        