import matplotlib.pyplot as plt
from typing import List
import numpy as np
from models.model import RiemannianAutoencoder
import torch
from pathlib import Path




def visualize_convergence(pred_trajectories: np.ndarray, actual_trajectories: np.ndarray, n:int, epoch_num: int,  val: bool, noise: int, hemisphere:bool = False,penalty: float = 0,  prior:bool = False, autoencoder:bool = False):

    plt.scatter(actual_trajectories[:,0], actual_trajectories[:,1], alpha=.3,color='blue', label = "Actual")
    plt.scatter(pred_trajectories[:,0], pred_trajectories[:,1], color = "red", alpha = .3, label = "Predicted")
    theta = np.linspace(0, 2 * np.pi, 100)

    x_circle =  np.cos(theta)
    y_circle =  np.sin(theta)

    # Plot the circle
    plt.plot(x_circle, y_circle, color='red', linestyle='dashed', label='Manual Circle')
    plt.xlim((-1.5,1.5))
    plt.ylim((-1.5,1.5))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    prior_path = "normal" 
    if prior:
        if penalty > 0:
             prior_path = "prior"
        elif penalty == 0:
            prior_path = "explicit_prior"
    if autoencoder:
        prior_path = "autoencoder"

    training_path = "val" if val else "training"
    fpath = f"data/{'plots' if not hemisphere else 'hemisphere_plots'}/{prior_path}/{training_path}/{n}/{noise}"

        # Specify the directory path
    directory_path = Path(fpath)

    # Check if the directory exists
    if not directory_path.exists():
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    plt.savefig(f"{fpath}/epoch_{epoch_num}_convergence_data.png")
    plt.clf()
    plt.close()

def visualize_convergence_sphere(pred_trajectories: np.ndarray, actual_trajectories: np.ndarray, n:int, noise:int,  epoch_num: int, hemisphere:bool = False,penalty: float = 0,  prior:bool = False, val:bool = False):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(actual_trajectories[:,0], actual_trajectories[:,1], actual_trajectories[:,2], alpha=.3,color='blue', label = "Actual")
    ax.scatter(pred_trajectories[:,0], pred_trajectories[:,1], pred_trajectories[:,2], alpha=.3,color='red', label = "Predicted")


    theta = np.linspace(0, 2 * np.pi, 10)
    phi = np.linspace(0, 2*np.pi, 10)

    u, v = np.meshgrid(theta, phi)
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    ax.plot_wireframe(x, y, z, color="green", label = "sphere", linestyle="dashed")


 

    prior_path = "normal"
    if prior:
        if penalty > 0:
             prior_path = "prior"
        elif penalty == 0:
            prior_path = "explicit_prior"

    training_path = "val" if val else "training"
    fpath = f"data/{'sphere_plots' if not hemisphere else 'sphere_hemisphere_plots'}/{prior_path}/{training_path}/{n}/{noise}"

        # Specify the directory path
    directory_path = Path(fpath)

    # Check if the directory exists
    if not directory_path.exists():
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    ax.legend()
    plt.savefig(f"{directory_path}/epoch_{epoch_num}_convergence_data.png")
    plt.clf()
    plt.close()




def visualize_circle_metric(model: RiemannianAutoencoder, basis: np.ndarray, n:int,  noise: int, hemisphere:bool = False, penalty: float = 0,  prior:bool = False):
    metric_matrix = model.metric_space.metric_matrix(basis)
    colors = metric_matrix[:,0,0]*basis[:,1]**2 + metric_matrix[:,1,1]*basis[:,0]**2 - 2*metric_matrix[:, 1, 0]*torch.prod(basis, axis=1)
    plt.scatter(basis[:,0], basis[:,1], c=colors, cmap='viridis')
    # Add labels and a colorbar
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.title('Metric Evaluation')
    plt.colorbar(label='Metric Value')

    prior_path = "normal"
    if prior:
        if penalty > 0:
             prior_path = "prior"
        elif penalty == 0:
            prior_path = "explicit_prior"
    fpath = f"data/{'plots' if not hemisphere else 'hemisphere_plots'}/{prior_path}/training/{n}/{noise}"
    directory_path = Path(fpath)

    # Check if the directory exists
    if not directory_path.exists():
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    plt.savefig(f"{fpath}/metric.png")
    # Show the plot
    plt.clf()
    plt.close()



def visualize_loss(loss_1: np.ndarray, loss_2: np.ndarray, n: List[int]):

    plt.plot(n, loss_1, label='L2 Loss')
    plt.plot(n, loss_2, label='Frechet Loss')

    # Add labels and a legend
    plt.xlabel('Training Data Size (samples)')
    plt.ylabel('Loss')
    # plt.title('L2 Loss vs. Frechet Loss')
    plt.legend()
    plt.savefig("data/plots/losses.png")

    # Show the plot
    plt.show()




