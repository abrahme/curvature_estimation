import matplotlib.pyplot as plt
from typing import List
import numpy as np
from scipy.interpolate import CubicSpline
from models.model import RiemannianAutoencoder
import torch
from pathlib import Path




def visualize_convergence(pred_trajectories: np.ndarray, actual_trajectories: np.ndarray, n:int, val:bool = False, manifold:str = "", noise:int=0):

    ax = plt.figure().add_subplot()

    n, T, _ = pred_trajectories.shape
    time = np.linspace(0, 1, 200)
    for i in range(n):
        predicted_smooth = CubicSpline(np.linspace(0,1, T), pred_trajectories[i])(time)
        actual_smooth = CubicSpline(np.linspace(0, 1, T), actual_trajectories[i])(time)
        ax.plot(predicted_smooth[:,0], predicted_smooth[:,1], color = "red", label = "prediction")
        ax.plot(actual_smooth[:, 0], actual_smooth[:, 1], label = "actual", color = "blue")
    training_path = "val" if val else "training"
    fpath = f"data/plots/{manifold}/{training_path}/{n}/{noise}"

        # Specify the directory path
    directory_path = Path(fpath)

    # Check if the directory exists
    if not directory_path.exists():
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    plt.savefig(f"{fpath}/convergence_data.png")
    plt.clf()
    plt.close()

def visualize_convergence_sphere(pred_trajectories: np.ndarray, actual_trajectories: np.ndarray, n:int, noise:int,  epoch_num: int, hemisphere:bool = False,penalty: float = 0,  prior:bool = False, val:bool = False):
    ax = plt.figure().add_subplot(projection='3d')

    n, T, _ = pred_trajectories.shape
    
    time = np.linspace(0, T, 200)

    for i in range(n):
        predicted_smooth = CubicSpline(time, pred_trajectories[i])(time)
        actual_smooth = CubicSpline(time, actual_trajectories[i])(time)
        ax.plot(actual_smooth[:,0], actual_smooth[:,1], actual_smooth[:,2], alpha=.3,color='blue', label = "Actual")
        ax.plot(predicted_smooth[:,0], predicted_smooth[:,1], predicted_smooth[:,2], alpha=.3,color='red', label = "Predicted")



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






    

def visualize_eigenvectors(A_true, A_learned, n, x_lims, y_lims, manifold: str):
    fig, ax = plt.subplots(1,1, sharex = True, sharey=True)
    x = torch.linspace(x_lims[0], x_lims[1], n)
    y = torch.linspace(y_lims[0], y_lims[1], n)
    X, Y = torch.meshgrid(x, y)
    U1_true = np.zeros((n, n))  # x-component of vector field
    V1_true = np.zeros((n, n))  # y-component of vector field
    U1_learned = np.zeros((n, n))  # x-component of vector field
    V1_learned = np.zeros((n, n))  # y-component of vector field
    U2_learned = np.zeros((n, n))  # x-component of vector field
    V2_learned = np.zeros((n, n))  # y-component of vector field
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]]).to("cuda:0")
            M_true = A_true(pt.cpu() if manifold =="normal_distribution" else pt).squeeze().detach()
            sorted_evals_true, sorted_evecs_true = torch.linalg.eigh(M_true)
            U1_true[i, j] = sorted_evecs_true[0, 0]
            V1_true[i, j] = sorted_evecs_true[1, 0]
            M_learned = A_learned(pt).squeeze().detach()
            sorted_evals_learned, sorted_evecs_learned = torch.linalg.eigh(M_learned)
            U1_learned[i, j] = sorted_evecs_learned[0, 0]
            V1_learned[i, j] = sorted_evecs_learned[1, 0]
            U2_learned[i, j] = sorted_evecs_learned[0, 1]
            V2_learned[i, j] = sorted_evecs_learned[1, 1]
    ax.quiver(X, Y, U1_true, V1_true, color="#5D3A9B")
    ax.quiver(X, Y, -U1_true, -V1_true, color="#5D3A9B")
    ax.quiver(X, Y, U2_learned, V2_learned, color="#E66100") # was U1_learned, V1_learned
    ax.quiver(X, Y, -U2_learned, -V2_learned, color="#E66100") # was -U1_learned, -V1_learned
    fig.suptitle(f"Recovered and Estimated Eigenvectors for {manifold}")
    plt.tight_layout()
    plt.savefig(f'data/eigenvectors/convergence_{manifold}.png',bbox_inches='tight')