import matplotlib.pyplot as plt
from typing import List
import numpy as np
from models.model import RiemannianAutoencoder
import torch
from scipy.stats import kde
import matplotlib.patches as patches

def visualize_convergence(pred_trajectories: np.ndarray, actual_trajectories: np.ndarray, n:int, epoch_num: int, penalty: float, val: bool):
    # nbins=300

    # k = kde.gaussian_kde(actual_trajectories.T)
    # xi, yi = np.meshgrid(np.linspace(-1.5,1.5,nbins), np.linspace(-1.5,1.5,nbins))
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)

    plt.scatter(actual_trajectories[:,0], actual_trajectories[:,1], alpha=.3,color='blue', label = "Actual")
    plt.scatter(pred_trajectories[:,0], pred_trajectories[:,1], color = "red", alpha = .3, label = "Predicted")
    theta = np.linspace(0, 2 * np.pi, 100)

    x_circle =  np.cos(theta)
    y_circle =  np.sin(theta)

    # Plot the circle
    plt.plot(x_circle, y_circle, color='red', linestyle='dashed', label='Manual Circle')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.savefig(f"data/plots/epoch_{epoch_num}_convergence_data_{n}{'_prior' if penalty > 0 else ''}{'_val' if val else ''}.png")
    plt.clf()


def visualize_training_data(trajectories: np.ndarray, n:int,  tangent_vecs:np.ndarray = None, train:bool = True, penalty:float = 0.0):
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    # nbins=300

    # k = kde.gaussian_kde(trajectories.T)
    # xi, yi = np.meshgrid(np.linspace(-1.5,1.5,nbins), np.linspace(-1.5,1.5,nbins))
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
    plt.scatter(trajectories[:,0], trajectories[:,1], alpha=.3,color='red' )
    theta = np.linspace(0, 2 * np.pi, 100)

    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    # Plot the circle
    plt.plot(x_circle, y_circle, color='red', linestyle='dashed', label='Manual Circle')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.savefig(f"data/plots/{'training' if train else 'predicted'}_data_{n}{'_prior' if penalty > 0 else ''}.png")
    plt.clf()

    if train:
        # k = kde.gaussian_kde(tangent_vecs.T)
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
        plt.scatter(trajectories[:,0], trajectories[:,1], alpha=.3,color='red' )
        plt.plot(x_circle, y_circle, color='red', linestyle='dashed', label='Manual Circle')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        # plt.title('Density Map of Sampled Tangent Space')
        plt.legend()
        plt.savefig(f"data/plots/training_data_tangent_{n}.png")
        plt.clf()


def visualize_circle_metric(model: RiemannianAutoencoder, basis: np.ndarray, n:int, penalty: float):
    metric_matrix = model.metric_space.metric.metric_matrix(basis)
    colors = metric_matrix[:,0,0]*basis[:,1]**2 + metric_matrix[:,1,1]*basis[:,0]**2 - 2*metric_matrix[:, 1, 0]*torch.prod(basis, axis=1)
    plt.scatter(basis[:,0], basis[:,1], c=colors, cmap='viridis')
    # Add labels and a colorbar
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.title('Metric Evaluation')
    plt.colorbar(label='Metric Value')
    plt.savefig(f"data/plots/circle_metric_{n}{'_prior' if penalty > 0 else ''}.png")
    # Show the plot
    plt.clf()



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