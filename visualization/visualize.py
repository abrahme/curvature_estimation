import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.model import RiemannianAutoencoder
import torch


def visualize_training_data(trajectories: np.ndarray, tangent_vecs:np.ndarray = None, train:bool = True):
    sns.kdeplot(trajectories)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Density of Trajectory Data')
    plt.show()
    plt.clf()

    if train:
        sns.kdeplot(tangent_vecs)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Density Map of Sampled Tangent Space')
        plt.show()
        plt.clf()


def visualize_circle_metric(model: RiemannianAutoencoder, basis: np.ndarray):
    metric_matrix = model.metric_space.metric.metric_matrix(basis)
    colors = metric_matrix[:,0,0]*basis[:,1]**2 + metric_matrix[:,1,1]*basis[:,0]**2 - 2*metric_matrix[:, 1, 0]*torch.prod(basis, axis=1)
    plt.scatter(basis[:,0], basis[:,1], c=colors, cmap='viridis')

    # Add labels and a colorbar
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Metric Evaluation')
    plt.colorbar(label='Color')
    # Show the plot
    plt.show()
    plt.clf()



def visualize_loss(loss_1: np.ndarray, loss_2: np.ndarray):
    n = list(range(len(loss_1)))
    plt.plot(n, loss_1, label='L2 Loss')
    plt.plot(n, loss_2, label='Frechet Loss')

    # Add labels and a legend
    plt.xlabel('Training Data Size (samples)')
    plt.ylabel('Loss')
    plt.title('L2 Loss vs. Frechet Loss')
    plt.legend()

    # Show the plot
    plt.show()