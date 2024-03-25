import torch
import pandas as pd
from geomstats.information_geometry.normal import NormalDistributions
from geomstats.geometry.hypersphere import Hypersphere
from models.train import train
from models.utils import save_model
from data.toy_examples import create_geodesic_pairs_normal_dist, create_geodesic_pairs_circle, create_geodesic_pairs_sphere
from visualization.visualize import visualize_eigenvectors, visualize_convergence

timesteps = 10

def circle_metric(x):
    vector_field = torch.matmul(x, torch.tensor([[0.0 , -1.0],[1.0, 0.0]]).to("cuda:0"))
    return torch.einsum("kj,ki -> kij", vector_field, vector_field)

def fisher_rao_univariate_normal_metric(x):
    metric =  torch.diag_embed(torch.tensor([1.0, 2.0]).to("cuda:0") / torch.square(x[1]))
    return metric

experiment_parameters = {

    "normal_distribution": {"sampling_func": lambda N: create_geodesic_pairs_normal_dist(N, time_steps=timesteps, noise = 0),
                             "metric_func": lambda x: torch.vmap(fisher_rao_univariate_normal_metric)(x)},
    "sphere": {"sampling_func": lambda N: create_geodesic_pairs_sphere(N, time_steps=timesteps, noise = 0), "metric_func": None},
        "circle": {"sampling_func": lambda N: create_geodesic_pairs_circle(N, time_steps=timesteps, noise = 0),
               "metric_func": circle_metric},
}

# torch.manual_seed(123)

num_samples = [5, 100, 1000]


manifolds = list(experiment_parameters.keys())

experiment_results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for manifold in manifolds:
    if manifold == "sphere":
        dim = 3
    else:
        dim = 2
    for sample_size in num_samples:
        data_generator = experiment_parameters[manifold]["sampling_func"]
        observed_trajectories, initial_point, initial_tangent_vec, validation_trajectories, validation_initial_point, validation_tangent_vec, _ = data_generator(sample_size)
        initial_conditions = initial_conditions = torch.hstack((initial_point, initial_tangent_vec))
        validation_initial_conditions = torch.hstack((validation_initial_point, validation_tangent_vec))
        model, preds, validation_loss = train(input_trajectories = observed_trajectories, initial_conditions=initial_conditions, epochs = 500,  n = dim,
                            t = timesteps,  val_initial_conditions=validation_initial_conditions, val_input_trajectories=validation_trajectories,
                          return_preds=True, val=True, loss_type = "L2", model_type="neural", hidden_dim=100)

        save_model(model, sample_size, manifold)
        eig_val_loss = None

        if dim == 2:
            num_points = 30

            # Generate points along each axis
            x = torch.linspace(-1.5, 1.5, num_points)
            y = torch.linspace(-1.5, 1.5, num_points)
            if manifold == "normal_distribution":
                y = torch.linspace(.01, 1.5, num_points)

            # Create a meshgrid
            X, Y = torch.meshgrid(x, y)

            # Flatten the grid
            
            grid = torch.stack((X.flatten(), Y.flatten()), dim=1).to(device)

            metric_func = experiment_parameters[manifold]["metric_func"]
            ground_truth_metric = metric_func(grid).to(device)
            batched_eig_vals = torch.vmap(lambda x: torch.linalg.eig(x)[1])
            ground_truth_eigvals = torch.real(batched_eig_vals(ground_truth_metric))
            estimated_metric = model.metric_space.metric_matrix(grid)
            estimated_eig_vals = torch.real(batched_eig_vals(estimated_metric))
            eig_val_loss = torch.abs(torch.nn.CosineSimilarity()(estimated_eig_vals,ground_truth_eigvals)).mean().item()

            visualize_eigenvectors(A_true = metric_func, A_learned = model.metric_space.metric_matrix, n = num_points,
                                   x_lims = [-1.5, 1.5], y_lims=[.01,1.5] if manifold == "normal_distribution" else [-1.5,1.5], manifold = manifold)
            visualize_convergence(preds[-1].detach().cpu(), observed_trajectories, sample_size, val = True, noise =0, manifold = manifold)
        experiment_dict = {"manifold":manifold, "timesteps": timesteps, "num_trajectories":sample_size, "dimension": dim, "MSE": validation_loss.item(),
                           "eigenvalue_loss": eig_val_loss}
        experiment_results.append(experiment_dict)

        

results_df = pd.DataFrame(experiment_results)
results_df.to_csv("data/sample_size_experiments.csv",index = False)
        
        
