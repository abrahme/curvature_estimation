import torch
import pandas as pd
from models.train import train
from models.utils import save_model
from data.toy_examples import create_geodesic_pairs_normal_dist, create_geodesic_pairs_circle, create_geodesic_pairs_beta_dist, create_geodesic_pairs_sphere


timesteps = 5
experiment_parameters = {
    "circle": lambda N: create_geodesic_pairs_circle(N, time_steps=timesteps, noise = 0),
    "normal_distribution": lambda N: create_geodesic_pairs_normal_dist(N, time_steps=timesteps, noise = 0),
    "sphere": lambda N: create_geodesic_pairs_sphere(N, time_steps=timesteps, noise = 0),
    "beta_distribution": lambda N: create_geodesic_pairs_beta_dist(N, time_steps=timesteps, noise = 0)
}

num_samples = [5, 100, 1000]

manifolds = list(experiment_parameters.keys())

experiment_results = []
for manifold in manifolds:
    if manifold == "sphere":
        dim = 3
    else:
        dim = 2
    for sample_size in num_samples:
        data_generator = experiment_parameters[manifold]
        observed_trajectories, initial_point, initial_tangent_vec, validation_trajectories, validation_initial_point, validation_tangent_vec, _ = data_generator(sample_size)
        initial_conditions = initial_conditions = torch.hstack((initial_point, initial_tangent_vec))
        validation_initial_conditions = torch.hstack((validation_initial_point, validation_tangent_vec))
        model, preds, validation_loss = train(input_trajectories = observed_trajectories, initial_conditions=initial_conditions, epochs = 300,  n = dim,
                            t = timesteps,  val_initial_conditions=validation_initial_conditions, val_input_trajectories=validation_trajectories,
                          return_preds=True, val=True, loss_type = "L2", model_type="neural")
        experiment_dict = {"manifold":manifold, "timesteps": timesteps, "num_trajectories":sample_size, "dimension": dim, "MSE": validation_loss.item}
        experiment_results.append(experiment_dict)
        save_model(model, sample_size, manifold)

results_df = pd.DataFrame(experiment_results)
results_df.to_csv("data/sample_size_experiments.csv",index = False)
        
        
