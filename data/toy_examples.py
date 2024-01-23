import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
import torch


def create_geodesic_pairs_circle(N,time_steps, noise = 0):
    torch.set_default_dtype(torch.float32)
    space = Hypersphere(dim = 1)
    t = torch.linspace(0,1,time_steps)
    start_points = space.random_point(n_samples=N)
    omega = torch.rand((N,1))*(2*np.pi/50)
    start_tangent_vecs = torch.stack([-1*start_points[:,1] , start_points[:,0]], axis = -1)*omega
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    geodesic_trajectories += torch.randn(*geodesic_trajectories.shape) * noise

    val_size = int(N/2)
    val_start_points = space.random_point(n_samples=val_size)
    val_omega = torch.rand((val_size,1))*(2*np.pi/50)
    val_start_tangent_vecs = torch.stack([-1*val_start_points[:,1] , val_start_points[:,0]], axis = -1)*val_omega
    val_geodesic = space.metric.geodesic(initial_point=val_start_points, initial_tangent_vec = val_start_tangent_vecs)
    val_geodesic_trajectories = torch.unsqueeze(val_geodesic(t),0) if N == 1 else val_geodesic(t)
    val_geodesic_trajectories += torch.randn(*val_geodesic_trajectories.shape) * noise
    return geodesic_trajectories, start_points, start_tangent_vecs, val_geodesic_trajectories, val_start_points, val_start_tangent_vecs


    
