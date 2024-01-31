import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
import torch


def create_geodesic_pairs_circle(N,time_steps, noise = 0):
    torch.set_default_dtype(torch.float32)
    space = Hypersphere(dim = 1)
    t = torch.linspace(0,1,time_steps)
    start_points = space.random_point(n_samples=N)
    omega = torch.rand((N,1))*(np.pi/5) 
    start_tangent_vecs = torch.stack([-1*start_points[:,1] , start_points[:,0]], axis = -1)*omega
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    noise_vec = torch.randn(*geodesic_trajectories.shape) * noise
    geodesic_trajectories += noise_vec

    val_size = 2*N
    val_start_points = space.random_point(n_samples=val_size)
    val_omega = torch.rand((val_size,1))*(np.pi/5) 
    val_start_tangent_vecs = torch.stack([-1*val_start_points[:,1] , val_start_points[:,0]], axis = -1)*val_omega
    val_geodesic = space.metric.geodesic(initial_point=val_start_points, initial_tangent_vec = val_start_tangent_vecs)
    val_geodesic_trajectories_clean = torch.unsqueeze(val_geodesic(t),0) if N == 1 else val_geodesic(t)
    val_geodesic_trajectories = torch.randn(*val_geodesic_trajectories_clean.shape) * noise + val_geodesic_trajectories_clean
    return geodesic_trajectories, start_points, start_tangent_vecs, val_geodesic_trajectories, val_start_points, val_start_tangent_vecs, val_geodesic_trajectories_clean


def create_geodesic_pairs_circle_hemisphere(N, time_steps, noise = 0):
    torch.set_default_dtype(torch.float32)
    space = Hypersphere(dim = 1)
    t = torch.linspace(0,1,time_steps)
    theta = torch.rand((N,1)) * np.pi ### sample angle in right quadrant 
    omega = torch.rand((N,1))*(np.pi/5)  ## sample 
    start_points = space.angle_to_extrinsic(theta)
    start_tangent_vecs = torch.stack([-1*start_points[:,1] , start_points[:,0]], axis = -1)*omega
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    noise_vec = torch.randn(*geodesic_trajectories.shape) * noise
    geodesic_trajectories += noise_vec

    val_start_points = space.angle_to_extrinsic(theta + np.pi)
    val_omega = omega
    val_start_tangent_vecs = torch.stack([-1*val_start_points[:,1] , val_start_points[:,0]], axis = -1)*val_omega
    val_geodesic = space.metric.geodesic(initial_point=val_start_points, initial_tangent_vec = val_start_tangent_vecs)
    val_geodesic_trajectories_clean = torch.unsqueeze(val_geodesic(t),0) if N == 1 else val_geodesic(t)
    val_geodesic_trajectories = noise_vec + val_geodesic_trajectories_clean
    return geodesic_trajectories, start_points, start_tangent_vecs, val_geodesic_trajectories, val_start_points, val_start_tangent_vecs, val_geodesic_trajectories_clean



def create_geodesic_pairs_sphere(N, time_steps, noise = 0):
    torch.set_default_dtype(torch.float32)
    space = Hypersphere(dim = 2)
    t = torch.linspace(0,1,time_steps)
    start_points = space.random_point(n_samples=N)
    start_tangent_vecs = space.random_tangent_vec(n_samples=N, base_point=start_points)
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    noise_vec = torch.randn(*geodesic_trajectories.shape) * noise
    geodesic_trajectories += noise_vec

    val_size = 2*N
    val_start_points = space.random_point(n_samples=val_size)
    val_start_tangent_vecs = space.random_tangent_vec(n_samples=val_size, base_point=val_start_points)
    val_geodesic = space.metric.geodesic(initial_point=val_start_points, initial_tangent_vec = val_start_tangent_vecs)
    val_geodesic_trajectories_clean = torch.unsqueeze(val_geodesic(t),0) if N == 1 else val_geodesic(t)

    val_geodesic_trajectories =  torch.randn(*val_geodesic_trajectories_clean.shape) * noise + val_geodesic_trajectories_clean
    return geodesic_trajectories, start_points, start_tangent_vecs, val_geodesic_trajectories, val_start_points, val_start_tangent_vecs, val_geodesic_trajectories_clean



    
def create_geodesic_pairs_sphere_hemisphere(N, time_steps, noise = 0):
    torch.set_default_dtype(torch.float32)
    space = Hypersphere(dim = 2)
    t = torch.linspace(0,1,time_steps)
    phi = torch.rand((N,1)) * np.pi ### sample angle in right quadrant 
    theta = torch.rand((N,1))*(2*np.pi)  ## sample 
    start_points = space.spherical_to_extrinsic(torch.hstack([theta, phi]))
    start_tangent_vecs = space.random_tangent_vec(n_samples = N, base_point=start_points)
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    noise_vec = torch.randn(*geodesic_trajectories.shape) * noise
    geodesic_trajectories += noise_vec

    val_start_points = space.spherical_to_extrinsic(torch.hstack([theta, phi + np.pi]))
    val_start_tangent_vecs = space.random_tangent_vec( n_samples = N, base_point = val_start_points)

    val_geodesic = space.metric.geodesic(initial_point=val_start_points, initial_tangent_vec = val_start_tangent_vecs)
    val_geodesic_trajectories_clean = torch.unsqueeze(val_geodesic(t),0) if N == 1 else val_geodesic(t)
    val_geodesic_trajectories = noise_vec + val_geodesic_trajectories_clean
    return geodesic_trajectories, start_points, start_tangent_vecs, val_geodesic_trajectories, val_start_points, val_start_tangent_vecs, val_geodesic_trajectories_clean