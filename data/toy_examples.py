import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.information_geometry.normal import NormalDistributions
from geomstats.information_geometry.beta import BetaDistributions
import torch





def find_non_nan_rows(tensor):
    non_nan_mask = ~torch.isnan(tensor)
    non_nan_rows = torch.all(non_nan_mask, dim=(1, 2))  # Specify the remaining dimensions
    non_nan_indices = torch.where(non_nan_rows)[0]
    return non_nan_indices

def create_geodesic_pairs_beta_dist(N, time_steps, noise = 0, dim = 1):
    torch.set_default_dtype(torch.float32)
    space = BetaDistributions()
    t = torch.linspace(0,1,time_steps)
    start_points = space.random_point(n_samples=N)
    start_tangent_vecs = space.random_tangent_vec(start_points)
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    nan_index_train = find_non_nan_rows(geodesic_trajectories)
    geodesic_trajectories_clean = geodesic_trajectories[nan_index_train]
    start_points_clean = start_points[nan_index_train]
    start_tangent_vecs_clean = start_tangent_vecs[nan_index_train]
    noise_vec = torch.randn(*geodesic_trajectories_clean.shape) * noise
    geodesic_trajectories_clean += noise_vec

    val_size = 2*N
    val_start_points = space.random_point(n_samples=val_size)
    val_start_tangent_vecs = space.random_tangent_vec(val_start_points)
    val_geodesic = space.metric.geodesic(initial_point=val_start_points, initial_tangent_vec = val_start_tangent_vecs)
    val_geodesic_trajectories = torch.unsqueeze(val_geodesic(t),0) if N == 1 else val_geodesic(t)
    nan_index_val = find_non_nan_rows(val_geodesic_trajectories)
    val_geodesic_trajectories_clean = val_geodesic_trajectories[nan_index_val]
    val_start_points_clean = val_start_points[nan_index_val]
    val_start_tangent_vecs_clean = val_start_tangent_vecs[nan_index_val]
    val_geodesic_trajectories = torch.randn(*val_geodesic_trajectories_clean.shape) * noise + val_geodesic_trajectories_clean
    return geodesic_trajectories_clean, start_points_clean, start_tangent_vecs_clean, val_geodesic_trajectories, val_start_points_clean, val_start_tangent_vecs_clean, val_geodesic_trajectories_clean

def create_geodesic_pairs_normal_dist(N, time_steps, noise = 0, dim = 1):
    torch.set_default_dtype(torch.float32)
    space = NormalDistributions(sample_dim=dim)
    t = torch.linspace(0,1,time_steps)
    start_points = space.random_point(n_samples=N)
    start_tangent_vecs = space.random_tangent_vec(start_points)
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    nan_index_train = find_non_nan_rows(geodesic_trajectories)
    geodesic_trajectories_clean = geodesic_trajectories[nan_index_train]
    start_points_clean = start_points[nan_index_train]
    start_tangent_vecs_clean = start_tangent_vecs[nan_index_train]
    noise_vec = torch.randn(*geodesic_trajectories_clean.shape) * noise
    geodesic_trajectories_clean += noise_vec

    val_size = 2*N
    val_start_points = space.random_point(n_samples=val_size)
    val_start_tangent_vecs = space.random_tangent_vec(val_start_points)
    val_geodesic = space.metric.geodesic(initial_point=val_start_points, initial_tangent_vec = val_start_tangent_vecs)
    val_geodesic_trajectories = torch.unsqueeze(val_geodesic(t),0) if N == 1 else val_geodesic(t)
    nan_index_val = find_non_nan_rows(val_geodesic_trajectories)
    val_geodesic_trajectories_clean = val_geodesic_trajectories[nan_index_val]
    val_start_points_clean = val_start_points[nan_index_val]
    val_start_tangent_vecs_clean = val_start_tangent_vecs[nan_index_val]
    val_geodesic_trajectories = torch.randn(*val_geodesic_trajectories_clean.shape) * noise + val_geodesic_trajectories_clean
    return geodesic_trajectories_clean, start_points_clean, start_tangent_vecs_clean, val_geodesic_trajectories, val_start_points_clean, val_start_tangent_vecs_clean, val_geodesic_trajectories_clean


def create_geodesic_pairs_circle(N,time_steps, noise = 0):
    torch.set_default_dtype(torch.float32)
    space = Hypersphere(dim = 1)
    t = torch.linspace(0,1,time_steps)
    start_points = space.random_point(n_samples=N)
    omega = torch.rand((N,1))*(np.pi/5) 
    start_tangent_vecs = torch.stack([-1*start_points[:,1] , start_points[:,0]], axis = -1)*omega
    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = torch.unsqueeze(geodesic(t),0) if N == 1 else geodesic(t)
    noise_vec = torch.distributions.Normal(0,1).sample(geodesic_trajectories.shape) * noise
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