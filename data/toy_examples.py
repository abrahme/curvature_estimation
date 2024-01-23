import numpy as np
import geomstats._backend as gs
from geomstats.geometry.hypersphere import Hypersphere
import torch




def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def create_geodesic_pairs_circle(N,time_steps, noise = 0):
    torch.set_default_dtype(torch.float32)
    space = Hypersphere(dim = 1)
    t = torch.linspace(0,1,time_steps)
    start_points = space.random_point(n_samples=N)
    start_tangent_vecs = space.random_tangent_vec(start_points)


    geodesic = space.metric.geodesic(initial_point=start_points, initial_tangent_vec = start_tangent_vecs)
    geodesic_trajectories = gs.expand_dims(geodesic(t),0) if N == 1 else geodesic(t)
    geodesic_trajectories += torch.randn(*geodesic_trajectories.shape) * noise
    
    return geodesic_trajectories, start_points, start_tangent_vecs


    
