import numpy as np
from .manifold import TwoDimensionalGaussianProcessRiemmanianMetric, GPRiemannianEuclidean
from .hsgp import HSGPExpQuadWithDerivative

from scipy.stats import norm
from scipy.optimize import minimize



def optim_func_loss(theta, input_trajectories, initial_conditions, gps: list[HSGPExpQuadWithDerivative]):
    
    beta = np.reshape(theta, (len(gps), -1))
    beta_loss = 0
    average_manifold_distance = 0
    for i, gp in enumerate(gps):
        gp._beta = beta[i,:]
        beta_loss += -1*norm.logpdf(beta[i,:], loc = 0, scale = gp._sqrt_psd)
    riemannian_metric_space = GPRiemannianEuclidean(2,gps,equip=True)
    predicted_vals_geodesics = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:2],
                                                                            initial_tangent_vec=initial_conditions[:,2:4])
    t = np.arange(0,1,input_trajectories.shape[1])
    predicted_vals = predicted_vals_geodesics(t)
    for i in range(len(input_trajectories)):
        average_manifold_distance += np.mean(riemannian_metric_space.metric.dist(input_trajectories[i,:,0:2], predicted_vals[i,:,0:2]))
    return average_manifold_distance + beta_loss.mean()



def minimize_function(gps: list[HSGPExpQuadWithDerivative], input_trajectories, initial_conditions):

    init_beta = np.concatenate([gp._beta for gp in gps])

    results = minimize(optim_func_loss, x0 = init_beta, args = (input_trajectories, initial_conditions, gps,))

    return results.x





