import numpy as np
from .manifold import GPRiemannianEuclidean
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
    
    average_manifold_distance  = np.mean(riemannian_metric_space.metric.dist(input_trajectories[:,:,0:2], predicted_vals[:,:,0:2]), axis = 1).sum()
    return average_manifold_distance + beta_loss.sum()



def optim_func_loss_l2(theta, input_trajectories, initial_conditions, gps: list[HSGPExpQuadWithDerivative]):
    
    beta = np.reshape(theta, (len(gps), -1))
    beta_loss = 0
    average_manifold_distance = 0
    for i, gp in enumerate(gps):
        gp._beta = beta[i,:]
        beta_loss += -1*norm.logpdf(beta[i,:], loc = 0, scale = gp._sqrt_psd)
    
    riemannian_metric_space = GPRiemannianEuclidean(2,gps,equip=True)
    predicted_vals_geodesics = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:2],
                                                                            initial_tangent_vec=initial_conditions[:,2:4])
    t = np.linspace(0,1,input_trajectories.shape[1])
    predicted_vals = predicted_vals_geodesics(t)
    average_manifold_distance = np.mean(np.square(input_trajectories[:,:,0:2] - predicted_vals[:,:,0:2]), axis = 1).sum()
    return average_manifold_distance  + beta_loss.sum()

def minimize_function(gps: list[HSGPExpQuadWithDerivative], input_trajectories, initial_conditions, loss = "l2"):

    init_beta = np.concatenate([gp._beta for gp in gps])

    results = minimize(optim_func_loss if loss != "l2" else optim_func_loss_l2, x0 = init_beta, args = (input_trajectories, initial_conditions, gps,))

    return results.x


def minimize_function_mcmc(gps: list[HSGPExpQuadWithDerivative], input_trajectories, initial_conditions, loss = "l2", burn_in_samples = 100, num_samples = 100):
    beta = np.concatenate([gp._beta for gp in gps])
    loss = optim_func_loss_l2(beta, input_trajectories, initial_conditions, gps)

    i = 0 
    betas = []
    avg_ratio = 1
    while i <= num_samples + burn_in_samples:
        
        new_beta = .27*np.random.randn(beta.shape[0]) + beta
        new_loss = optim_func_loss_l2(new_beta, input_trajectories, initial_conditions, gps)

        ratio = min(1, loss / new_loss)
        avg_ratio = ((ratio) + avg_ratio*(i+1))/(i+2)
        alpha = np.random.rand()
        print(f"Iteration {i}: Acceptance ratio: {ratio}, Average acceptance ratio: {avg_ratio}")
        if alpha <= ratio:
            beta = new_beta
            loss = new_loss
        else:
            beta = beta
            loss = loss
        
        if i > burn_in_samples:
            betas.append(beta.reshape((3,-1)))
        i += 1
    
    return np.stack(betas,axis = 0 )




