from typing import List, Tuple
import torch
import torch.optim as optim
from .model import RiemannianAutoencoder


def train(input_trajectories, initial_conditions: Tuple[torch.Tensor, torch.Tensor], epochs, regularizer:float, n, t, m:List[int], c:float, basis, active_dims: List):
    model = RiemannianAutoencoder(n,t,m,c,regularizer, basis, active_dims)
    param_list = []
    for gp in model.gp_components:
        param_list += list(gp.parameters())
    optimizer = optim.Adam(param_list, lr=0.001)


    for epoch in range(epochs):
        
        optimizer.zero_grad()
        # Forward pass
        predicted_trajectories, _ = model.forward(initial_conditions)
        loss = model.loss(torch.permute(predicted_trajectories.float(), (1,0,2)), input_trajectories.float())
        # Backward pass and optimization
        
        loss.backward(retain_graph=True)
        optimizer.step()

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")



    return model






# import numpy as np
# from .manifold import GPRiemannianEuclidean
# from .hsgp import HSGPExpQuadWithDerivative
# from scipy.spatial.distance import directed_hausdorff
# from scipy.stats import norm
# from scipy.optimize import minimize


# def symmetric_prior_loss(manifold_basis, metric_space: GPRiemannianEuclidean):
#     metric_matrix = metric_space.metric.metric_matrix(manifold_basis)
#     val = metric_matrix[:,0,0] * np.square(manifold_basis[:,0]) + metric_matrix[:,1,1] * np.square(manifold_basis[:,1]) - 2*metric_matrix[:,0,1]*np.prod(manifold_basis, axis = 1)
#     return np.mean(np.square(val - 1))

# def lie_derivative_circle_loss(manifold_basis, metric_space: GPRiemannianEuclidean):
#     #### only works if dimension is 2
#     christoffels = metric_space.metric.christoffels(manifold_basis)

#     return np.square(manifold_basis[:,0]*(christoffels[:,1,:,:].sum(axis=(-1,-2))) - manifold_basis[:,1]*(christoffels[:,0,:,:].sum(axis=(-1,-2)))).sum()

# def optim_func_loss(theta, input_trajectories, initial_conditions, manifold_basis, gps: list[HSGPExpQuadWithDerivative]):
    
#     beta = np.reshape(theta, (len(gps), -1))
#     beta_loss = -1*norm.logpdf(theta, loc = 1, scale = 1).sum()
#     for i, gp in enumerate(gps):
#         gp._beta = beta[i,:]
#     riemannian_metric_space = GPRiemannianEuclidean(2,gps,equip=True)
#     predicted_vals_geodesics = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:2],
#                                                                             initial_tangent_vec=initial_conditions[:,2:4])
#     t = np.arange(0,1,input_trajectories.shape[1])
#     predicted_vals = predicted_vals_geodesics(t)
    
#     average_manifold_distance  = np.sum(riemannian_metric_space.metric.dist(input_trajectories[:,:,0:2], predicted_vals[:,:,0:2]), axis = 1).sum()
#     return average_manifold_distance + beta_loss.sum() + symmetric_prior_loss(manifold_basis, riemannian_metric_space)



# def optim_func_loss_l2(theta, input_trajectories, initial_conditions, manifold_basis, gps: list[HSGPExpQuadWithDerivative], n_dims: int):
    
#     beta = np.reshape(theta, (len(gps), -1))
#     beta_loss = -1*norm.logpdf(theta, loc = 1, scale = 1).sum()
#     for i, gp in enumerate(gps):
#         gp._beta = beta[i,:]
    
#     riemannian_metric_space = GPRiemannianEuclidean(n_dims,gps,equip=True)
#     predicted_vals_geodesics = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:n_dims],
#                                                                             initial_tangent_vec=initial_conditions[:,n_dims:2*n_dims])
#     t = np.linspace(0,1,input_trajectories.shape[1])
#     predicted_vals = predicted_vals_geodesics(t)
#     average_manifold_distance = np.sum(np.square(input_trajectories[:,:,0:n_dims] - predicted_vals[:,:,0:n_dims]), axis = 1).sum()
#     prior_loss =  lie_derivative_circle_loss(manifold_basis, riemannian_metric_space)
#     print(f"Manifold L2 loss: {average_manifold_distance}", f"parameter loss: {beta_loss.sum()}", f"prior loss: {prior_loss}")
#     return average_manifold_distance  + beta_loss + prior_loss

# def optim_func_loss_l2_full(theta, input_trajectories, initial_conditions, manifold_basis, gps: list[HSGPExpQuadWithDerivative], n_dims: int):
    
#     beta = np.reshape(theta, (len(gps), -1))
#     beta_loss = -1*norm.logpdf(theta, loc = 1, scale = 1).sum()
#     for i, gp in enumerate(gps):
#         gp._beta = beta[i,:]
    
#     riemannian_metric_space = GPRiemannianEuclidean(n_dims,gps,manifold_basis,equip=True)
#     predicted_vals_geodesics = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:n_dims],
#                                                                             initial_tangent_vec=initial_conditions[:,n_dims:2*n_dims])
#     t = np.linspace(0,1,input_trajectories.shape[1])
#     predicted_vals = predicted_vals_geodesics(t)
#     predicted_velocity = np.stack([np.gradient(predicted_vals[i], 1/input_trajectories.shape[0], edge_order=2, axis = 0) for i in range(len(predicted_vals))], 
#                                   axis = 0)
#     actual_velocity = np.stack([np.gradient(input_trajectories[i], 1/input_trajectories.shape[0], edge_order = 2, axis = 0) for i in range(len(predicted_vals))],
#                                axis = 0)
#     average_manifold_distance = np.sum(np.square(input_trajectories[:,:,0:n_dims] - predicted_vals[:,:,0:n_dims]) + 
#                                         np.square(actual_velocity[:,:,0:n_dims] - predicted_velocity[:,:,0:n_dims]), axis = 1).sum()
#     prior_loss =  lie_derivative_circle_loss(manifold_basis, riemannian_metric_space)
#     print(f"Manifold L2 Full loss: {average_manifold_distance}", f"parameter loss: {beta_loss.sum()}", f"prior loss: {prior_loss}")
#     return average_manifold_distance  + beta_loss.sum() + prior_loss


# def optim_func_loss_hausdorff(theta, input_trajectories, initial_conditions, manifold_basis, gps: list[HSGPExpQuadWithDerivative], n_dims: int):
    
#     beta = np.reshape(theta, (len(gps), -1))
#     beta_loss = -1*norm.logpdf(theta, loc = 1, scale = 1).sum()
#     for i, gp in enumerate(gps):
#         gp._beta = beta[i,:]
    
#     riemannian_metric_space = GPRiemannianEuclidean(n_dims,gps,equip=True)
#     predicted_vals_geodesics = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:n_dims],
#                                                                             initial_tangent_vec=initial_conditions[:,n_dims:n_dims*2])
#     t = np.linspace(0,1,input_trajectories.shape[1])
#     predicted_vals = predicted_vals_geodesics(t)
    
#     flattened_predicted_vals = np.vstack(predicted_vals)
#     flattened_trajectories = np.vstack(input_trajectories)
#     average_manifold_distance = max(directed_hausdorff(flattened_predicted_vals, flattened_trajectories)[0],
#                                     directed_hausdorff(flattened_trajectories, flattened_predicted_vals)[0])
    
#     prior_loss =  lie_derivative_circle_loss(manifold_basis, riemannian_metric_space)
#     print(f"Manifold Hausdorff loss: {average_manifold_distance}", f"parameter loss: {beta_loss.sum()}", f"prior loss : {prior_loss}")
#     return average_manifold_distance  + beta_loss.sum() + prior_loss

# def minimize_function(gps: list[HSGPExpQuadWithDerivative], input_trajectories, initial_conditions, manifold_basis, n_dims:int, loss = "l2"):

#     init_beta = np.concatenate([gp._beta for gp in gps])
#     if loss == "l2":
#         loss_func = optim_func_loss_l2
#     elif loss == "l2_full":
#         loss_func = optim_func_loss_l2_full
#     else:
#         loss_func = optim_func_loss
#     results = minimize( loss_func, x0 = init_beta, args = (input_trajectories, initial_conditions, manifold_basis, gps, n_dims))

#     return results.x


# def minimize_function_mcmc(gps: list[HSGPExpQuadWithDerivative], input_trajectories, initial_conditions, manifold_basis, n_dims, loss_type = "l2", burn_in_samples = 100, num_samples = 100):
#     beta = np.concatenate([gp._beta for gp in gps])

#     if loss_type == "l2":
#         loss_func = optim_func_loss_l2
#     elif loss_type == "l2_full":
#         loss_func = optim_func_loss_l2_full
#     else:
#         loss_func = optim_func_loss
#     loss = loss_func(beta, input_trajectories, initial_conditions, manifold_basis, gps, n_dims)

#     i = 0 
#     betas = []
#     avg_ratio = 1
#     while i <= num_samples + burn_in_samples:
        
#         new_beta = .27*np.random.randn(beta.shape[0]) + beta
#         new_loss = loss_func(new_beta, input_trajectories, initial_conditions, manifold_basis, gps, n_dims)
#         ratio = min(1, loss / new_loss)
#         avg_ratio = ((ratio) + avg_ratio*(i+1))/(i+2)
#         alpha = np.random.rand()
#         print(f"Iteration {i}: Acceptance ratio: {ratio}, Average acceptance ratio: {avg_ratio}")
#         if alpha <= ratio:
#             beta = new_beta
#             loss = new_loss
#         else:
#             beta = beta
#             loss = loss
        
#         if i > burn_in_samples:
#             betas.append(beta.reshape((len(gps),-1)))
#         i += 1
#     return np.stack(betas,axis = 0 )




