import numpy as np
import matplotlib.pyplot as plt
from models.basic_model import minimize_function, minimize_function_mcmc
from data.toy_examples import create_geodesic_pairs_circle
from models.hsgp import HSGPExpQuadWithDerivative
from models.manifold import GPRiemannianEuclidean
from geomstats.geometry.hypersphere import Hypersphere

if __name__ == "__main__":
    basis_x, basis_y = np.meshgrid(np.arange(-1,1, .01), np.arange(-1,1, .01))
    basis = np.stack((basis_x.ravel(), basis_y.ravel()), axis = 1)
    theta = np.linspace(0, np.pi * 2, 1000)
    basis_on_manifold = np.vstack([np.cos(theta), np.sin(theta)]).T
    trajectories, start_points, start_velo = create_geodesic_pairs_circle(40, 20, noise=.1)
    initial_conditions = np.hstack([start_points, start_velo])
    m = [5,5]
    c = 4.0
    ls = np.array([.25,.25])
    active_dims = [0,1]
    n_dims = len(active_dims)
    n_gps = int(n_dims*(n_dims + 1)/2)
    gps = [HSGPExpQuadWithDerivative(m=m,active_dims=active_dims,c = c ) for _ in range(n_gps)]
    for gp in gps:
        gp.prior_linearized(basis)
        gp.ls = ls.copy()

   
    
    
    # result = minimize_function_mcmc(gps, trajectories, initial_conditions, scale, loss = "hausdorff", burn_in_samples=200, num_samples=500)
    result = minimize_function(gps, trajectories, initial_conditions, basis_on_manifold, n_dims, loss = "l2")
    # fitted_beta = np.mean(result, axis = 0) ### get posterior mea
    fitted_beta = np.reshape(result, (n_gps,-1))
    # for i, gp in enumerate(gps):
    #     gp._beta = fitted_beta[i,:]
    riemannian_metric_space = GPRiemannianEuclidean(n_dims,gps,equip=True)

    
    

    predicted_trajectories = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:n_dims],
                                                                            initial_tangent_vec=initial_conditions[:,n_dims:2*n_dims])
    
    traj_flattened = np.vstack(trajectories)
    predicted_flattened = np.vstack(predicted_trajectories(np.linspace(0,1,20)))
    plt.scatter(traj_flattened[:,0], traj_flattened[:,1], c = "blue")
    plt.scatter(predicted_flattened[:,0], predicted_flattened[:,1], c = "red")
    plt.title("Data vs Actual")
    plt.savefig("sample_circle_data_vs_predicted.png")
    plt.clf()
    
    space = Hypersphere(dim = n_dims - 1)
    
    metric_tensor = riemannian_metric_space.metric.metric_matrix(basis_on_manifold)
    christoffels = riemannian_metric_space.metric.christoffels(basis_on_manifold)
    for i in range(n_dims):
        for j in range(n_dims):
            title = f"Metric Tensor Component {i},{j}"
            plt.scatter(basis_on_manifold[:,0], basis_on_manifold[:,1], c = metric_tensor[:,i,j])
            plt.title(title)
            plt.colorbar()
            plt.savefig(f"tensor_component_{i}_{j}.png") 
            # plt.show()
            plt.clf()
            

    for i in range(n_dims):
        for j in range(n_dims):
            for k in range(n_dims):
                title = f"Christoffel Symbol {i},{j}, {k}"
                plt.scatter(basis_on_manifold[:,0], basis_on_manifold[:,1], c = christoffels[:,i,j,k])
                plt.title(title)
                plt.colorbar()
                plt.savefig(f"christoffel_symbol_{i}_{j}_{k}.png") 
                # plt.show()
                plt.clf()
    

    # Plot the straight line with a different color
    plt.scatter(basis_on_manifold[:,0], basis_on_manifold[:,1], c = metric_tensor[:,0,0]*np.square(basis_on_manifold[:,0]) + 
              metric_tensor[:,1,1]*np.square(basis_on_manifold[:,1]) - 2*metric_tensor[:,0,1]*basis_on_manifold[:,0]*basis_on_manifold[:,1], label='metric value')

    # Customize the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.colorbar(label='Estimated Metric Value')
    plt.title('Circle: $x^2 + y^2 = 1$')
    plt.legend()
    plt.savefig("EstimatedMetricValue.png")
    plt.show()





    
        