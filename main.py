import numpy as np
import matplotlib.pyplot as plt
from models.basic_model import minimize_function, minimize_function_mcmc
from data.toy_examples import create_geodesic_pairs_circle
from models.hsgp import HSGPExpQuadWithDerivative
from models.manifold import GPRiemannianEuclidean


if __name__ == "__main__":
    basis_x, basis_y = np.meshgrid(np.arange(-1,1, .01), np.arange(-1,1, .01))
    basis = np.stack((basis_x.ravel(), basis_y.ravel()), axis = 1)

    trajectories, start_points = create_geodesic_pairs_circle(50, 20)
    initial_conditions_velo = np.stack([np.gradient(trajectory, 1/trajectory.shape[0], axis=0, edge_order=2)[0,:] 
                          for trajectory in trajectories], axis = 0)
    initial_conditions = np.hstack([start_points, initial_conditions_velo])

    ### initialize gp
    m = [15,15]
    c = 4.0
    ls = np.array([.5,.5])
    active_dims = [0,1]
    gps = [HSGPExpQuadWithDerivative(m=m,active_dims=active_dims,c = c ) for _ in range(3)]
    for gp in gps:
        gp.prior_linearized(basis)
        gp.ls = ls.copy()

    ### 
    traj_flattened = np.vstack(trajectories)
    plt.scatter(traj_flattened[:,0], traj_flattened[:,1])
    plt.title("Data")
    plt.show()
    plt.savefig("sample_circle_data.png")
    
    result = minimize_function_mcmc(gps, trajectories, initial_conditions, loss = "l2")
    fitted_beta = np.mean(result, axis = 0) ### get posterior mea
    # fitted_beta = np.reshape(result, (3,-1))
    for i, gp in enumerate(gps):
        gp._beta = fitted_beta[i,:]
    riemannian_metric_space = GPRiemannianEuclidean(2,gps,equip=True)

    metric_tensor = riemannian_metric_space.metric.metric_matrix(basis)
    christoffels = riemannian_metric_space.metric.christoffels(basis)


    
    for i in range(2):
        for j in range(2):
            title = f"Metric Tensor Component {i},{j}"
            plt.scatter(basis[:,0], basis[:,1], c = metric_tensor[:,i,j])
            plt.title(title)
            plt.colorbar()
            plt.show()
            plt.savefig(f"tensor_component_{i}_{j}.png") 

    for i in range(2):
        for j in range(2):
            for k in range(2):
                title = f"Christoffel Symbol {i},{j}, {k}"
                plt.scatter(basis[:,0], basis[:,1], c = christoffels[:,i,j,k])
                plt.title(title)
                plt.colorbar()
                plt.show()
                plt.savefig(f"christoffel_symbol_{i}_{j}_{k}.png") 





    
        