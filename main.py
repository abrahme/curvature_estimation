import numpy as np
import matplotlib.pyplot as plt
from models.basic_model import minimize_function, minimize_function_mcmc
from data.toy_examples import create_geodesic_pairs_circle
from models.hsgp import HSGPExpQuadWithDerivative
from models.manifold import GPRiemannianEuclidean


if __name__ == "__main__":
    basis_x, basis_y = np.meshgrid(np.arange(-1,1, .01), np.arange(-1,1, .01))
    basis = np.stack((basis_x.ravel(), basis_y.ravel()), axis = 1)
    # print(basis.shape)

    trajectories, start_points, start_velo = create_geodesic_pairs_circle(10, 20)
    # initial_conditions_velo = np.stack([np.gradient(trajectory, 1/trajectory.shape[0], axis=0, edge_order=2)[0,:] 
    #                       for trajectory in trajectories], axis = 0)
    initial_conditions = np.hstack([start_points, start_velo])

    # traj_flattened = np.vstack(trajectories)
    # # predicted_flattened = np.vstack(predicted_trajectories(np.linspace(0,1,20)))
    # plt.scatter(traj_flattened[:,0], traj_flattened[:,1], c = "blue")
    # # plt.scatter(predicted_flattened[:,0], predicted_flattened[:,1], c = "red")
    # plt.title("Data ")
    # plt.savefig("sample_circle_data.png")
    # plt.show()

    ## initialize gp
    m = [5,5]
    c = 4.0
    ls = np.array([.25,.25])
    active_dims = [0,1]
    gps = [HSGPExpQuadWithDerivative(m=m,active_dims=active_dims,c = c ) for _ in range(3)]
    for gp in gps:
        gp.prior_linearized(basis)
        gp.ls = ls.copy()

    ### 
    scale = 1
   
    
    
    result = minimize_function(gps, trajectories, initial_conditions, scale, loss = "l2")
    # fitted_beta = np.mean(result, axis = 0) ### get posterior mea
    fitted_beta = np.reshape(result, (3,-1))
    # for i, gp in enumerate(gps):
    #     print(gp._beta)
    riemannian_metric_space = GPRiemannianEuclidean(2,gps,scale,equip=True)

    
    metric_tensor = riemannian_metric_space.metric.metric_matrix(basis)
    christoffels = riemannian_metric_space.metric.christoffels(basis)

    predicted_trajectories = riemannian_metric_space.metric.geodesic(initial_point = initial_conditions[:,0:2],
                                                                            initial_tangent_vec=initial_conditions[:,2:4])
    
    traj_flattened = np.vstack(trajectories)
    predicted_flattened = np.vstack(predicted_trajectories(np.linspace(0,1,20)))
    plt.scatter(traj_flattened[:,0], traj_flattened[:,1], c = "blue")
    plt.scatter(predicted_flattened[:,0], predicted_flattened[:,1], c = "red")
    plt.title("Data vs Actual")
    plt.savefig("sample_circle_data_vs_predicted.png")
    plt.show()
    
    for i in range(2):
        for j in range(2):
            title = f"Metric Tensor Component {i},{j}"
            plt.scatter(basis[:,0], basis[:,1], c = metric_tensor[:,i,j])
            plt.title(title)
            plt.colorbar()
            plt.savefig(f"tensor_component_{i}_{j}.png") 
            plt.show()
            plt.clf()
            

    for i in range(2):
        for j in range(2):
            for k in range(2):
                title = f"Christoffel Symbol {i},{j}, {k}"
                plt.scatter(basis[:,0], basis[:,1], c = christoffels[:,i,j,k])
                plt.title(title)
                plt.colorbar()
                plt.savefig(f"christoffel_symbol_{i}_{j}_{k}.png") 
                plt.show()
                plt.clf()
                





    
        