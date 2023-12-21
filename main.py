import numpy as np
import matplotlib.pyplot as plt
from models.basic_model import minimize_function
from data.toy_examples import create_geodesic_pairs_circle
from models.hsgp import HSGPExpQuadWithDerivative


if __name__ == "__main__":
    basis_x, basis_y = np.meshgrid(np.arange(-1,1, .01), np.arange(-1,1, .01))
    basis = np.stack((basis_x.ravel(), basis_y.ravel()), axis = 1)

    trajectories, start_points = create_geodesic_pairs_circle(2, 20)

    initial_conditions_velo = np.stack([np.gradient(trajectory, 1/trajectory.shape[0], axis=0, edge_order=2)[0,:] 
                          for trajectory in trajectories], axis = 0)
    initial_conditions = np.hstack([start_points, initial_conditions_velo])

    ### initialize gp
    m = [5,5]
    c = 4.0
    ls = np.array([.8,.6])
    active_dims = [0,1]
    gps = [HSGPExpQuadWithDerivative(m=m,active_dims=active_dims,c = c ) for _ in range(3)]
    for gp in gps:
        gp.prior_linearized(basis)
        gp.ls = ls.copy()

    ### 
    
    result = minimize_function(gps, trajectories, initial_conditions, loss = "l2")
    
    for i,gp in enumerate(gps):
        predictions = gp.predict(basis)
        plt.scatter(basis[:,0], basis[:,1], predictions)
        plt.title(f"Cholesky Val: {i}")