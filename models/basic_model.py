import pymc as pm
import pytensor.tensor as pt
from .hgsp_utils import HSGPExpQuadWithDerivative, set_boundary, calculate_conditional
import numpy as np

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

def one_step_ahead_prediction(theta_vec , dt, beta_coeffs, psd_original, m, L, cov_func, X_mean):
            """_summary_

            Args:
                state_vector (_type_): _description_
            dt (_type_): _description_
                base_gp_list (list[pm.gp.HSGP]): _description_
                deriv_x_gp_list (list[pm.gp.HSGP]): _description_
                deriv_y_gp_list (list[pm.gp.HSGP]): _description_
            """
            ### reshape into appropriate vectors
            X_new = theta_vec[:,0:2]
            dX_new = theta_vec[:, 2:4]

            
            g_matrix = create_g_matrix(X_new, beta_coeffs, [m,m], L, cov_func, X_mean, 
                                       psd_original) 
            partial_deriv_matrices = [create_delta_g_matrix(g_matrix, X_new, beta_coeffs, [m,m], L, cov_func, X_mean, 
                                                     psd_original, k) for k in range(2)]
            g_inv = create_g_matrix_inv(g_matrix)
            ### get christoffel symbols 
            christoffel_symbols = calculate_christoffel(g_inv, partial_deriv_matrices)
            #### simulate geodesic 
            
            d2x = -(christoffel_symbols[0][0]*pm.math.sqr(dX_new[:,0]) + 
                                                2 * christoffel_symbols[0][1]*pm.math.prod(dX_new, axis = 1, keepdims = False) + 
                                                christoffel_symbols[0][2] * pm.math.sqr(dX_new[:,1]))
            d2y = -(christoffel_symbols[1][0]*pm.math.sqr(dX_new[:,0]) + 
                            2 * christoffel_symbols[1][1]*pm.math.prod(dX_new, axis = 1, keepdims = False) + 
                            christoffel_symbols[1][2] * pm.math.sqr(dX_new[:,1]))
            dx = dX_new[:,0]  + dt * d2x
            dy = dX_new[:,1] + dt * d2y
            x = X_new[:,0] + dt * dX_new[:,0] + .5*pm.math.sqr(dt) * d2x
            y = X_new[:,1] + dt * dX_new[:,1] + .5*pm.math.sqr(dt) * d2y
            return pm.math.stack([x, y , dx, dy, d2x, d2y], axis = 1)

def create_g_matrix(X_new, beta, m, L, cov_func, X_mean, psd):
    """ creates a g matrix lower triangular matrix 

    Args:
        X_new (_type_): where to evaluate numpy array
        betas (list): list of predictors for gp
        psds (list): list of psd for gp
        cov_funcs (list): list of cov func for gp
        ms (list): list of m for gp
        Ls (list): list of L for gp 

    Returns:
        _type_: matrix 2 x 2
    """
    vals = [calculate_conditional(beta[:,i], X_new, X_mean, psd, cov_func, L, m) for i in range(3)]
    
    g_vals = [pm.math.sqr(vals[0]), vals[1], pm.math.sqr(vals[2])]
    return g_vals

def create_delta_g_matrix(g_matrix: list, X_new, beta, m, L, cov_func, X_mean, psd, deriv_dim: int ):
    """ creates a g matrix lower triangular matrix 

    Args:
        g_ matrix list
        X_new (_type_): where to evaluate numpy array 
        f (list): pm.gp.HSGP objects

    Returns:
        _type_: matrix 2 x 2
    """
    vals = [calculate_conditional(beta[:,i], X_new, X_mean, psd, cov_func, L, m, deriv_dim) for i in range(3)]
    
    d_g_11 = 2 * vals[0] * pm.math.sqrt(g_matrix[0])
    d_g_12 = vals[1]
    d_g_22 = 2 * vals[2] * pm.math.sqrt(g_matrix[2])
    return [d_g_11, d_g_12, d_g_22]


def create_g_matrix_inv(g_matrix_vals):
    determinant = 1/ (g_matrix_vals[0]*g_matrix_vals[-1] - pm.math.sqr(g_matrix_vals[1]))
    g_vals_inv = [ determinant * g_matrix_vals[-1], -1 * g_matrix_vals[1] * determinant, determinant * g_matrix_vals[0]]
    return g_vals_inv

def calculate_christoffel(g_matrix_inv, delta_g_matrices: list[list]):
    """ calculates the christoffel symbols 

    Args:
        g_matrix (_type_): _description_
        g_matrix_inv (_type_): _description_
        delta_g_matrices (list[list]): _description_

    Returns:
        _type_: list of list of christoffel symbols
    """

    christoffel_x_11 =  .5 * g_matrix_inv[0]*(delta_g_matrices[0][0]) + .5*g_matrix_inv[1]*(2*delta_g_matrices[0][1] - delta_g_matrices[1][0])
    christoffel_y_11 = .5*g_matrix_inv[1]*(delta_g_matrices[0][0]) +.5*g_matrix_inv[2]*(2*delta_g_matrices[0][1] - delta_g_matrices[1][0])
    
    christoffel_x_12 = .5*g_matrix_inv[0]*(delta_g_matrices[1][0]) + .5*g_matrix_inv[1]*(delta_g_matrices[0][2])
    christoffel_y_12 = .5*g_matrix_inv[1]*(delta_g_matrices[1][0]) + .5*g_matrix_inv[2]*(delta_g_matrices[0][2])
    
    christoffel_x_22 = .5*g_matrix_inv[0]*(2*delta_g_matrices[1][1] - delta_g_matrices[0][2]) + .5*g_matrix_inv[1]*(delta_g_matrices[1][2])
    
    christoffel_y_22 = .5*g_matrix_inv[1]*(2*delta_g_matrices[1][1] - delta_g_matrices[0][2]) + .5*g_matrix_inv[2]*(delta_g_matrices[1][2])

    symbols =  [[christoffel_x_11, christoffel_x_12, christoffel_x_22], [christoffel_y_11, christoffel_y_12, christoffel_y_22]]
    return symbols

def build_geodesic_model(basis, geodesics, m):
    """_summary_

    Args:
        basis (_type_): B x 2 ndarray describing basis for gaussian process
        geodesics (_type_): list of N t_i by 2 curves representing geodesics of length t_i 
        initial_conditions (_type_): list of N dictionaries representing initial position and velocity 
        m (_type_): number of basis expansions to use 
    """
    with pm.Model() as ode_model:
        ### prior on gp 
        # sigma = pm.HalfNormal("sigma")
        ls_cov = pm.HalfNormal("ls_cov")
        ### base functions for gp prior on grid
        cov_func_base = pm.gp.cov.ExpQuad(input_dim = 2, ls = ls_cov*pt.ones(2), active_dims=[0,1])
        X_mean = pt.mean(basis, axis = 0)
        L = set_boundary(basis - X_mean, c = 4.0)
        #### gp prior these are the same for all of our functions so only need to compute once
        phis, psds = HSGPExpQuadWithDerivative(m = [m, m], c = 4.0, cov_func = cov_func_base).prior_linearized(basis)
        psd, psd_deriv_x, psd_deriv_y = psds
        phi, phi_deriv_x, phi_deriv_y = phis
        ### partial derivs gp prior (only one for each of the cholesky elements)
        hsgp_beta_coeffs = pm.Normal("hsgp_coeffs", mu = 0, sigma = 1, shape = (m**2, 3))
        
        

        inputs = []
        targets = []

        for val in geodesics:
            # x,y,dx,dy = initial_conditions[i].astype(np.float64)
            # decorator with input and output types a Pytensor double float tensors
            dt = 1/val.shape[0]
            shifted_val = shift_array(val, -1)
            input_val = val[~np.isnan(shifted_val).any(axis = 1)]
            target_val = shifted_val[~np.isnan(shifted_val).any(axis = 1)]
            inputs.append(input_val)
            targets.append(target_val)
        total_inputs = np.vstack(inputs)
        total_targets = np.vstack(targets)
        print(total_targets.shape)
        print(total_inputs.shape)
        output_vals = one_step_ahead_prediction(total_inputs, dt, hsgp_beta_coeffs, psd, m, L, cov_func_base, X_mean)

        likelihood = pm.Normal(f"likelihood",mu = pm.math.flatten(output_vals), 
                                observed = total_targets.flatten(), sigma = .1)
    
    return ode_model

