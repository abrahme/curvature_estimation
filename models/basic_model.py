import pymc as pm
from pymc.pytensorf import constant_fold
import warnings
from pytensor.graph.basic import Variable
import pytensor.tensor as pt
from .hgsp_utils import build_conditional
import numpy as np


def create_g_matrix(X_new, gp_list: list[pm.gp.HSGP]):
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
    vals = [build_conditional(X_new, f) for 
            f in gp_list]
    g_vals = [pm.math.exp(2*vals[0]), vals[1]*vals[0], pm.math.exp(2*vals[2]) + pm.math.sqr(vals[1]*vals[0])]
    return g_vals

def create_delta_g_matrix(g_matrix: list, X_new, gp_list: list[pm.gp.HSGP]):
    """ creates a g matrix lower triangular matrix 

    Args:
        g_ matrix list
        X_new (_type_): where to evaluate numpy array 
        f (list): pm.gp.HSGP objects

    Returns:
        _type_: matrix 2 x 2
    """
    vals = [build_conditional(X_new, f) for 
            f in gp_list]
    
    d_g_11 = 2 * vals[0] * g_matrix[0]
    d_g_12 = pm.math.sqrt(g_matrix[0])*(vals[1]) + g_matrix[1] * vals[0]
    d_g_22 = 2*(d_g_12 * g_matrix[1] + vals[1]*(g_matrix[2] - pm.math.sqr(g_matrix[1])))
    return [d_g_11, d_g_12, d_g_22]


def create_g_matrix_inv(g_matrix_vals):
    determinant = 1/ (g_matrix_vals[0]*g_matrix_vals[-1]) - pm.math.sqr(g_matrix_vals[1])
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

    return [[christoffel_x_11, christoffel_x_12, christoffel_x_22], [christoffel_y_11, christoffel_y_12, christoffel_y_22]]





    

class DerivativeExpQuad(pm.gp.cov.Stationary):
    def __init__(self, alpha, ls, deriv_dim, input_dim, active_dims=None):
        super(DerivativeExpQuad, self).__init__(input_dim, active_dims)
        self.alpha = alpha
        self.ls = ls[deriv_dim]
        self.deriv_dim = deriv_dim

    def diag(self, X):
        return pt.mul((self.alpha**2)/(self.ls **2), pt.ones(X.shape[0]))

    def square_dist(self, X, Xs):
        X = pt.mul(X, 1.0 / self.ls)
        X2 = pt.sum(pt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * pt.dot(X, pt.transpose(X)) + (
                pt.reshape(X2, (-1, 1)) + pt.reshape(X2, (1, -1))
            )
        else:
            Xs = pt.mul(Xs, 1.0 / self.ls)
            Xs2 = pt.sum(pt.square(Xs), 1)
            sqd = -2.0 * pt.dot(X, pt.transpose(Xs)) + (
                pt.reshape(X2, (-1, 1)) + pt.reshape(Xs2, (1, -1))
            )
        return pt.clip(sqd, 0.0, np.inf)

    def euclidean_dist(self, X, Xs):
        r2 = self.square_dist(X, Xs)
        return self._sqrt(r2)

    def _sqrt(self, r2):
        return pt.sqrt(r2 + 1e-12)

    def _slice(self, X, Xs=None):
        xdims = X.shape[-1]
        if isinstance(xdims, Variable):
            [xdims] = constant_fold([xdims])
        if self.input_dim != xdims:
            warnings.warn(
                f"Only {self.input_dim} column(s) out of {xdims} are"
                " being used to compute the covariance function. If this"
                " is not intended, increase 'input_dim' parameter to"
                " the number of columns to use. Ignore otherwise.",
                UserWarning,
            )
        X = pt.as_tensor_variable(X[:, self.active_dims])
        if Xs is not None:
            Xs = pt.as_tensor_variable(Xs[:, self.active_dims])
        return X, Xs

    def full_from_distance(self, dist, squared: bool = False):
        r2 = dist if squared else dist**2
        return pt.exp(-0.5 * r2)
    
    def full(self, X, Xs = None):
        X, Xs = self._slice(X, Xs)
        r2 = self.square_dist(X, Xs)
        normal_cov = self.full_from_distance(r2, squared=True)

        X_deriv, Xs_deriv = X[:, self.deriv_dim], Xs[:, self.deriv_dim]
        r2_deriv = self.square_dist(X_deriv, Xs_deriv)
        return pt.mul(pt.mul((self.alpha**2 / self.ls **4),(pt.sum(self.ls**2, - r2_deriv))),normal_cov)
    
    def power_spectral_density(self, omega):
        r"""
        The power spectral density for the Derivative ExpQuad kernel is:

        .. math::

           S(\boldsymbol\omega) = 
               
        """
        ls = pt.ones(self.n_dims) * self.ls
        c = pt.power(pt.sqrt(2.0 * np.pi), self.n_dims)
        exp = pt.exp(-2 * pt.dot(pt.square(omega), pt.square(ls*np.pi)))
        d =  4 * pt.square(self.alpha) * pt.prod(ls) * pt.square(np.pi) * pt.square(omega[:,self.deriv_dim])
        return c * exp * d


def rhs_ode(state_vector, t, theta):

    """_summary_

    Args:
        state_vector (_type_): _description_
        t (_type_): _description_
        base_gp_list (list[pm.gp.HSGP]): _description_
        deriv_x_gp_list (list[pm.gp.HSGP]): _description_
        deriv_y_gp_list (list[pm.gp.HSGP]): _description_
    """
    ### reshape into appropriate vectors
    base_gp_list = theta[0]
    deriv_x_gp_list = theta[1]
    deriv_y_gp_list = theta[2]
    X_new =  state_vector[:, 0:2].reshape((2,1))
    dX_new = state_vector[:, 2:4].reshape((2,1))
    g_matrix = create_g_matrix(X_new, base_gp_list) 
    delta_g_matrix_x = create_delta_g_matrix(g_matrix, X_new, deriv_x_gp_list)
    delta_g_matrix_y = create_delta_g_matrix(g_matrix, X_new, deriv_y_gp_list)
    g_inv = create_g_matrix_inv(g_matrix)

    ### get christoffel symbols 
    christoffel_symbols = calculate_christoffel(g_inv, [delta_g_matrix_x, delta_g_matrix_y])
    #### simulate geodesic 

    return pm.math.concatenate([dX_new,
                pm.math.stack([-1 * (christoffel_symbols[0][0]*pm.math.sqr(dX_new[0]) + 
                                        2 * christoffel_symbols[0][1]*pm.math.prod(dX_new) + 
                                        christoffel_symbols[0][2] * pm.math.sqr(dX_new[1])),
                                
                -1 * (christoffel_symbols[1][0]*pm.math.sqr(dX_new[0]) + 
                    2 * christoffel_symbols[1][1]*pm.math.prod(dX_new) + 
                    christoffel_symbols[1][2] * pm.math.sqr(dX_new[1]))])])
        

def build_geodesic_model(basis, geodesics, initial_conditions, m):
    """_summary_

    Args:
        basis (_type_): B x 2 ndarray describing basis for gaussian process
        geodesics (_type_): list of N t_i by 2 curves representing geodesics of length t_i 
        initial_conditions (_type_): list of N dictionaries representing initial position and velocity 
        m (_type_): number of basis expansions to use 
    """
    with pm.Model() as ode_model:
        ### prior on gp 
        sigma = pm.HalfNormal("sigma")
        ls_cov = pm.HalfNormal("ls_cov")
        alpha = pm.Normal("alpha",mu = 0, sigma = 1)
        ### base functions for gp prior on grid
        cov_func_base = (alpha**2)*pm.gp.cov.ExpQuad(input_dim = 2, ls = ls_cov*pt.ones(2), active_dims=[0,1])
        cov_func_deriv_x = DerivativeExpQuad(alpha = alpha, input_dim = 2, deriv_dim = 0, ls = ls_cov*pt.ones(2) ,active_dims=[0,1])
        cov_func_deriv_y = DerivativeExpQuad(alpha = alpha, input_dim = 2, deriv_dim = 1, ls = ls_cov*pt.ones(2),active_dims=[0,1])

        #### gp prior
        f_11 = pm.gp.HSGP(m = [m, m], c = 4.0, cov_func = cov_func_base ).prior("f_11", X=basis)
        f_12 = pm.gp.HSGP(m = [m, m], c = 4.0, cov_func = cov_func_base ).prior("f_12", X=basis)
        f_22 = pm.gp.HSGP(m = [m, m], c = 4.0, cov_func = cov_func_base ).prior("f_22", X=basis)

        ### partial derivs gp prior

        df_11_x = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_x).prior("df_11_x", X=basis)
        df_12_x = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_x).prior("df_12_x", X=basis)
        df_22_x = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_x).prior("df_22_x", X=basis)

        df_11_y = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_y).prior("df_11_y", X=basis)
        df_12_y = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_y).prior("df_12_y", X=basis)
        df_22_y = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_y).prior("df_22_y", X=basis)

        theta_vec = [[f_11, f_12, f_22],
                    [df_11_x, df_12_x, df_22_x],
                    [df_11_y, df_12_y, df_22_y]]

        for i, val in enumerate(geodesics):
            ode_model = pm.ode.DifferentialEquation(func=rhs_ode, times = np.arange(1,len(val)), n_states=4, n_theta=3, t0=0)
            likelihood = pm.Normal(f"likelihood_{i}", mu = ode_model( y0=initial_conditions[i], theta = theta_vec), observed = val, sigma = sigma)
    
    return ode_model


    


    




def build_metric_model(x_features, y_features, target_x, target_y, points, m):
    with pm.Model() as metric_model:
        ### prior on output
        sigma_x = pm.Exponential("sigma_x", 1)
        sigma_y = pm.Exponential("sigma_y", 1)
        basis = points
        ### prior on gp 
        ls_cov = pm.HalfNormal("ls_cov")
        alpha = pm.Normal("alpha",mu = 0, sigma = 1)
        ### base functions
        cov_func_base = pm.gp.cov.ExpQuad(input_dim = 2, ls = ls_cov*pt.ones(2), active_dims=[0,1])
        cov_func_deriv_x = DerivativeExpQuad(alpha = alpha, input_dim = 2, deriv_dim = 0, ls = ls_cov*pt.ones(2) ,active_dims=[0,1])
        cov_func_deriv_y = DerivativeExpQuad(alpha = alpha, input_dim = 2, deriv_dim = 1, ls = ls_cov*pt.ones(2),active_dims=[0,1])
        gp = pm.gp.HSGP(m = [m, m], c = 4.0, cov_func = cov_func_base )
        f_11 = gp.prior("f_11", X = basis)
        f_12 = gp.prior("f_12", X = basis)
        f_22 = gp.prior("f_22", X = basis)

        ### derivative f base functions
        gp_deriv_x = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_x)
        gp_deriv_y = pm.gp.HSGP(m = [m,m], c = 4.0, cov_func = cov_func_deriv_y)
        df_11_x = gp_deriv_x.prior("df_11_x", X = basis)
        df_12_x = gp_deriv_x.prior("df_12_x", X = basis)
        df_22_x = gp_deriv_x.prior("df_22_x", X = basis)
        df_11_y = gp_deriv_y.prior("df_11_y", X = basis)
        df_12_y = gp_deriv_y.prior("df_12_y", X = basis)
        df_22_y = gp_deriv_y.prior("df_22_y", X = basis)
        

        ## cholesky components
        l_11 = pm.Deterministic("l_11", pm.math.exp(f_11))
        l_12 = pm.Deterministic("l_12", f_12)
        l_22 = pm.Deterministic("l_22", pm.math.exp(f_22))

        ## metric components
        g_11 = pm.Deterministic("g_11", pm.math.sqr(l_11))
        g_12 = pm.Deterministic("g_12", l_11 * l_12)
        g_22 = pm.Deterministic("g_22", pm.math.sqr(l_22) + pm.math.sqr(g_12))

        ## metric determinant and inverses
        metric_determinant = pm.Deterministic("determinant", g_11 * g_22 - pm.math.sqr(g_12))
        g_11_inv = pm.Deterministic("g_11_inv", g_22 / metric_determinant)
        g_12_inv = pm.Deterministic("g_12_inv", - g_12 / metric_determinant) ## note that since symmetric g_12 = g_21
        g_22_inv = pm.Deterministic("g_22_inv", g_11 / metric_determinant )

        ### derivatives
        delta_g_11_x = pm.Deterministic("delta_g_11_x", 2 * g_11 * df_11_x )
        delta_g_11_y = pm.Deterministic("delta_g_11_y", 2 * g_11 * df_11_y )
        delta_g_12_x = pm.Deterministic("delta_g_12_x", l_11 * (df_11_x*f_12 + df_12_x))
        delta_g_12_y = pm.Deterministic("delta_g_12_y", l_11 * (df_11_y*f_12 + df_12_y))
        delta_g_22_x = pm.Deterministic("delta_g_22_x", 2*delta_g_12_x*g_12 +  2*df_22_x*pm.math.sqr(l_22))
        delta_g_22_y = pm.Deterministic("delta_g_22_y", 2*delta_g_12_y*g_12 +  2*df_22_y*pm.math.sqr(l_22))


        #### christoffel symbols
        christoffel_x_11 = pm.Deterministic("christoffel_x_11",
                                            .5 * g_11_inv*(delta_g_11_x) + .5*g_12_inv*(2*delta_g_12_x - delta_g_11_y ))
        christoffel_y_11 = pm.Deterministic("christoffel_y_11", 
                                            .5*g_12_inv*(delta_g_11_x) +.5*g_22_inv*(2*delta_g_12_x - delta_g_11_y))
        
        christoffel_x_12 = pm.Deterministic("christoffel_x_12",
                                            .5*g_11_inv*(delta_g_11_y) + .5*g_12_inv*(delta_g_22_x))
        christoffel_y_12 = pm.Deterministic("christoffel_y_12",
                                            .5*g_12_inv*(delta_g_11_y) + .5*g_22_inv*(delta_g_22_x))
        
        christoffel_x_22 = pm.Deterministic("christoffel_x_22",
                                            .5*g_11_inv*(2*delta_g_12_y - delta_g_22_x) + .5*g_12_inv*(delta_g_22_y))
        
        christoffel_y_22 = pm.Deterministic("christoffel_y_22",
                                            .5*g_12_inv*(2*delta_g_12_y - delta_g_22_x) + .5*g_22_inv*(delta_g_22_y))


        christoffel_x = pm.Deterministic("christoffel_x", pm.math.stack([christoffel_x_11,christoffel_x_12,christoffel_x_22], axis = 0))
        christoffel_y = pm.Deterministic("christoffel_y", pm.math.stack([christoffel_y_11,christoffel_y_12,christoffel_y_22], axis = 0))

        predicted_x = pm.Deterministic("predicted_x", pm.math.sum(christoffel_x.T * x_features, axis = 1))
        predicted_y = pm.Deterministic("predicted_y", pm.math.sum(christoffel_y.T * y_features, axis = 1))

        likelihood_x = pm.Normal("likelihood_x", mu = predicted_x, sigma = sigma_x, observed=-1*target_x)
        likelihood_y = pm.Normal("likelihood_y", mu = predicted_y, sigma = sigma_y, observed = -1*target_y)
    return metric_model


