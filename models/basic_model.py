import pymc as pm
import pytensor.tensor as pt 
import pytensor
from pytensor.graph.basic import Variable
from pymc.pytensorf import constant_fold
import warnings
import numpy as np

class DerivativeExpQuad(pm.gp.cov.Covariance):
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

           S(\boldsymbol\omega) = \alpha^2 8 \pi^3 \omega^2 l^2 e^{-4l^2\pi^2\omega^2}
               
        """
        ls = pt.ones(self.n_dims) * self.ls
        c = pt.power(pt.sqrt(2.0 * np.pi), self.n_dims)
        exp = pt.exp(-2 * pt.dot(pt.square(omega), pt.square(ls*np.pi)))
        d =  4 * pt.square(self.alpha) * pt.prod(ls) * pt.square(np.pi) * pt.square(omega[:,self.deriv_dim])
        return c * exp * d
        


def build_metric_model(x_features, y_features, target_x, target_y, points, m):
    with pm.Model() as metric_model:
        ### prior on output
        sigma_x = pm.Exponential("sigma_x", 1)
        sigma_y = pm.Exponential("sigma_y", 1)
        basis = points
        ### prior on gp 
        ls_cov = pm.Exponential("ls_cov", .5)
        alpha = pm.Normal("alpha",mu = 0, sigma = 1)
        ### base functions
        cov_func_base = pm.gp.cov.ExpQuad(input_dim = 2, ls = [ls_cov,ls_cov], active_dims=[0,1])
        cov_func_deriv_x = DerivativeExpQuad(alpha = alpha, input_dim = 2, deriv_dim = 0, ls = [ls_cov,ls_cov],active_dims=[0,1])
        cov_func_deriv_y = DerivativeExpQuad(alpha = alpha, input_dim = 2, deriv_dim = 1, ls = [ls_cov,ls_cov],active_dims=[0,1])
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
        g_22 = pm.Deterministic("g_22", pm.math.sqr(l_22) + g_12) 

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
        delta_g_22_x = pm.Deterministic("delta_g_22_x", delta_g_12_x +  2*df_22_x*pm.math.sqr(l_22))
        delta_g_22_y = pm.Deterministic("delta_g_22_y", delta_g_12_y +  2*df_22_y*pm.math.sqr(l_22))


        #### christoffel symbols
        christoffel_x_11 = pm.Deterministic("christoffel_x_11",
                                            .5 * g_11_inv*(delta_g_11_x) + .5*g_12_inv*(2*delta_g_12_x - delta_g_11_y ))
        christoffel_y_11 = pm.Deterministic("christoffel_y_11", 
                                            .5*g_12_inv*(delta_g_11_x) +.5*g_22_inv*(2*delta_g_12_x - delta_g_11_y))
        
        christoffel_x_12 = pm.Deterministic("christoffel_x_12",
                                            .5*g_11_inv*(delta_g_11_y) + .5*g_12_inv*(delta_g_22_x + delta_g_12_x - delta_g_12_y))
        christoffel_y_12 = pm.Deterministic("christoffel_y_12",
                                            .5*g_12_inv*(delta_g_11_y) + .5*g_22_inv*(delta_g_22_x + delta_g_12_x - delta_g_12_y))
        
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


