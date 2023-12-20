import pymc as pm
import numpy as np
import numbers
from pymc.gp.hsgp_approx import TensorLike
import pytensor.tensor as pt
from types import ModuleType
from typing import Sequence, Union
from pymc.pytensorf import constant_fold
import warnings
from pytensor.graph.basic import Variable



TensorLike = Union[np.ndarray, pt.TensorVariable]




def set_boundary(Xs: TensorLike, c: Union[numbers.Real, TensorLike]) -> TensorLike:
    """Set the boundary using the mean-subtracted `Xs` and `c`.  `c` is usually a scalar
    multiplyer greater than 1.0, but it may be one value per dimension or column of `Xs`.
    """
    S = pt.max(pt.abs(Xs), axis=0)
    L = c * S
    return L


def calc_eigenvalues(L: TensorLike, m: Sequence[int], tl: ModuleType = np):
    """Calculate eigenvalues of the Laplacian."""
    S = np.meshgrid(*[np.arange(1, 1 + m[d]) for d in range(len(m))])
    S_arr = np.vstack([s.flatten() for s in S]).T
    return tl.square((np.pi * S_arr) / (2 * L))


def calc_eigenvectors(
    Xs: TensorLike,
    L: TensorLike,
    eigvals: TensorLike,
    m: Sequence[int],
    tl: ModuleType = np,
):
    """Calculate eigenvectors of the Laplacian. These are used as basis vectors in the HSGP
    approximation.
    """
    m_star = int(np.prod(m))
    phi = tl.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / tl.sqrt(L[d])
        term1 = tl.sqrt(eigvals[:, d])
        term2 = tl.tile(Xs[:, d][:, None], m_star) + L[d]
        phi *= c * tl.sin(term1 * term2)
    return phi

def calc_eigenvectors_deriv(
    Xs: TensorLike,
    L: TensorLike,
    eigvals: TensorLike,
    m: Sequence[int],
    tl: ModuleType = np,
    deriv_dim: int = 0
):
    """Calculate eigenvectors of the Laplacian. These are used as basis vectors in the HSGP
    approximation.
    """
    m_star = int(np.prod(m))
    phi = tl.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / tl.sqrt(L[d])
        term1 = tl.sqrt(eigvals[:, d])
        term2 = tl.tile(Xs[:, d][:, None], m_star) + L[d]
        phi *= c * tl.sin(term1 * term2) if d != deriv_dim else c*tl.cos(term1 * term2) * term1
    return phi

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


class HSGPExpQuadWithDerivative(pm.gp.HSGP):
    def __init__(self, m: Sequence[int], c: numbers.Real, cov_func: pm.gp.cov.ExpQuad,
                drop_first: bool = False, parametrization: str = "noncentered",
                L: Sequence[float] | None = None):
        super(HSGPExpQuadWithDerivative, self).__init__(m = m, c = c,  drop_first = drop_first, 
                                                        parameterization = parametrization, cov_func=cov_func, L = L)
    
        self.cov_derivs = [DerivativeExpQuad(alpha = 1, ls = cov_func.ls, deriv_dim = i, input_dim = cov_func.input_dim, 
                                             active_dims = cov_func.active_dims) for i in range(len(m))]
    
    def prior_linearized(self, Xs: TensorLike):
        ### returns with derivative values 
        # Index Xs using input_dim and active_dims of covariance function
        Xs, _ = self.cov_func._slice(Xs)

        # If not provided, use Xs and c to set L
        if self._L is None:
            assert isinstance(self._c, (numbers.Real, np.ndarray, pt.TensorVariable))
            self.L = pt.as_tensor(set_boundary(Xs, self._c))
        else:
            self.L = self._L

        i = int(self._drop_first == True)
        eigvals = calc_eigenvalues(self.L, self._m, tl=pt)
        phi = calc_eigenvectors(Xs, self.L, eigvals, self._m, tl=pt)
        phi_derivs = [calc_eigenvectors_deriv(Xs, self.L, eigvals, self._m, tl = pt, deriv_dim=k)[:, i:]
                      for k in range(len(self.cov_derivs))]
        omega = pt.sqrt(eigvals)
        psd = self.cov_func.power_spectral_density(omega)
        psd_derivs = [cov_deriv.power_spectral_density(omega) for cov_deriv in self.cov_derivs]
        
        return [phi[:, i:]] + phi_derivs, [pt.sqrt(psd[i:])] + [pt.sqrt(psd_deriv[i:]) for psd_deriv in psd_derivs]
    


def calculate_conditional(beta, Xnew, X_mean, sqrt_psd, cov_func, L , m, deriv_dim: int | None = None):
    Xnew, _ = cov_func._slice(Xnew)

    eigvals = calc_eigenvalues(L, m, tl=pt)
    phi = calc_eigenvectors(Xnew - X_mean, L, eigvals, m, tl=pt) if not deriv_dim else calc_eigenvectors_deriv(Xnew - X_mean, L, eigvals, m, tl=pt, deriv_dim=deriv_dim)
    return  phi[:, 0:] @ (beta * sqrt_psd)






    

