import numpy as np

def set_boundary(Xs , c):
    """Set the boundary using the mean-subtracted `Xs` and `c`.  `c` is usually a scalar
    multiplyer greater than 1.0, but it may be one value per dimension or column of `Xs`.
    """
    S = np.max(np.abs(Xs), axis=0)
    L = c * S
    return L


def calc_eigenvalues(L, m,):
    """Calculate eigenvalues of the Laplacian."""
    S = np.meshgrid(*[np.arange(1, 1 + m[d]) for d in range(len(m))])
    S_arr = np.vstack([s.flatten() for s in S]).T
    return np.square((np.pi * S_arr) / (2 * L))


def calc_eigenvectors(
    Xs,
    L,
    eigvals,
    m):
    """Calculate eigenvectors of the Laplacian. These are used as basis vectors in the HSGP
    approximation.
    """
    m_star = int(np.prod(m))
    phi = np.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / np.sqrt(L[d])
        term1 = np.sqrt(eigvals[:, d])
        term2 = np.tile(Xs[:, d][:, None], m_star) + L[d]
        phi *= c * np.sin(term1 * term2)
    return phi

def calc_eigenvectors_deriv(
    Xs,
    L,
    eigvals,
    m,
    deriv_dim: int = 0
):
    """Calculate eigenvectors of the Laplacian. These are used as basis vectors in the HSGP
    approximation.
    """
    m_star = int(np.prod(m))
    phi = np.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / np.sqrt(L[d])
        term1 = np.sqrt(eigvals[:, d])
        term2 = np.tile(Xs[:, d][:, None], m_star) + L[d]
        phi *= c * np.sin(term1 * term2) if d != deriv_dim else c*np.cos(term1 * term2) * term1
    return phi




class HSGPExpQuadWithDerivative(object):
    def __init__(self, m, c, active_dims,
                drop_first: bool = False, parametrization: str = "noncentered",
                L = None):
        
        self._drop_first = drop_first
        self.parametrization = parametrization
        self.active_dims = active_dims
        self.n_dims = len(active_dims)
        self._m = m
        self._L = L
        self._c = c
        self._m_star = int(np.prod(m))
        self._beta = np.random.randn(self._m_star,)
        self.ls = np.random.random(size = self.n_dims)
    
    def _slice(self, X, Xs=None):
        X = X[:, self.active_dims]
        if Xs is not None:
            Xs = Xs[:, self.active_dims]
        return X, Xs
    
    def prior_linearized(self, Xs):
        ### returns with derivative values 
        # Index Xs using input_dim and active_dims of covariance function
        self._X_mean = np.mean(Xs, axis=0)
        Xs, _ = self._slice(Xs - self._X_mean)
        
        # If not provided, use Xs and c to set L
        if self._L is None:
            self.L = set_boundary(Xs, self._c)
        else:
            self.L = self._L

        i = int(self._drop_first == True)
        eigvals = calc_eigenvalues(self.L, self._m)
        omega = np.sqrt(eigvals)
        psd = self.power_spectral_density(omega)
        self._sqrt_psd = np.sqrt(psd[i:])
        self._eigvals = eigvals

    def power_spectral_density_deriv(self, omega, deriv_dim):
        ls = np.ones(self.n_dims) * self.ls
        return self.power_spectral_density(omega) * np.prod(ls) * np.square(omega[:,deriv_dim])

    def power_spectral_density(self, omega):
        r"""
        The power spectral density for the ExpQuad kernel is:

        .. math::

           S(\boldsymbol\omega) =
               (\sqrt(2 \pi)^D \prod_{i}^{D}\ell_i
                \exp\left( -\frac{1}{2} \sum_{i}^{D}\ell_i^2 \omega_i^{2} \right)
        """
        ls = np.ones(self.n_dims) * self.ls
        c = np.power(np.sqrt(2.0 * np.pi), self.n_dims)
        exp = np.exp(-0.5 * np.dot(np.square(omega), np.square(ls)))
        return c * np.prod(ls) * exp

    
    def predict(self, Xnew, deriv_dim : int | None = None):
        Xnew, _ = self._slice(Xnew - self._X_mean)
        if deriv_dim is None:
            phi = calc_eigenvectors(Xnew, self.L, self._eigvals, self._m)
            prediction = phi[:,0:] @ (self._beta * self._sqrt_psd)
            return prediction
        else:
            phi_deriv = calc_eigenvectors_deriv(Xnew, self.L, self._eigvals, self._m, deriv_dim)
            prediction = phi_deriv[:,0:] @ (self._beta * self._sqrt_psd)
            return prediction
    

        


    

