import numpy as np
import torch
import torch.nn as nn

def set_boundary(Xs , c):
    """Set the boundary using the mean-subtracted `Xs` and `c`.  `c` is usually a scalar
    multiplyer greater than 1.0, but it may be one value per dimension or column of `Xs`.
    """
    S, _ = torch.max(torch.abs(Xs),0)
    L = c * S
    return L


def calc_eigenvalues(L, m):
    """Calculate eigenvalues of the Laplacian."""
    S = torch.meshgrid(*[torch.arange(1, 1 + m[d]) for d in range(len(m))])
    S_arr = torch.vstack([torch.flatten(s) for s in S]).T
    return torch.square((np.pi * S_arr) / (2 * L)).to(torch.float64)


def calc_eigenvectors(
    Xs,
    L,
    eigvals,
    m):
    """Calculate eigenvectors of the Laplacian. These are used as basis vectors in the HSGP
    approximation.
    """
    m_star = int(np.prod(m))
    phi = torch.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / torch.sqrt(L[d])
        term1 = torch.sqrt(eigvals[:, d])
        term2 = torch.tile(Xs[:, d][:, None], (m_star,)) + L[d]
        phi *= c * torch.sin(term1 * term2)
    return phi.to(torch.float64)

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
    phi = torch.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / torch.sqrt(L[d])
        term1 = torch.sqrt(eigvals[:, d])
        term2 = torch.tile(Xs[:, d][:, None], (m_star,)) + L[d]
        phi *= c * torch.sin(term1 * term2) if d != deriv_dim else c*torch.cos(term1 * term2) * term1
    return phi.to(torch.float64)




class HSGPExpQuadWithDerivative(nn.Module):
    def __init__(self, m, c, active_dims,
                drop_first: bool = False, parametrization: str = "noncentered",
                L = None):
        super(HSGPExpQuadWithDerivative, self).__init__()
        self._drop_first = drop_first
        self.parametrization = parametrization
        self.active_dims = active_dims
        self.n_dims = len(active_dims)
        self._m = m
        self._L = L
        self._c = c
        self._m_star = int(np.prod(m))
        self._beta = nn.Parameter(torch.FloatTensor(self._m_star,1))
        self._ls = nn.Parameter(torch.FloatTensor(self.n_dims))
        # initialize weights and biases
        nn.init.normal_(self._beta) # weight init
        nn.init.normal_(self._ls)

        self.ls = torch.exp(self._ls)

    
    def _slice(self, X, Xs=None):
        X = X[:, self.active_dims]
        if Xs is not None:
            Xs = Xs[:, self.active_dims]
        return X, Xs
    
    def prior_linearized(self, Xs):
        ### returns with derivative values 
        # Index Xs using input_dim and active_dims of covariance function
        self._X_mean = torch.mean(Xs, axis=0)
        Xs, _ = self._slice(Xs - self._X_mean)

        # If not provided, use Xs and c to set L
        if self._L is None:
            self.L = set_boundary(Xs, self._c)
        else:
            self.L = self._L

        eigvals = calc_eigenvalues(self.L, self._m)
        omega = torch.sqrt(eigvals)
        psd = self.power_spectral_density(omega).to(torch.float64)
        self._sqrt_psd = torch.sqrt(psd).to(torch.float64)
        self._eigvals = eigvals.to(torch.float64)

    def power_spectral_density_deriv(self, omega, deriv_dim):
        ls = torch.ones(self.n_dims) * self.ls
        return self.power_spectral_density(omega) * torch.prod(ls) * torch.square(omega[:,deriv_dim])

    def power_spectral_density(self, omega):
        r"""
        The power spectral density for the ExpQuad kernel is:

        .. math::

           S(\boldsymbol\omega) =
               (\sqrt(2 \pi)^D \prod_{i}^{D}\ell_i
                \exp\left( -\frac{1}{2} \sum_{i}^{D}\ell_i^2 \omega_i^{2} \right)
        """
        ls = torch.ones(self.n_dims) * self.ls
        c = torch.pow(torch.sqrt(2.0 * torch.Tensor([np.pi])), self.n_dims)
        exp = torch.exp(-0.5 * torch.matmul(torch.square(omega) , torch.square(ls[:,None])))
        return c * torch.prod(ls) * exp

    
    def predict(self, Xnew, deriv_dim : int | None = None):
        Xnew, _ = self._slice(Xnew - self._X_mean)
        if deriv_dim is None:
            phi = calc_eigenvectors(Xnew, self.L, self._eigvals, self._m)

            prediction = phi @ (self._beta * self._sqrt_psd)
            return torch.squeeze(prediction)
        else:
            phi_deriv = calc_eigenvectors_deriv(Xnew, self.L, self._eigvals, self._m, deriv_dim)
            prediction = phi_deriv @ (self._beta * self._sqrt_psd)
            return torch.squeeze(prediction)
    

        


    

