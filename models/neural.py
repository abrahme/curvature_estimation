import torch 
import torch.nn as nn
import numpy as np

pi = torch.FloatTensor([np.pi])

def set_boundary(Xs , c):
    """Set the boundary using the mean-subtracted `Xs` and `c`.  `c` is usually a scalar
    multiplyer greater than 1.0, but it may be one value per dimension or column of `Xs`.
    """
    S, _ = torch.max(torch.abs(Xs),0)
    L = c * S
    return L


def calc_eigenvalues(L, m):
    """Calculate eigenvalues of the Laplacian."""
    S = torch.meshgrid(*[torch.arange(1, 1 + m[d], dtype=torch.float32) for d in range(len(m))])
    S_arr = torch.vstack([torch.flatten(s) for s in S]).T
    return torch.square(( pi * S_arr) / (2 * L))


def calc_eigenvectors(
    Xs,
    L,
    eigvals,
    m):
    """Calculate eigenvectors of the Laplacian. These are used as basis vectors in the HSGP
    approximation.
    """
    m_star = int(np.prod(m))
    c = 1.0 / torch.sqrt(L)
    term1 = torch.sqrt(eigvals)
    term2 = torch.repeat_interleave(Xs[:, None, :], m_star, 1) + L
    sin_term = torch.sin(term1 * term2) * c
    return torch.prod(sin_term, dim = -1)

class HSGPExpQuadWithDerivative(nn.Module):
    def __init__(self, m, c, active_dims, output_dim, basis,
                drop_first: bool = False, parametrization: str = "noncentered",
                L = None):
        super(HSGPExpQuadWithDerivative, self).__init__()
        self._drop_first = drop_first
        self.parametrization = parametrization
        self.active_dims = active_dims
        self.n_dims = active_dims
        self._m = m
        self._L = L
        self._c = torch.FloatTensor([c])
        self._m_star = int(np.prod(m))
        self._beta = nn.Linear(bias = False, in_features=self._m_star, out_features=output_dim)
        self.basis = basis
        # initialize weights and biases
        nn.init.normal_(self._beta.weight) # weight init

        self.ls = nn.Parameter(torch.ones((active_dims,1)))
        self._prior_linearized(basis)
    
    def _slice(self, X, Xs=None):
        X = X[..., :self.active_dims]
        if Xs is not None:
            Xs = Xs[..., :self.active_dims]
        return X, Xs
    
    def _prior_linearized(self, Xs):
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
        self._eigvals = eigvals
        self._omega = omega

    def power_spectral_density(self, omega):
        r"""
        The power spectral density for the ExpQuad kernel is:

        .. math::

           S(\boldsymbol\omega) =
               (\sqrt(2 \pi)^D \prod_{i}^{D}\ell_i
                \exp\left( -\frac{1}{2} \sum_{i}^{D}\ell_i^2 \omega_i^{2} \right)
        """
        c = torch.pow(torch.sqrt(2.0 * pi), self.n_dims)
        exp = torch.exp(-0.5 * torch.matmul(torch.square(omega) , torch.square(torch.exp(self.ls))))
        return c * torch.prod(torch.exp(self.ls)) * exp

    
    def forward(self, Xnew):
        psd = self.power_spectral_density(self._omega)
        sqrt_psd = torch.sqrt(psd)
        phi = calc_eigenvectors(Xnew - self._X_mean, self.L, self._eigvals, self._m)
        prediction = self._beta(phi * sqrt_psd[:,0])
        return prediction 

    

     
class PSDGP(nn.Module):
    '''A GP which outputs a positive semi-definite matrix'''
    def __init__(self, input_dim, diag_dim, basis):
        assert diag_dim == input_dim
        super(PSDGP, self).__init__()
        self.diag_dim = diag_dim
        self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
        self.basis = basis
        self.GP = HSGPExpQuadWithDerivative(active_dims=input_dim, output_dim= diag_dim*(diag_dim+1)//2, basis = basis, m = [5]*diag_dim, c = 4.0)


    def forward(self, q):
        bs = q.shape[0]
        h = self.GP(q)
        diag, off_diag = torch.split(h, [self.diag_dim, self.off_diag_dim], dim=1)

        L = torch.diag_embed(nn.Softplus()(diag))

        ind = np.tril_indices(self.diag_dim, k=-1)
        flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = off_diag
        L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

        D = torch.bmm(L, L.permute(0, 2, 1))
        return D
    




class PSD(nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix'''
    def __init__(self, input_dim, hidden_dim, diag_dim):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

        for l in [self.linear1, self.linear2]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization
        
        self.nonlinearity = nn.Tanh()

    def forward(self, q):
        bs = q.shape[0]
        h = self.nonlinearity( self.linear1(q) )
        diag, off_diag = torch.split(self.linear2(h), [self.diag_dim, self.off_diag_dim], dim=1)

        L = torch.diag_embed(nn.Softplus()(diag))

        ind = np.tril_indices(self.diag_dim, k=-1)
        flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = off_diag
        L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

        D = torch.bmm(L, L.permute(0, 2, 1))
        return D