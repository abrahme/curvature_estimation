import numpy as np
import torch
import torch.nn as nn
import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from typing import List
from .hsgp import HSGPExpQuadWithDerivative
from geomstats.numerics.optimizers import ScipyMinimize
from geomstats.numerics.geodesic import _LogShootingSolverFlatten, ExpODESolver

class MyBVPSolver(_LogShootingSolverFlatten):
    def __init__(self, optimizer=None, initialization=None):
        super().__init__(optimizer, initialization)
        self.optimizer = ScipyMinimize()





def vector_to_lower_triangular(vector, dim_size: int):
    """ converts a vector to a lower triangular matrix

    Args:
        vector (np.ndarray): _description_
        dim_size (int): _description_

    Returns:
        _type_: _description_
    """
    array = torch.zeros(dim_size, dim_size)
    indices = torch.tril_indices(row = dim_size, col = dim_size, offset = 0)
    array[indices.tolist()] = vector
    return array

def make_diagonal_positive(array):
    v = torch.exp(.5*torch.diagonal(array))
    mask = torch.diag(torch.ones_like(v))
    result =  mask*torch.diag(v) + (1-mask)*array
    return result




class GPRiemannianEuclidean(VectorSpace):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, dim, gps, equip=True):
        self.gps = gps
        super().__init__(
            dim=dim,
            shape=(dim,),
            equip=equip,
        )
        

    def equip_with_metric(self, Metric=None, **metric_kwargs):

        if Metric is None:
            Metric = GaussianProcessRiemmanianMetric
            self.metric = Metric(self, self.gps)
        else:
            self.metric = Metric(self, **metric_kwargs)
        return self

    @property
    def identity(self):
        """Identity of the group.

        Returns
        -------
        identity : array-like, shape=[n]
        """
        return torch.zeros(self.dim)

    def _create_basis(self):
        """Create the canonical basis."""
        return torch.eye(self.dim)







class GaussianProcessRiemmanianMetric(nn.Module):

    """ Riemannian Metric Subclass where the metric tensor elements are
        parametrized
        by an underlying spatial process
    """

    def __init__(self, dim, gaussian_processes: List[HSGPExpQuadWithDerivative]):
        super(GaussianProcessRiemmanianMetric, self).__init__()
        self.gaussian_processes = gaussian_processes
        self.dimension = dim
        

    def _make_chol(self,array):
        return torch.stack([make_diagonal_positive(vector_to_lower_triangular(array[i,:], self.dimension)) for i in range(array.shape[0])], axis = 0)
    
    def _construct_gp_evaluation(self, base_point):
        return torch.stack([gp.predict(base_point) 
                         for gp in self.gaussian_processes], axis = 0).T
    
    def metric_matrix(self, base_point=None):
        ### note that this returns a N x k matrix for k gps
        gp_evaluation = self._construct_gp_evaluation(base_point)
        gp_eval_chol = self._make_chol(gp_evaluation)

        ### now this returns a N x 2 x 2 tensor 
        
        ### now we have to take the dot product

        gp_eval_g = torch.einsum("...jk,...mk -> ...jm" , gp_eval_chol, gp_eval_chol)
        return gp_eval_g
        
    def inner_product_derivative_matrix(self, base_point=None):
        r"""Compute derivative of the inner prod matrix at base point.

        Writing :math:`g_{ij}` the inner-product matrix at base point,
        this computes :math:`mat_{ijk} = \partial_k g_{ij}`, where the
        index k of the derivation is put last.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        metric_derivative : array-like, shape=[..., dim, dim, dim]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: math:`mat_{ijk} = \partial_k g_{ij}`.
            Note that we don't need to use autodiff with the gaussian process
            as the derivatives can be explicit
            but maybe we have to because we don't wantto hand write derivatives 
        """
        n_dims = self.dimension 
        gp_derivative_evaluation = torch.stack(
            [torch.stack([gp.predict(base_point,deriv_dim = i) for gp in self.gaussian_processes],
                      axis = 0).T for i in range(n_dims)], 
                      axis = -1) ### N x dim*(dim+1)/2 x n_dims tensor

        gp_derivative_evaluation_lower_tri = torch.stack([torch.stack([vector_to_lower_triangular(gp_derivative_evaluation[i,:,j], n_dims) for i in range(gp_derivative_evaluation.shape[0])], axis = 0) 
                                                       for j in range(n_dims)], axis = -1)  ### N x n_dim x n_dim x n_dim (last dim is partial deriv)

        gp_evaluation = self._construct_gp_evaluation(base_point)
        
        gp_eval_chol = self._make_chol(gp_evaluation) ## N x n_dims x n_dims

        gp_eval_chol_diagonals = torch.stack([.5*torch.diagonal(gp_eval_chol[i]) for i in range(gp_eval_chol.shape[0])], axis = 0) ### N x n_dims


        gp_derivative_evaluation_diagonals = torch.stack([torch.diagonal(gp_derivative_evaluation_lower_tri[i], dim1=0, dim2=-1) for i in range(gp_eval_chol.shape[0])], axis = 0) ### N x n_dims x n_dims

        
        partial_diagonals = gp_eval_chol_diagonals[:,:,None] * gp_derivative_evaluation_diagonals

        # gp_derivative_evaluation_lower_tri[:,np.diag_indices_from(gp_derivative_evaluation_lower_tri)] = partial_diagonals

        for dim in range(n_dims):
            gp_derivative_evaluation_lower_tri[:,*np.diag_indices(n_dims),dim] = partial_diagonals[:,:,dim]

        partials = torch.einsum("nij, nljk -> nilk", gp_eval_chol, gp_derivative_evaluation_lower_tri) + torch.einsum("nijk, nlj -> nilk", gp_derivative_evaluation_lower_tri, gp_eval_chol)
        return partials

    def cometric_matrix(self, base_point=None):
        
        gp_evaluation = self._construct_gp_evaluation(base_point)
        gp_eval_chol = self._make_chol(gp_evaluation)
        batch_size = gp_evaluation.shape[0]
        n_dims = self.dimension
        B = torch.eye(n_dims).unsqueeze(0).expand(batch_size, -1, -1)
        chol_inv = torch.linalg.solve_triangular(gp_eval_chol, B, upper = False)
        return torch.einsum("njl,nij -> nli", chol_inv, chol_inv)
    
    def christoffels(self, base_point):
        r"""Compute Christoffel symbols of the Levi-Civita connection.

        The Koszul formula defining the Levi-Civita connection gives the
        expression of the Christoffel symbols with respect to the metric:
        :math:`\Gamma^k_{ij}(p) = \frac{1}{2} g^{lk}(
        \partial_i g_{jl} + \partial_j g_{li} - \partial_l g_{ij})`,
        where:

        - :math:`p` represents the base point, and
        - :math:`g` represents the Riemannian metric tensor.

        Note that the function computing the derivative of the metric matrix
        puts the index of the derivation last.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels: array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, where the contravariant index is first.
        """
        cometric_mat_at_point = self.cometric_matrix(base_point)
        metric_derivative_at_point = self.inner_product_derivative_matrix(base_point)

        term_1 = torch.einsum(
            "...lk,...jli->...kij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_2 = torch.einsum(
            "...lk,...lij->...kij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_3 = -torch.einsum(
            "...lk,...ijl->...kij", cometric_mat_at_point, metric_derivative_at_point
        )

        return 0.5 * (term_1 + term_2 + term_3)
    

    def geodesic_equation(self, state, _time):
        """Compute the geodesic ODE associated with the connection.

        Parameters
        ----------
        state : array-like, shape=[..., dim]
            Tangent vector at the position.
        _time : array-like, shape=[..., dim]
            Point on the manifold, the position at which to compute the
            geodesic ODE.

        Returns
        -------
        geodesic_ode : array-like, shape=[..., dim]
            Value of the vector field to be integrated at position.
        """
        position, velocity = state
        gamma = self.christoffels(position)
        equation = torch.einsum("...kij,...i->...kj", gamma, velocity)
        equation = -torch.einsum("...kj,...j->...k", equation, velocity)
        return torch.hstack([velocity, equation])
    
    def riemann_tensor(self, base_point):
        r"""Compute Riemannian tensor at base_point.

        In the literature the Riemannian curvature tensor is noted :math:`R_{ijk}^l`.

        Following tensor index convention (ref. Wikipedia), we have:
        :math:`R_{ijk}^l = dx^l(R(X_j, X_k)X_i)`

        which gives :math:`R_{ijk}^l` as a sum of four terms:

        .. math::
            \partial_j \Gamma^l_{ki} - \partial_k \Gamma^l_{ji}
            + \Gamma^l_{jm} \Gamma^m_{ki} - \Gamma^l_{km} \Gamma^m_{ji}

        Note that geomstats puts the contravariant index on
        the first dimension of the Christoffel symbols.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        riemann_curvature : array-like, shape=[..., dim, dim, dim, dim]
            riemann_tensor[...,i,j,k,l] = R_{ijk}^l
            Riemannian tensor curvature,
            with the contravariant index on the last dimension.
        """
        christoffels = self.christoffels(base_point)
        jacobian_christoffels = gs.autodiff.jacobian_vec(self.christoffels)(base_point)

        prod_christoffels = torch.einsum(
            "...ijk,...klm->...ijlm", christoffels, christoffels
        )
        riemann_curvature = (
            torch.einsum("...ijlm->...lmji", jacobian_christoffels)
            - torch.einsum("...ijlm->...ljmi", jacobian_christoffels)
            + torch.einsum("...ijlm->...mjli", prod_christoffels)
            - torch.einsum("...ijlm->...lmji", prod_christoffels)
        )

        return riemann_curvature

    def ricci_tensor(self, base_point):
        r"""Compute Ricci curvature tensor at base_point.

        The Ricci curvature tensor :math:`\mathrm{Ric}_{ij}` is defined as:
        :math:`\mathrm{Ric}_{ij} = R_{ikj}^k`
        with Einstein notation.

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        ricci_tensor : array-like, shape=[..., dim, dim]
            ricci_tensor[...,i,j] = Ric_{ij}
            Ricci tensor curvature.
        """
        riemann_tensor = self.riemann_tensor(base_point)
        ricci_tensor = torch.einsum("...ijkj -> ...ik", riemann_tensor)
        return ricci_tensor
    
    def ricci_scalar(self, base_point):
        ricc_tensor = self.ricci_tensor(base_point)
        return torch.trace(ricc_tensor, dim1=-2, dim2=-1)

    
class GaussianProcessRiemmanianMetricSymmetricCircle(GaussianProcessRiemmanianMetric):
    def __init__(self, dim, gaussian_processes: List[HSGPExpQuadWithDerivative]):
        super(GaussianProcessRiemmanianMetricSymmetricCircle, self).__init__(dim, gaussian_processes)

    def christoffels(self, base_point):
        cometric_mat_at_point = self.cometric_matrix(base_point)

        metric_derivative_at_point = self.inner_product_derivative_matrix(base_point)
        ratio = (base_point[:,1]).view(-1, 1, 1, 1) ### modulating factor 
        base_dim = 0 ### dimension to relate to others to 
        cometric_base = cometric_mat_at_point[:,:,base_dim][:,:, None]
        term_1_base = torch.einsum(
            "...lk,...jli->...kij",cometric_base , metric_derivative_at_point
        )
        term_2_base = torch.einsum(
            "...lk,...lij->...kij", cometric_base, metric_derivative_at_point
        )
        term_3_base = -torch.einsum(
            "...lk,...ijl->...kij", cometric_base, metric_derivative_at_point
        )

        christoffel_base = 0.5 *(term_1_base + term_2_base + term_3_base)

        christoffel_transformed = ratio * christoffel_base
        result = torch.cat([christoffel_base, christoffel_transformed], axis = 1)
        return result

    def geodesic_equation(self, state, _time):
        """Compute the geodesic ODE associated with the connection.

        Parameters
        ----------
        state : array-like, shape=[..., dim]
            Tangent vector at the position.
        _time : array-like, shape=[..., dim]
            Point on the manifold, the position at which to compute the
            geodesic ODE.

        Returns
        -------
        geodesic_ode : array-like, shape=[..., dim]
            Value of the vector field to be integrated at position.
        """
        position, velocity = state
        gamma = self.christoffels(position)
        base_dim = 0
        augmented_acceleration = torch.cat([torch.ones_like(position[:,base_dim][:, None]), position[:,base_dim][:, None]], axis = -1)

        equation = torch.einsum("...kij,...i->...kj", gamma, velocity) 
        equation = -torch.einsum("...kj,...j->...k", equation, velocity) * augmented_acceleration
        return torch.hstack([velocity, equation])


class GaussianProcessRiemmanianMetricSymmetricSphere(GaussianProcessRiemmanianMetric):
    def __init__(self, dim, gaussian_processes: List[HSGPExpQuadWithDerivative]):
        super(GaussianProcessRiemmanianMetricSymmetricSphere, self).__init__(dim, gaussian_processes)

    def christoffels(self, base_point):
        cometric_mat_at_point = self.cometric_matrix(base_point)
        base_dim = 0 ### dimension to relate to others to 
        metric_derivative_at_point = self.inner_product_derivative_matrix(base_point)

        denominator = (base_point[:,1]).view(-1,1,1,1) ### modulating factor 
        numerator =  (base_point[:,base_dim]).view(-1,1,1,1) ### modulating factor 

        cometric_base = cometric_mat_at_point[:,:,base_dim][:,:, None]
        term_1_base = torch.einsum(
            "...lk,...jli->...kij",cometric_base , metric_derivative_at_point
        )
        term_2_base = torch.einsum(
            "...lk,...lij->...kij", cometric_base, metric_derivative_at_point
        )
        term_3_base = -torch.einsum(
            "...lk,...ijl->...kij", cometric_base, metric_derivative_at_point
        )

        christoffel_base = 0.5 * (term_1_base + term_2_base + term_3_base)

        cometric_untouched = cometric_mat_at_point[:,:,-1][:,:, None]
        term_1_untouched = torch.einsum(
            "...lk,...jli->...kij",cometric_untouched , metric_derivative_at_point
        )
        term_2_untouched = torch.einsum(
            "...lk,...lij->...kij", cometric_untouched, metric_derivative_at_point
        )
        term_3_untouched = -torch.einsum(
            "...lk,...ijl->...kij", cometric_untouched, metric_derivative_at_point
        )

        christoffel_untouched = 0.5 * (term_1_untouched + term_2_untouched + term_3_untouched)

        cometric_transformed = cometric_mat_at_point[:,:,1][:,:, None]

        ### ...3,1, ...1,3,3 -> ...1,1,3
        
        term_1_transformed = torch.einsum(
            "...lk,...jli->...kij",cometric_transformed , metric_derivative_at_point[:,-1,:,:][:,None,:,:]
        )

        term_2_transformed = torch.einsum(
            "...lk,...lij->...kij", cometric_transformed, metric_derivative_at_point[:,:,:,-1][:,:,:,None]
        )
        term_3_transformed = -torch.einsum(
            "...lk,...ijl->...kij", cometric_transformed, metric_derivative_at_point[:,:,-1,:][:,:,None,:]
        )

        christoffel_transformed  =  torch.permute((term_1_transformed + term_2_transformed + term_3_transformed)  * numerator * .05, (0,1,3,2))

        
        ### the above is a tensor of shape (N, 1, 1, 3) where the second index is the  contravariant index


        christoffel_transformed_y = christoffel_base[:,:,base_dim:2,base_dim:2] * denominator


        final_christoffel_matrix = torch.zeros_like(christoffel_untouched)
        final_christoffel_matrix[:,:,base_dim:2, base_dim:2] = christoffel_transformed_y[:,:,base_dim:2, base_dim:2]
        final_christoffel_matrix[:,:,-1,:] = christoffel_transformed[:,:,0,:]
        final_christoffel_matrix[:,:,:,base_dim:2] = christoffel_transformed[:,:,0,base_dim:2][:,:,None,:]


        ### the above is a tensor of shape (N, 1, 3, 3) where the second index is the  contravariant indices, 
        #### 

        result = torch.cat([christoffel_base, final_christoffel_matrix, christoffel_untouched], dim = 1)
        
        return result

    def geodesic_equation(self, state, _time):
        """Compute the geodesic ODE associated with the connection.

        Parameters
        ----------
        state : array-like, shape=[..., dim]
            Tangent vector at the position.
        _time : array-like, shape=[..., dim]
            Point on the manifold, the position at which to compute the
            geodesic ODE.

        Returns
        -------
        geodesic_ode : array-like, shape=[..., dim]
            Value of the vector field to be integrated at position.
        """
        position, velocity = state
        gamma = self.christoffels(position)
        base_dim = 0
        augmented_acceleration = torch.cat([torch.ones_like(position[:,base_dim][:, None]), position[:,base_dim][:, None], torch.ones_like(position[:,base_dim][:, None])], axis = -1)
        equation = torch.einsum("...kij,...i->...kj", gamma, velocity) 
        equation = -torch.einsum("...kj,...j->...k", equation, velocity) * augmented_acceleration
        return torch.hstack([velocity, equation])




