import numpy as np
import torch
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.base import VectorSpace
from .ode import TorchExpODESolver
from typing import List
from .hsgp import HSGPExpQuadWithDerivative

def vector_to_lower_triangular(vector, dim_size: int):
    """ converts a vector to a lower triangular matrix

    Args:
        vector (np.ndarray): _description_
        dim_size (int): _description_

    Returns:
        _type_: _description_
    """
    array = torch.zeros((dim_size, dim_size))
    indices = torch.tril_indices(n = dim_size, m = dim_size, k = 0)
    array[indices] = vector
    return array

def make_diagonal_positive(array):
    v = torch.exp(.5*torch.diagonal(array))
    mask = torch.diag(torch.ones_like(v))
    return mask*torch.diag(v) + (1-mask)*array


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



class GaussianProcessRiemmanianMetric(RiemannianMetric):

    """ Riemannian Metric Subclass where the metric tensor elements are
        parametrized
        by an underlying spatial process
    """

    def __init__(self, space, gaussian_processes: List[HSGPExpQuadWithDerivative], signature=None):
        super().__init__(space, signature)
        self.gaussian_processes = gaussian_processes
        self.exp_solver = ExpODESolver()
        self.log_solver = LogODESolver()
        self.dimension = space.dim
        

    def _make_chol(self,array):
        return torch.stack([make_diagonal_positive(vector_to_lower_triangular(array[i], self.dimension)) for i in range(array.shape[0])], axis = 0)
    
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


        gp_derivative_evaluation_diagonals = torch.stack([torch.diagonal(gp_derivative_evaluation_lower_tri[i], axis1=0, axis2=-1) for i in range(gp_eval_chol.shape[0])], axis = 0) ### N x n_dims x n_dims

        
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
        n_dims = self.space.dim
        B = torch.eye(n_dims).unsqueeze(0).expand(batch_size, -1, -1)
        chol_inv = torch.triangular_solve(gp_eval_chol, B, upper = False)
        return torch.einsum("njl,nij -> nli", chol_inv, chol_inv)



    



    


        
        

