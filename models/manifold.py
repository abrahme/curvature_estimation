import numpy as np
from scipy.linalg.lapack import dtrtri
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.base import VectorSpace
from geomstats.numerics.geodesic import ExpODESolver, LogODESolver
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
    array = np.zeros((dim_size, dim_size))
    indices = np.tril_indices(n = dim_size, m = dim_size, k = 0)
    array[indices] = vector
    return array

def make_diagonal_positive(array):
    n = array.shape[0]
    array[np.diag_indices(n)] = np.exp(.5*np.diagonal(array))
    return array


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
        return np.zeros(self.dim)

    def _create_basis(self):
        """Create the canonical basis."""
        return np.eye(self.dim)



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
        
        # self.scale = scale

    def _make_chol(self,array):
        return np.stack([make_diagonal_positive(vector_to_lower_triangular(array[i], self.dimension)) for i in range(array.shape[0])], axis = 0)
    
    def _construct_gp_evaluation(self, base_point):
        return np.stack([gp.predict(base_point) 
                         for gp in self.gaussian_processes], axis = 0).T
    
    def metric_matrix(self, base_point=None):
        ### note that this returns a N x k matrix for k gps
        gp_evaluation = self._construct_gp_evaluation(base_point)
        gp_eval_chol = self._make_chol(gp_evaluation)

        ### now this returns a N x 2 x 2 tensor 
        
        ### now we have to take the dot product

        gp_eval_g = np.einsum("...jk,...mk -> ...jm" , gp_eval_chol, gp_eval_chol)
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
        gp_derivative_evaluation = np.stack(
            [np.stack([gp.predict(base_point,deriv_dim = i) for gp in self.gaussian_processes],
                      axis = 0).T for i in range(n_dims)], 
                      axis = -1) ### N x dim*(dim+1)/2 x n_dims tensor

        gp_derivative_evaluation_lower_tri = np.stack([np.stack([vector_to_lower_triangular(gp_derivative_evaluation[i,:,j], n_dims) for i in range(gp_derivative_evaluation.shape[0])], axis = 0) 
                                                       for j in range(n_dims)], axis = -1)  ### N x n_dim x n_dim x n_dim (last dim is partial deriv)

        gp_evaluation = self._construct_gp_evaluation(base_point)



        
        
        gp_eval_chol = self._make_chol(gp_evaluation) ## N x n_dims x n_dims

        gp_eval_chol_diagonals = np.stack([.5*np.diagonal(gp_eval_chol[i]) for i in range(gp_eval_chol.shape[0])], axis = 0) ### N x n_dims


        gp_derivative_evaluation_diagonals =np.stack([np.diagonal(gp_derivative_evaluation_lower_tri[i], axis1=0, axis2=-1) for i in range(gp_eval_chol.shape[0])], axis = 0) ### N x n_dims x n_dims

        
        partial_diagonals = gp_eval_chol_diagonals[:,:,np.newaxis] * gp_derivative_evaluation_diagonals

        # gp_derivative_evaluation_lower_tri[:,np.diag_indices_from(gp_derivative_evaluation_lower_tri)] = partial_diagonals

        for dim in range(n_dims):
            gp_derivative_evaluation_lower_tri[:,*np.diag_indices(n_dims),dim] = partial_diagonals[:,:,dim]

        partials = np.einsum("nij, nljk -> nilk", gp_eval_chol, gp_derivative_evaluation_lower_tri) + np.einsum("nijk, nlj -> nilk", gp_derivative_evaluation_lower_tri, gp_eval_chol)

        return partials

    def cometric_matrix(self, base_point=None):
        
        gp_evaluation = self._construct_gp_evaluation(base_point)
        gp_eval_chol = self._make_chol(gp_evaluation)


        chol_inv = np.stack([dtrtri(gp_eval_chol[i], lower = 1)[0] for i in range(gp_evaluation.shape[0])], axis = 0)
        return np.einsum("njl,nij -> nli", chol_inv, chol_inv)



    



    


        
        

