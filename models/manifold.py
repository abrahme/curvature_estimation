import geomstats as gm 
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.base import VectorSpace
from geomstats.numerics.geodesic import ExpODESolver, LogODESolver
import numpy as np
from typing import List
from .hsgp import HSGPExpQuadWithDerivative

def vector_to_lower_triangular(vector: np.ndarray, dim_size: int):
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

def compute_partial_derivatives(chol_value_array, partial_deriv_array, partial_dim: int = 0):
    partial_deriv_array_dim = partial_deriv_array[:,:,partial_dim]
    x = partial_deriv_array_dim[:,0] * np.square(chol_value_array[:,0,0])
    y = partial_deriv_array_dim[:,1] * chol_value_array[:,1,0] + .5 * chol_value_array[:,1,0] * partial_deriv_array_dim[:,0]*chol_value_array[:,1,0]
    z = x + 2*y*chol_value_array[:,1,0]*chol_value_array[:,0,0]
    return np.stack([np.stack([x, y], axis = -1),np.stack([y, z], axis = -1)], axis = -1)



class GPRiemannianEuclidean(VectorSpace):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, dim, gps, scale, equip=True):
        self.gps = gps
        self.scale = scale
        super().__init__(
            dim=dim,
            shape=(dim,),
            equip=equip,
        )
        

    def equip_with_metric(self, Metric=None, **metric_kwargs):

        if Metric is None:
            Metric = TwoDimensionalGaussianProcessRiemmanianMetric
            self.metric = Metric(self, self.gps, self.scale)
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



class TwoDimensionalGaussianProcessRiemmanianMetric(RiemannianMetric):

    """ Riemannian Metric Subclass where the metric tensor elements are
        parametrized
        by an underlying spatial process
    """

    def __init__(self, space, gaussian_processes: List[HSGPExpQuadWithDerivative], scale, signature=None):
        super().__init__(space, signature)
        self.gaussian_processes = gaussian_processes
        self.exp_solver = ExpODESolver()
        self.log_solver = LogODESolver()
        # self.scale = scale
    
    def metric_matrix(self, base_point=None):
        ### note that this returns a N x k matrix for k gps
        gp_evaluation = np.stack([gp.predict(base_point) 
                         for gp in self.gaussian_processes], axis = 0).T
        
        gp_eval_chol = np.stack([make_diagonal_positive(vector_to_lower_triangular(gp_evaluation[i], 2)) for i in range(gp_evaluation.shape[0])], axis = 0)
        ### now this returns a N x 2 x 2 tensor 
        
        ### now we have to take the dot product

        gp_eval_g = np.einsum("...jk,...mk -> ...jm" , gp_eval_chol, gp_eval_chol)
        # gp_eval_g = np.einsum("...jk,...mk -> ...jm" , gp_evaluation, gp_evaluation)
        # gp_eval_g += np.eye(2) * self.scale
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
        n_dims = 2 ### fixed for 2
        gp_derivative_evaluation = np.stack(
            [np.stack([gp.predict(base_point,deriv_dim = i) for gp in self.gaussian_processes],
                      axis = 0).T for i in range(n_dims)], 
                      axis = -1) ### N x 3 x 2 tensor

        # gp_derivative_evaluation = np.stack(
        #     [np.stack([gp.predict(base_point,deriv_dim = i) for gp in self.gaussian_processes],
        #               axis = 0).reshape((-1, n_dims, n_dims)) for i in range(n_dims)], 
        #               axis = -1) ### N x 2 x 2 x 2 tensor
        gp_evaluation = np.stack([gp.predict(base_point) 
                         for gp in self.gaussian_processes], axis = 0).T  ### N x 

        # gp_evaluation = np.stack([gp.predict(base_point) 
        #                  for gp in self.gaussian_processes], axis = 0).reshape((-1, n_dims, n_dims))  ### N x 3
        
        gp_eval_chol = np.stack([make_diagonal_positive(vector_to_lower_triangular(gp_evaluation[i], 2)) for i in range(gp_evaluation.shape[0])], axis = 0)
        ### N x 2 x 2

        ### some chain rule 
        partials = np.stack([compute_partial_derivatives(gp_eval_chol, gp_derivative_evaluation, i) for i in range(n_dims)], axis = -1)

        # partials = np.einsum("nij, nljk -> nilk", gp_evaluation, gp_derivative_evaluation) + np.einsum("nijk, nlj -> nilk", gp_derivative_evaluation, gp_evaluation)

        return partials

    def cometric_matrix(self, base_point=None):
        metric_matrix = self.metric_matrix(base_point)
        determinant = 1 /(metric_matrix[:,0,0]*metric_matrix[:,1,1] - np.square(metric_matrix[:,0,1]))

        x = metric_matrix[:,0,0] * determinant
        y = metric_matrix[:,0,1] * determinant
        z = metric_matrix[:,1,1] * determinant

        cometric_matrix = np.stack([np.stack([z, -y], axis = -1),np.stack([-y, x], axis = -1)], axis = -1)

        return cometric_matrix



    



    


        
        

