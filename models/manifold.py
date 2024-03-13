
import torch
import torch.nn as nn
from torch.autograd import functional
import emlp.nn.pytorch as nn_emlp
from .neural import PSD, PSDGP, PSDGroup



class GPRiemmanianMetric(nn.Module):

    """ Riemannian Metric Subclass where the metric tensor elements are
        parametrized
        by an underlying neural network
    """

    def __init__(self, dim, metric_func: PSDGP):
        super(GPRiemmanianMetric, self).__init__()
        self.metric_function = metric_func
        self.dimension = dim
    
    def metric_matrix(self, base_point=None):
        ### note that this returns a N x k matrix for k gps
        return self.metric_function(base_point)
        
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

        """
        sum_func = lambda x: self.metric_function(x).sum(axis = 0)
        partials = functional.jacobian(sum_func, base_point, create_graph=True)
        return torch.swapaxes(partials, 2,0)

    def cometric_matrix(self, base_point=None):
        return torch.linalg.inv(self.metric_matrix(base_point))
    
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

    
    
class GroupNeuralRiemmanianMetric(nn.Module):

    """ Riemannian Metric Subclass where the metric tensor elements are
        parametrized
        by an underlying neural network
    """

    def __init__(self, dim, metric_func: PSDGroup):
        super(GroupNeuralRiemmanianMetric, self).__init__()
        self.metric_function = metric_func
        self.dimension = dim
    
    def metric_matrix(self, base_point=None):
        ### note that this returns a N x k matrix for k gps
        return self.metric_function(base_point)
        
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

        """
        # with torch.set_grad_enabled(True):
        #     base_point = base_point.requires_grad_(True)
        #     partials = functional.jacobian(self.metric_matrix, base_point, create_graph=True)
        sum_func = lambda x: self.metric_function(x).sum(axis = 0)
        partials = functional.jacobian(sum_func, base_point, create_graph=True)
        return torch.swapaxes(partials, 2,0)

    def cometric_matrix(self, base_point=None):
        return torch.linalg.inv(self.metric_matrix(base_point))
    
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


class NeuralRiemmanianMetric(nn.Module):

    """ Riemannian Metric Subclass where the metric tensor elements are
        parametrized
        by an underlying neural network
    """

    def __init__(self, dim, metric_func: PSD):
        super(NeuralRiemmanianMetric, self).__init__()
        self.metric_function = metric_func
        self.dimension = dim
    
    def metric_matrix(self, base_point=None):
        ### note that this returns a N x k matrix for k gps
        return self.metric_function(base_point)
        
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

        """
        # with torch.set_grad_enabled(True):
        #     base_point = base_point.requires_grad_(True)
        #     partials = functional.jacobian(self.metric_matrix, base_point, create_graph=True)
        sum_func = lambda x: self.metric_function(x).sum(axis = 0)
        partials = functional.jacobian(sum_func, base_point, create_graph=True)
        return torch.swapaxes(partials, 2,0)

    def cometric_matrix(self, base_point=None):
        return torch.linalg.inv(self.metric_matrix(base_point))
    
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
    
    

    





