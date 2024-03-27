
import torch
import torch.nn as nn
from torch.autograd import functional
# import emlp.nn.pytorch as nn_emlp
from .neural import PSD, PSDGP



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

    
    
# class GroupNeuralRiemmanianMetric(nn.Module):

#     """ Riemannian Metric Subclass where the metric tensor elements are
#         parametrized
#         by an underlying neural network
#     """

#     def __init__(self, dim, metric_func: PSDGroup):
#         super(GroupNeuralRiemmanianMetric, self).__init__()
#         self.metric_function = metric_func
#         self.dimension = dim
    
#     def metric_matrix(self, base_point=None):
#         ### note that this returns a N x k matrix for k gps
#         return self.metric_function(base_point)
        
#     def inner_product_derivative_matrix(self, base_point=None):
#         r"""Compute derivative of the inner prod matrix at base point.

#         Writing :math:`g_{ij}` the inner-product matrix at base point,
#         this computes :math:`mat_{ijk} = \partial_k g_{ij}`, where the
#         index k of the derivation is put last.

#         Parameters
#         ----------
#         base_point : array-like, shape=[..., dim]
#             Base point.
#             Optional, default: None.

#         Returns
#         -------
#         metric_derivative : array-like, shape=[..., dim, dim, dim]
#             Derivative of the inner-product matrix, where the index
#             k of the derivation is last: math:`mat_{ijk} = \partial_k g_{ij}`.

#         """
#         # with torch.set_grad_enabled(True):
#         #     base_point = base_point.requires_grad_(True)
#         #     partials = functional.jacobian(self.metric_matrix, base_point, create_graph=True)
#         sum_func = lambda x: self.metric_function(x).sum(axis = 0)
#         partials = functional.jacobian(sum_func, base_point, create_graph=True)
#         return torch.swapaxes(partials, 2,0)

#     def cometric_matrix(self, base_point=None):
#         return torch.linalg.inv(self.metric_matrix(base_point))
    
#     def christoffels(self, base_point):
#         r"""Compute Christoffel symbols of the Levi-Civita connection.

#         The Koszul formula defining the Levi-Civita connection gives the
#         expression of the Christoffel symbols with respect to the metric:
#         :math:`\Gamma^k_{ij}(p) = \frac{1}{2} g^{lk}(
#         \partial_i g_{jl} + \partial_j g_{li} - \partial_l g_{ij})`,
#         where:

#         - :math:`p` represents the base point, and
#         - :math:`g` represents the Riemannian metric tensor.

#         Note that the function computing the derivative of the metric matrix
#         puts the index of the derivation last.

#         Parameters
#         ----------
#         base_point: array-like, shape=[..., dim]
#             Base point.

#         Returns
#         -------
#         christoffels: array-like, shape=[..., dim, dim, dim]
#             Christoffel symbols, where the contravariant index is first.
#         """
#         cometric_mat_at_point = self.cometric_matrix(base_point)
#         metric_derivative_at_point = self.inner_product_derivative_matrix(base_point)
#         term_1 = torch.einsum(
#             "...lk,...jli->...kij", cometric_mat_at_point, metric_derivative_at_point
#         )
#         term_2 = torch.einsum(
#             "...lk,...lij->...kij", cometric_mat_at_point, metric_derivative_at_point
#         )
#         term_3 = -torch.einsum(
#             "...lk,...ijl->...kij", cometric_mat_at_point, metric_derivative_at_point
#         )

#         return 0.5 * (term_1 + term_2 + term_3)


#     def geodesic_equation(self, state, _time):
#         """Compute the geodesic ODE associated with the connection.

#         Parameters
#         ----------
#         state : array-like, shape=[..., dim]
#             Tangent vector at the position.
#         _time : array-like, shape=[..., dim]
#             Point on the manifold, the position at which to compute the
#             geodesic ODE.

#         Returns
#         -------
#         geodesic_ode : array-like, shape=[..., dim]
#             Value of the vector field to be integrated at position.
#         """
#         position, velocity = state
#         gamma = self.christoffels(position)
#         equation = torch.einsum("...kij,...i->...kj", gamma, velocity)
#         equation = -torch.einsum("...kj,...j->...k", equation, velocity)
#         return torch.hstack([velocity, equation])


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

    def riemann_tensor(self, base_point=None):
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
        sum_func = lambda x: self.christoffels(x).sum(axis = 0)
        jacobian_christoffels = torch.swapaxes(functional.jacobian(sum_func, base_point), 0, 3)
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
    
    def ricci_tensor(self, base_point=None):
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

    def scalar_curvature(self, base_point=None):
        r"""Compute scalar curvature at base_point.

        In the literature scalar_curvature is noted S and writes:
        :math:`S = g^{ij} Ric_{ij}`,
        with Einstein notation, where we recognize the trace of the Ricci
        tensor according to the Riemannian metric :math:`g`.

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        curvature : array-like, shape=[...,]
            Scalar curvature.
        """
        ricci_tensor = self.ricci_tensor(base_point)
        cometric_matrix = self.cometric_matrix(base_point)
        return torch.einsum("...ij, ...ij -> ...", cometric_matrix, ricci_tensor)
    
    

    





