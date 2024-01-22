import torch
from torchdiffeq import odeint_adjoint as odeint
from typing import List
import torch.nn as nn
from .hsgp import HSGPExpQuadWithDerivative
from .manifold import GPRiemannianEuclidean



class _ODEFunc(nn.Module):
    def __init__(self, module: GPRiemannianEuclidean):
        super().__init__()
        self.module = module

    def forward(self, x, t):

        return self.module.metric.geodesic_equation(x, t)


class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = 'dopri5',
                 rtol: float = 1e-4, atol: float = 1e-4):
        super().__init__()
        self.odefunc = _ODEFunc(odefunc)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver

 

    def forward(self, X: torch.Tensor, integration_time):
     
        out = odeint(
            self.odefunc, X, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver)
        return out

class RiemannianAutoencoder(nn.Module):

    def __init__(self, n: int,t: int,  m: List[int], c: List, regularization: float, basis):
        super().__init__()

        d = int(n*(n-1)/2)
        self.gp_components = [HSGPExpQuadWithDerivative(m[d], c[d]) for _ in range(d)]
        for gp_component in self.gp_components:
            gp_component.prior_linearized(basis) ### initialize basis
        self.metric_space = GPRiemannianEuclidean(n, self.gp_components)
        self.ode_layer = ODEBlock(_ODEFunc(GPRiemannianEuclidean))
        self.n = n ### dimension of manifold
        self.d = d ### number of free functions
        self.t = t ### timepoints to extend
        self.regularization = regularization
        self.basis = basis

    def forward(self,initial_conditions):

        time_steps = torch.arange(0,1,self.t)
        predicted_vals = self.ode_layer.forward(initial_conditions, time_steps)
        return predicted_vals
    
    def loss(self, predicted_vals, actual_vals):

        reconstruction_loss =  nn.MSELoss(reduce=True, reduction="sum")(predicted_vals, actual_vals) 
        parameter_loss = 0
        for gp in self.gp_components:
            parameter_loss += nn.MSELoss(reduce=True, reduction="sum")(gp._beta, 0)
            parameter_loss += torch.sum(gp.ls)
        
        return reconstruction_loss + parameter_loss + self.regularization*self.prior_loss()
    
    def prior_loss(self):
        #### lie derivative loss with symmetry of circle
        ### TODO generalize to other symmetries 
        christoffels = self.metric_space.metric.christoffels(self.basis)
        return nn.MSELoss(reduce=True, reduction="sum")(self.basis[:,0]*(christoffels[:,1,:,:].sum(axis=(-1,-2))), self.basis[:,1]*(christoffels[:,0,:,:].sum(axis=(-1,-2))))




