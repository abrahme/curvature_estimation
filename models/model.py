import torch
# from torchdiffeq import odeint as odeint
from geomloss.samples_loss import SamplesLoss
from torchdyn.core import NeuralODE
from typing import List
import torch.nn as nn
from .hsgp import HSGPExpQuadWithDerivative
from .manifold import GaussianProcessRiemmanianMetric



class _ODEFunc(nn.Module):
    def __init__(self, module: GaussianProcessRiemmanianMetric):
        super(_ODEFunc, self).__init__()
        self.module = module
    def forward(self, t, x, *args, **kwargs):
        ode_eq = self.module.geodesic_equation(torch.hsplit(x,2), t)
        return ode_eq


class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = 'dopri5',
                 sensitivity:str = 'autograd',
                 rtol: float = 1e-4, atol: float = 1e-4):
        super(ODEBlock, self).__init__()
        self.odefunc = _ODEFunc(odefunc)
        self.ode_solver = NeuralODE(self.odefunc, solver = solver, 
                                    sensitivity=sensitivity, atol=atol, rtol=rtol)
    
    def forward(self, X: torch.Tensor, integration_time):

        t_eval, out = self.ode_solver(X, integration_time)
        return t_eval, out

class RiemannianAutoencoder(nn.Module):

    def __init__(self, n: int,t: int,  m: List[int], c: float, regularization: float, basis, active_dims: List, loss_type: str = "L2"):
        super(RiemannianAutoencoder, self).__init__()
        d = int(n*(n+1)/2)
        self.gp_components = nn.ModuleList([HSGPExpQuadWithDerivative(m, c, active_dims) for _ in range(d)])
        for gp_component in self.gp_components:
            gp_component.prior_linearized(basis) ### initialize basis
        self.metric_space = GaussianProcessRiemmanianMetric(n, self.gp_components)
        self.ode_layer = ODEBlock(self.metric_space)
        self.n = n ### dimension of manifold
        self.d = d ### number of free functions
        self.t = t ### timepoints to extend
        self.regularization = regularization
        self.basis = basis
        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        _, predicted_vals = self.ode_layer(initial_conditions, time_steps)
        return torch.split(predicted_vals,split_size_or_sections=2,dim=2)[0]

    
    def loss(self, predicted_vals, actual_vals, val = False):
        if self.loss_type == "L2":
            reconstruction_loss = nn.MSELoss()(predicted_vals,actual_vals)
        elif self.loss_type == "Hausdorff":
            reconstruction_loss = torch.FloatTensor([0.0])
            for i in range(predicted_vals.shape[0]):
                reconstruction_loss += SamplesLoss("sinkhorn")(predicted_vals[i], actual_vals[i])
        if not val:
            reconstruction_loss = reconstruction_loss + self.parameter_loss()
        
        if self.regularization > 0:
            if not val:
                reconstruction_loss = self.prior_loss() + reconstruction_loss
        return reconstruction_loss
    
    def parameter_loss(self):
        parameter_loss = torch.FloatTensor([0.0])
        for gp in self.gp_components:
            parameter_loss += torch.square(gp._beta).mean()/len(self.gp_components)
            parameter_loss += torch.mean(gp.ls)/len(self.gp_components)
        return parameter_loss

    
    def prior_loss(self):
        #### lie derivative loss with symmetry of circle
        ### TODO generalize to other symmetries 
        christoffels = self.metric_space.christoffels(self.basis)
        prior_loss = torch.FloatTensor([self.regularization]) * torch.square(self.basis[:,0]*(christoffels[:,1,:,:].sum((-1,-2))) - self.basis[:,1]*(christoffels[:,0,:,:].sum((-1,-2)))).mean()
        return prior_loss










