import torch
from torchdyn.core import NeuralODE
import torch.nn as nn
from .manifold import NeuralRiemmanianMetric, GPRiemmanianMetric
from .neural import PSD, PSDGP
from typing import Union






class ODEFunc(nn.Module):
    def __init__(self, module:Union[NeuralRiemmanianMetric, GPRiemmanianMetric]):
        super(ODEFunc, self).__init__()
        self.module = module
    def forward(self, t, x, *args, **kwargs):
        ode_eq = self.module.geodesic_equation(torch.hsplit(x,2), t)
        return ode_eq

class NNODE(nn.Module):
    def __init__(self, odefunc: ODEFunc, sensitivity="adjoint", solver="dopri5", atol=1e-3, rtol=1e-3) -> None:
        super(NNODE, self).__init__()
        self.odefunc = odefunc
        self.neural_ode_layer = NeuralODE(self.odefunc, solver = solver, sensitivity=sensitivity, atol=atol, rtol=rtol)
    def forward(self, x, t, *args, **kwargs):
        return self.neural_ode_layer.forward(x, t)

class RiemannianAutoencoder(nn.Module):

    def __init__(self, n: int, hidden_dim: int,  t: int, loss_type: str = "L2"):
        super(RiemannianAutoencoder, self).__init__()

        self.metric_space = NeuralRiemmanianMetric(dim = n, metric_func=PSD(input_dim = n, hidden_dim=hidden_dim, diag_dim = n))
        self.ode_layer = NNODE(odefunc=ODEFunc(self.metric_space))
        self.n = n ### dimension of manifold
        self.t = t ### timepoints to extend

        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        _, predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[...,:split_size]
 

# class GroupRiemannianAutoencoder(nn.Module):

#     def __init__(self, n: int, hidden_dim: int,  t: int, rep_in, rep_out, group, loss_type: str = "L2"):
#         super(GroupRiemannianAutoencoder, self).__init__()

#         self.metric_space = GroupNeuralRiemmanianMetric(dim = n, metric_func= PSDGroup(hidden_dim=hidden_dim, diag_dim=n,
#                                                                                        group=group, rep_in=rep_in, rep_out=rep_out))
#         self.ode_layer = NNODE(odefunc=ODEFunc(self.metric_space))
#         self.n = n ### dimension of manifold
#         self.t = t ### timepoints to extend

#         self.loss_type = loss_type

#     def forward(self,initial_conditions):
#         time_steps = torch.linspace(0.0,1.0,self.t)
#         _, predicted_vals = self.ode_layer(initial_conditions, time_steps)
#         split_size = self.n
#         return predicted_vals[...,:split_size]


class GPRiemannianAutoencoder(nn.Module):

    def __init__(self, n: int, basis: torch.Tensor,  t: int, loss_type: str = "L2", m_val: int = 15):
        super(GPRiemannianAutoencoder, self).__init__()

        self.metric_space = GPRiemmanianMetric(dim = n, metric_func=PSDGP(input_dim = n, diag_dim = n, basis = basis, m_val = m_val))
        self.ode_layer = NNODE(odefunc=ODEFunc(self.metric_space))
        self.n = n ### dimension of manifold
        self.t = t ### timepoints to extend

        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        _, predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[...,:split_size]

    
class VanillaAutoencoder(nn.Module):
    def __init__(self, n: int,t: int, loss_type: str = "L2"):
        super(VanillaAutoencoder, self).__init__()
        self.ode_layer = nn.Sequential(
        nn.Linear(2*n, 16),
        nn.Tanh(),
        nn.Linear(16, 2*n)
    )
        self.n = n ### dimension of manifold
        self.t = t ### timepoints to extend

        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[...,:split_size]

    
    def loss(self, predicted_vals, actual_vals, val = False):
        if self.loss_type == "L2":
            reconstruction_loss = nn.MSELoss()(predicted_vals,actual_vals)
        return reconstruction_loss
    












