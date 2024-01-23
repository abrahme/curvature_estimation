from typing import List, Tuple
import torch
import torch.optim as optim
from .model import RiemannianAutoencoder


def train(input_trajectories, initial_conditions: Tuple[torch.Tensor, torch.Tensor], val_input_trajectories, 
          val_initial_conditions:Tuple[torch.Tensor, torch.Tensor],  epochs, 
          regularizer:float, n, t, m:List[int], c:float, basis, active_dims: List, return_preds:bool = False, val: bool = False):
    model = RiemannianAutoencoder(n,t,m,c,regularizer, basis, active_dims)
    param_list = []
    for gp in model.gp_components:
        param_list += list(gp.parameters())
    optimizer = optim.Adam(param_list, lr=0.01)

    preds = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        predicted_trajectories, _ = model.forward(initial_conditions)
        loss = model.loss(torch.permute(predicted_trajectories.float(), (1,0,2)), input_trajectories.float())
        # Backward pass and optimization
        
        loss.backward(retain_graph=True)
        optimizer.step()

        if val:
            with torch.no_grad():
                predicted_val_trajectories, _ = model.forward(val_initial_conditions)
                val_loss = model.loss(torch.permute(predicted_val_trajectories.float(), (1,0,2)), val_input_trajectories.float(), val = True)
        else:
            val_loss = "None"
        if return_preds:
            preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")



    return model, preds


