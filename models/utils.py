import torch 
from pathlib import Path


def save_model(model: torch.nn.Module, sample_size: int, manifold: str):
    parent_dir = "trained_models"
    fpath = f"{parent_dir}/{manifold}/{sample_size}"

        # Specify the directory path
    directory_path = Path(fpath)

    # Check if the directory exists
    if not directory_path.exists():
        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    torch.save(model.state_dict(), fpath + "/model.pth")

def load_model(model: torch.nn.Module, sample_size: int, manifold:str):
    parent_dir = "trained_models"
    fpath = f"{parent_dir}/{manifold}/{sample_size}"

        # Specify the directory path
    
    
    model.load_state_dict(torch.load(fpath + "/model.pth"))
    return model 