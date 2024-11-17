import os
import torch

# Function to save the model
def save_model(model, path):
    """
    Saves the model state_dict to the specified path.
    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path where the model will be saved.
    """
    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(model, filepath):
    """
    Loads the model state_dict from the specified file path.
    Args:
        model (torch.nn.Module): The model to load the state_dict into.
        filepath (str): The file path where the model is saved.
    """
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set to evaluation mode
        print(f"Model loaded from {filepath}")
    else:
        print(f"Error: File not found at {filepath}")

# Function to ensure directory exists
def ensure_dir_exists(path):
    """
    Checks if a directory exists, and creates it if not.
    Args:
        path (str): The directory path to check.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at {path}")

# Function to initialize a model from a checkpoint
def initialize_model_from_checkpoint(model, checkpoint_path, device):
    """
    Loads a model from the checkpoint into the specified device.
    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        checkpoint_path (str): The path to the checkpoint file.
        device (torch.device): The device to load the model on (CPU or CUDA).
    """
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        print(f"Model initialized from checkpoint at {checkpoint_path}")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")

if __name__ == "__main__":
    # Example: Create a dummy model and use the utility functions
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = torch.nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)

    model = DummyModel()

    # Example usage: Save and load model
    save_model(model, "dummy_model.pth")
    load_model(model, "dummy_model.pth")
