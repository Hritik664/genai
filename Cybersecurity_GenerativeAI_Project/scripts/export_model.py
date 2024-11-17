import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gan import Generator


# Define the dimensions and paths
z_dim = 100
output_dim = 2048  # Match with your generator's output dimension
checkpoint_path = 'models/generator_epoch_0.pth'  # Path to your trained model checkpoint
export_path = 'generator_model.pt'  # Path to save the exported model

# Check if the checkpoint file exists
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

# Load and prepare the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(z_dim=z_dim, output_dim=output_dim).to(device)
generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()  # Set the model to evaluation mode

# Convert the model to TorchScript
scripted_generator = torch.jit.script(generator)

# Save the TorchScript model
scripted_generator.save(export_path)
print(f"Model successfully exported to {export_path}")
