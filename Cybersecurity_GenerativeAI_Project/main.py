import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gan import Generator

# Test loading the generator model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator(z_dim=100, output_dim=141).to(device)
generator.load_state_dict(torch.load(os.getenv("MODEL_SAVE_PATH") + 'generator_model.pt', map_location=device))

print(f"Model loaded successfully on {device}")
