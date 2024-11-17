import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gan import Generator

# Load environment variables
load_dotenv()

def visualize_latent_space(generator, num_samples=1000, z_dim=100, device='cpu'):
    # Generate random latent vectors
    z = torch.randn(num_samples, z_dim).to(device)
    generated_samples = generator(z).cpu().detach().numpy()
    
    # Apply t-SNE to reduce dimensions to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(generated_samples)

    # Plot the result
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, color='blue')
    plt.title("Latent Space Visualization using t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

if __name__ == "__main__":
    # Initialize the generator model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(z_dim=100, output_dim=141).to(device)
    
    # Load model using path from .env file
    generator.load_state_dict(torch.load(os.getenv("MODEL_SAVE_PATH") + 'generator_model.pt', map_location=device))
    generator.eval()
    
    # Visualize the latent space
    visualize_latent_space(generator, num_samples=1000, z_dim=100, device=device)
