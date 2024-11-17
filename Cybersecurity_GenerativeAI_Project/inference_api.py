from flask import Flask, request, jsonify, send_file
import torch
import io
import zipfile
from torchvision.utils import save_image
from torchmetrics.functional import cosine_similarity
from functools import wraps
from utils.preprocessing import preprocess_input  # Ensure this is implemented correctly
import os
import sys
from dotenv import load_dotenv
import numpy as np

# Add parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gan import Generator

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Constants
z_dim = 100  # Latent vector size for the generator
output_dim = 141  # Adjust to your generator's output dimension
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.getenv('MODEL_PATH', 'D:/genai_project/models/generator_epoch_0.pth')

# Load API key from .env
API_KEY = os.getenv('API_KEY', 'default_fallback_value')
if not API_KEY or API_KEY == 'default_fallback_value':
    raise ValueError("API_KEY is not set. Please provide a valid API_KEY.")

# Fix the state_dict keys (if needed)
def fix_state_dict_keys(model_path):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    updated_state_dict = {key.replace("model.", "fc."): value for key, value in state_dict.items()}
    torch.save(updated_state_dict, model_path)
    print("State_dict keys updated successfully.")

fix_state_dict_keys(model_path)

# Load the pre-trained generator model
generator = Generator(z_dim, output_dim).to(device)

try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    generator.load_state_dict(checkpoint)
    generator.eval()
    print(f"Model loaded successfully from {model_path}.")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    # Debugging state_dict mismatch
    model_dict = generator.state_dict()
    for key in model_dict.keys():
        if key in checkpoint:
            print(f"{key}: Expected {model_dict[key].shape}, Found {checkpoint[key].shape}")
        else:
            print(f"{key} is missing in checkpoint.")
    exit(1)


# Middleware to validate API key
def require_api_key(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if request.headers.get('API-Key') != API_KEY:
            return jsonify({'error': 'Unauthorized access'}), 403
        return func(*args, **kwargs)
    return decorated_function

@app.route('/generate_ddos', methods=['POST'])
def generate_ddos():
    # Simulate timestamps
    timestamps = np.arange(0, 100, 1)  # 100 seconds
    traffic_rates = [100 if i < 30 or i > 70 else 10000 for i in timestamps]  # Low traffic, spike, low traffic

    # Create output data
    traffic_data = [
        {"timestamp": int(ts), "traffic_rate": rate}
        for ts, rate in zip(timestamps, traffic_rates)
    ]

    return jsonify(traffic_data)

@app.route('/simulate_malware', methods=['POST'])
def simulate_malware():
    # Simulated malware data
    malware_data = [
        {"timestamp": "2024-11-17T10:00:00Z", "file": "data.txt", "action": "encrypted", "source": "ransomware.exe"},
        {"timestamp": "2024-11-17T10:01:00Z", "file": "logs.txt", "action": "deleted", "source": "trojan.exe"}
    ]

    return jsonify(malware_data)


# Route to generate images
@app.route('/generate', methods=['POST'])
@require_api_key
def generate():
    try:
        # Parse the request data
        data = request.get_json()
        num_images = data.get('num_images', 1)  # Default to 1 image
        reference_sample = data.get('reference_sample')  # Optional: to calculate similarity

        # Generate latent vectors
        z = torch.randn(num_images, z_dim).to(device)
        with torch.no_grad():
            generated_images = generator(z)

        # Calculate similarity if reference sample is provided
        similarity = None
        if reference_sample:
            reference_tensor = preprocess_input(reference_sample).to(device)
            similarity = cosine_similarity(generated_images, reference_tensor).mean().item()

        # Send generated images as response
        if num_images == 1:
            img_buffer = io.BytesIO()
            save_image(generated_images, img_buffer, format="png")
            img_buffer.seek(0)
            return send_file(img_buffer, mimetype='image/png')
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for i, img in enumerate(generated_images):
                    img_buffer = io.BytesIO()
                    save_image(img, img_buffer, format="png")
                    img_buffer.seek(0)
                    zip_file.writestr(f"image_{i + 1}.png", img_buffer.getvalue())
            zip_buffer.seek(0)
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, attachment_filename='generated_images.zip')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
