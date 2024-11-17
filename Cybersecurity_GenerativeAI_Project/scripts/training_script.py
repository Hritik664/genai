import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.regression import MeanSquaredError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gan import Generator, Discriminator
from utils.utils import save_model
from utils.preprocessing import preprocess_input

z_dim = 100
batch_size = 64
lr = 0.0002
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    train_data_path = r'D:\genai_project\Cybersecurity_GenerativeAI_Project\data\processed\nsl_kdd_train_imputed.csv'
    labels_data_path = r'D:\genai_project\Cybersecurity_GenerativeAI_Project\data\processed\nsl_y_train.csv'

    if not os.path.exists(train_data_path) or not os.path.exists(labels_data_path):
        raise FileNotFoundError(f"Data files not found. Please check the paths: {train_data_path}, {labels_data_path}")

    train_data = pd.read_csv(train_data_path)
    train_labels = pd.read_csv(labels_data_path)

    X_train = preprocess_input(train_data)
    y_train = torch.tensor(train_labels.values, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

generator = Generator(z_dim, 141).to(device)
discriminator = Discriminator(141).to(device)

writer = SummaryWriter(log_dir='logs')
mse_metric = MeanSquaredError().to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

def clip_weights(model, clip_value):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)

def save_model_checkpoint(generator, discriminator, epoch):
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    save_model(generator, f'{model_dir}/generator_epoch_{epoch}.pth')
    save_model(discriminator, f'{model_dir}/discriminator_epoch_{epoch}.pth')

def save_generated_samples(generator, epoch, z_dim=100, num_samples=64, output_dir='generated_samples'):
    os.makedirs(output_dir, exist_ok=True)
    z = torch.randn(num_samples, z_dim).to(device)
    fake_images = generator(z)
    save_image(fake_images, os.path.join(output_dir, f'generated_images_epoch_{epoch}.png'), nrow=8, normalize=True)
    print(f"Generated images saved for epoch {epoch}")

def train_gan():
    data_loader = load_data()
    scaler = GradScaler()
    clip_value = 0.01

    accuracy_metric = BinaryAccuracy().to(device)
    d_losses, g_losses = [], []

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            real_labels = torch.ones(real_images.size(0), 1, device=device)
            fake_labels = torch.zeros(real_images.size(0), 1, device=device)

            optimizer_d.zero_grad()
            with autocast():
                real_outputs = discriminator(real_images)
                d_loss_real = criterion(real_outputs, real_labels)

                z = torch.randn(real_images.size(0), z_dim, device=device)
                fake_images = generator(z)
                fake_outputs = discriminator(fake_images.detach())
                d_loss_fake = criterion(fake_outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()
            clip_weights(discriminator, clip_value)

            optimizer_g.zero_grad()
            with autocast():
                fake_outputs = discriminator(fake_images)
                g_loss = criterion(fake_outputs, real_labels)

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            acc_real = accuracy_metric(real_outputs, real_labels)
            acc_fake = accuracy_metric(fake_outputs, fake_labels)
            writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(data_loader) + i)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(data_loader) + i)

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            mse = mse_metric(fake_images, real_images)
            writer.add_scalar('MSE for Generated Data', mse.item(), epoch * len(data_loader) + i)

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(data_loader)}], "
                        f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        save_generated_samples(generator, epoch)
        save_model_checkpoint(generator, discriminator, epoch)

        if epoch % 10 == 0:
            with torch.no_grad():
                z = torch.randn(64, z_dim).to(device)
                fake_samples = generator(z)
                save_image(fake_samples, f"generated_samples_epoch_{epoch}.png")
                writer.add_image(f'Sample Outputs Epoch {epoch}', fake_samples)

    print("Training Finished!")
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_gan()
