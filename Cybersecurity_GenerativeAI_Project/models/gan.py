import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.constant_(m.bias, 0)

def train_gan(generator, discriminator, dataloader, num_epochs=100, lr=0.0002, beta1=0.5):
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(dataloader):
            real_data = real_data.float()

            # Train Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(real_data.size(0), 1)
            real_validity = discriminator(real_data)
            real_loss = criterion(real_validity, real_labels)

            noise = torch.randn(real_data.size(0), generator.fc[0].in_features)
            fake_data = generator(noise)
            fake_labels = torch.zeros(real_data.size(0), 1)
            fake_validity = discriminator(fake_data.detach())
            fake_loss = criterion(fake_validity, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_validity = discriminator(fake_data)
            g_loss = criterion(fake_validity, real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

    print("Training complete!")
