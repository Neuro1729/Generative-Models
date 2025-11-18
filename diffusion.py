import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 5
batch_size = 128
lr = 1e-3
timesteps = 100  # Number of diffusion steps
beta_start = 0.0001
beta_end = 0.02
img_size = 28

# Linear noise schedule
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# Simple MLP as denoiser
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size*img_size + 1, 256),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(256, img_size*img_size),
            nn.Sigmoid()
        )
    def forward(self, x, t):
        t = t.view(-1, 1)
        x_in = torch.cat([x.view(x.size(0), -1), t], dim=1)
        return self.net(x_in).view(-1, 1, img_size, img_size)

# Load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Denoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Forward diffusion: add noise
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0).to(device)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1,1,1,1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod[t]).view(-1,1,1,1)
    xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
    return xt, noise

# Training loop
for epoch in range(epochs):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        b = imgs.size(0)
        t = torch.randint(0, timesteps, (b,), device=device)  # Random timestep
        xt, noise = forward_diffusion(imgs, t)
        
        noise_pred = model(xt, t.float()/timesteps)
        loss = criterion(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Sampling from noise
model.eval()
with torch.no_grad():
    sample = torch.randn(16, 1, img_size, img_size).to(device)
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((16,), t, device=device, dtype=torch.float)/timesteps
        pred_noise = model(sample, t_tensor)
        alpha_t = alphas[t]
        alpha_cumprod_t = alpha_cumprod[t]
        beta_t = betas[t]
        sample = (1/torch.sqrt(alpha_t)) * (sample - ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t))*pred_noise)
        if t > 0:
            sample += torch.sqrt(beta_t) * torch.randn_like(sample)

import matplotlib.pyplot as plt

# Show 16 generated samples
plt.figure(figsize=(4,4))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(sample[i,0].cpu(), cmap='gray')
    plt.axis('off')
plt.show()
