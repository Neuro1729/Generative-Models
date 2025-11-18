import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 5
batch_size = 128
lr = 1e-3
timesteps = 50
beta_start = 0.0001
beta_end = 0.02
img_size = 28
n_classes = 10  # MNIST labels

betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# Conditional MLP Denoiser
class ConditionalDenoiser(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.embed = nn.Embedding(n_classes, 16)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size*img_size + 1 + 16, 256),  # +1 for timestep, +16 for label
            nn.ReLU(),
            nn.Linear(256, img_size*img_size),
            nn.Sigmoid()
        )
    def forward(self, x, t, y):
        t = t.view(-1, 1)
        y_emb = self.embed(y)
        x_in = torch.cat([x.view(x.size(0), -1), t, y_emb], dim=1)
        return self.net(x_in).view(-1, 1, img_size, img_size)

# Load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = ConditionalDenoiser(n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Forward diffusion
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0).to(device)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1,1,1,1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod[t]).view(-1,1,1,1)
    xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
    return xt, noise

# Training
for epoch in range(epochs):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        b = imgs.size(0)
        t = torch.randint(0, timesteps, (b,), device=device)
        xt, noise = forward_diffusion(imgs, t)
        
        noise_pred = model(xt, t.float()/timesteps, labels)
        loss = criterion(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Sampling conditioned on specific labels
model.eval()
with torch.no_grad():
    target_labels = torch.tensor([0,1,2,3,4,5,6,7,8,9], device=device)
    n_samples = len(target_labels)
    sample = torch.randn(n_samples, 1, img_size, img_size).to(device)
    
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.float)/timesteps
        pred_noise = model(sample, t_tensor, target_labels)
        alpha_t = alphas[t]
        alpha_cumprod_t = alpha_cumprod[t]
        beta_t = betas[t]
        sample = (1/torch.sqrt(alpha_t)) * (sample - ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t))*pred_noise)
        if t > 0:
            sample += torch.sqrt(beta_t) * torch.randn_like(sample)

# Plot results
plt.figure(figsize=(10,2))
for i in range(n_samples):
    plt.subplot(1, n_samples, i+1)
    plt.imshow(sample[i,0].cpu(), cmap='gray')
    plt.axis('off')
plt.show()
