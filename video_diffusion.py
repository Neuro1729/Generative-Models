import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 4
seq_len = 5
img_size = 16
channels = 1
timesteps = 50
lr = 1e-3
epochs = 5
beta_start = 0.0001
beta_end = 0.02

# Noise schedule
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# Dummy “video” dataset: random tensors
def get_batch():
    return torch.rand(batch_size, seq_len, channels, img_size, img_size).to(device)

# Temporal denoiser (simple 3D conv)
class TemporalDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(seq_len, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, seq_len, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x, t):
        t_embed = t.view(-1,1,1,1,1).expand_as(x)
        x_in = torch.cat([x, t_embed], dim=2)
        return self.net(x_in)

model = TemporalDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Forward diffusion
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1,1,1,1,1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod[t]).view(-1,1,1,1,1)
    xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
    return xt, noise

# Training
for epoch in range(epochs):
    x0 = get_batch()
    t = torch.randint(0, timesteps, (batch_size,), device=device)
    xt, noise = forward_diffusion(x0, t)
    
    noise_pred = model(xt, t.float()/timesteps)
    loss = criterion(noise_pred, noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Sampling
model.eval()
with torch.no_grad():
    sample = torch.randn(batch_size, seq_len, channels, img_size, img_size).to(device)
    for t_step in reversed(range(timesteps)):
        t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.float)/timesteps
        pred_noise = model(sample, t_tensor)
        alpha_t = alphas[t_step]
        alpha_cumprod_t = alpha_cumprod[t_step]
        beta_t = betas[t_step]
        sample = (1/torch.sqrt(alpha_t)) * (sample - ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t))*pred_noise)
        if t_step > 0:
            sample += torch.sqrt(beta_t) * torch.randn_like(sample)

print("Sampled sequence shape:", sample.shape)
