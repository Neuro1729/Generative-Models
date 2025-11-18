import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
seq_len = 20
batch_size = 64
epochs = 50
lr = 1e-3
timesteps = 50  # diffusion steps
hidden_dim = 64

# Linear noise schedule
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# Synthetic dataset: noisy sine waves
def generate_sine(batch_size, seq_len):
    x = np.linspace(0, 2*np.pi, seq_len)
    sequences = []
    for _ in range(batch_size):
        phase = np.random.rand() * 2 * np.pi
        seq = np.sin(x + phase)
        sequences.append(seq)
    sequences = np.array(sequences)
    return torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # [B, seq_len, 1]

# LSTM Denoiser
class LSTMDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t):
        # x: [B, seq_len, 1], t: [B]
        t_embed = t.view(-1, 1, 1).expand(-1, x.size(1), 1)
        x_in = torch.cat([x, t_embed], dim=-1)
        lstm_out, _ = self.lstm(x_in)
        out = self.fc(lstm_out)
        return out

model = LSTMDenoiser(input_dim=1, hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Forward diffusion: add noise
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0).to(device)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1,1,1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod[t]).view(-1,1,1)
    xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
    return xt, noise

# Training loop
for epoch in range(epochs):
    sequences = generate_sine(batch_size, seq_len).to(device)
    t = torch.randint(0, timesteps, (batch_size,), device=device)
    xt, noise = forward_diffusion(sequences, t)
    
    optimizer.zero_grad()
    noise_pred = model(xt, t.float()/timesteps)
    loss = criterion(noise_pred, noise)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Sampling from noise
model.eval()
with torch.no_grad():
    sample = torch.randn(batch_size, seq_len, 1).to(device)
    for t_step in reversed(range(timesteps)):
        t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.float)/timesteps
        pred_noise = model(sample, t_tensor)
        alpha_t = alphas[t_step]
        alpha_cumprod_t = alpha_cumprod[t_step]
        beta_t = betas[t_step]
        sample = (1/torch.sqrt(alpha_t)) * (sample - ((1-alpha_t)/torch.sqrt(1-alpha_cumprod_t))*pred_noise)
        if t_step > 0:
            sample += torch.sqrt(beta_t) * torch.randn_like(sample)

# Plot some generated sequences
plt.figure(figsize=(10,4))
for i in range(5):
    plt.plot(sample[i].cpu().numpy(), label=f"Seq {i+1}")
plt.title("Generated sequences from LSTM Diffusion")
plt.legend()
plt.show()
