import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embedding_dim = 128
batch_size = 128
lr = 1e-3
epochs = 10
temperature = 0.07

# Dataset
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Text labels
labels_text = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Image encoder (simple CNN)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, embedding_dim)
        )
    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x, dim=-1)
        return x

# Text encoder (embedding + linear)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.fc = nn.Linear(64, embedding_dim)
    def forward(self, x):
        x = self.emb(x).mean(dim=1)  # average over tokens
        x = F.normalize(self.fc(x), dim=-1)
        return x

# Simple tokenizer for labels
word2idx = {word:i for i, word in enumerate(set(" ".join(labels_text).split()))}
vocab_size = len(word2idx)

def tokenize(text):
    return torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)

text_embeddings = [tokenize(label) for label in labels_text]

# Models
img_encoder = ImageEncoder(embedding_dim).to(device)
txt_encoder = TextEncoder(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(list(img_encoder.parameters()) + list(txt_encoder.parameters()), lr=lr)

# Training
for epoch in range(epochs):
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        batch_size_curr = imgs.size(0)
        optimizer.zero_grad()
        
        # Encode images
        img_emb = img_encoder(imgs)
        
        # Encode text labels for batch
        txt_emb = torch.stack([txt_encoder(text_embeddings[l].to(device)) for l in labels])
        
        # Compute similarity matrix
        logits = img_emb @ txt_emb.T / temperature
        labels_gt = torch.arange(batch_size_curr).to(device)
        loss_i2t = F.cross_entropy(logits, labels_gt)
        loss_t2i = F.cross_entropy(logits.T, labels_gt)
        loss = (loss_i2t + loss_t2i) / 2
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Test similarity for a few images
imgs, labels = next(iter(train_loader))
imgs = imgs.to(device)
img_emb = img_encoder(imgs)
txt_emb = torch.stack([txt_encoder(text_embeddings[l].to(device)) for l in labels])
sim = img_emb @ txt_emb.T
print("Similarity matrix (image x text):")
print(sim[:5, :5].detach().cpu())
