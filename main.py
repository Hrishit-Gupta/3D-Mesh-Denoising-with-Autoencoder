import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=3):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MeshDataset(Dataset):
    def __init__(self, noisy_vertices, clean_vertices):
        self.noisy = torch.tensor(noisy_vertices, dtype=torch.float32)
        self.clean = torch.tensor(clean_vertices, dtype=torch.float32)
    
    def __len__(self):
        return len(self.noisy)
    
    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

def generate_noisy_mesh(noise_level=0.02):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    noise = np.random.normal(scale=noise_level, size=vertices.shape)
    noisy_vertices = vertices + noise
    mesh.vertices = o3d.utility.Vector3dVector(noisy_vertices)
    return mesh, vertices

def train_autoencoder(model, dataloader, epochs=200, lr=0.01, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(epochs):
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

def denoise_mesh(model, noisy_vertices, device='cpu'):
    model.eval()
    with torch.no_grad():
        denoised_vertices = model(torch.tensor(noisy_vertices, dtype=torch.float32).to(device)).cpu().numpy()
    return denoised_vertices

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder()
    
    mesh, clean_vertices = generate_noisy_mesh()
    noisy_vertices = np.asarray(mesh.vertices)
    dataset = MeshDataset(noisy_vertices, clean_vertices)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    train_autoencoder(model, dataloader, device=device)
    denoised_vertices = denoise_mesh(model, noisy_vertices, device)
    
    mesh.vertices = o3d.utility.Vector3dVector(denoised_vertices)
    o3d.visualization.draw_geometries([mesh], window_name="Denoised Mesh")

if __name__ == "__main__":
    main()
