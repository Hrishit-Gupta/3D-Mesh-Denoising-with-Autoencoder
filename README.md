# 3D Mesh Denoising with Autoencoder

## Overview
This project implements a **deep learning-based 3D mesh denoising** method using a **Denoising Autoencoder (DAE)**. The model is trained on synthetic noisy 3D meshes and learns to remove noise from the vertices.

## Features
- Generates a synthetic noisy 3D **cube** for training.
- Implements a **PyTorch-based Autoencoder** for denoising.
- Uses **Open3D** for 3D mesh processing and visualization.
- Supports **CUDA acceleration** if available.

## Requirements
Make sure you have the following dependencies installed:

```bash
pip install numpy open3d torch torchvision torchaudio
```

## How It Works
1. A synthetic **3D cube** mesh is created using Open3D.
2. **Random noise** is added to the mesh vertices.
3. A **Denoising Autoencoder (DAE)** is trained using the noisy and clean vertex pairs.
4. The trained model denoises the noisy mesh and **restores** its original shape.
5. The results are displayed using Open3D.

## File Structure
```
/3D-Mesh-Denoising
│── main.py          # Main script to train and denoise the mesh
│── requirements.txt # List of dependencies
│── README.md        # Project documentation
```

## Usage
### Run the script:
```bash
python main.py
```
This will:
- Train the autoencoder on a **synthetic noisy cube**.
- Denoise the cube using the trained model.
- Display the denoised mesh.

### GPU Acceleration
If you have a CUDA-compatible GPU, the script will **automatically use it** for training.

## Output Example
The script will output:
```
Epoch 0, Loss: 0.636847
Epoch 50, Loss: 0.139348
Epoch 100, Loss: 0.071972
...
Denoised mesh displayed.
```

A window will open showing the **denoised 3D mesh**.

## Future Improvements
- **Support for real-world noisy meshes** (e.g., loading `.ply` or `.obj` files).
- **Enhance autoencoder architecture** for better denoising.
- **Use point cloud-based learning** instead of vertex-based processing.


