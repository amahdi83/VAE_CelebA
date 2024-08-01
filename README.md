# Variational Autoencoder (VAE) for CelebA Dataset

This repository contains a Jupyter Notebook implementation of a Variational Autoencoder (VAE) model for generating and reconstructing images from the CelebA dataset. The VAE model is designed to perform image reconstruction and latent space interpolation.

![Alt Text](output.gif)


## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)


## Project Overview

The notebook includes the following main components:
- Data preparation and loading
- VAE model definition (encoder, decoder, and VAE class)
- Training and validation routines
- Visualization of results (real images, reconstructed images, generated images, and latent space interpolation)


## Requirements

- Python 3.7 or higher
- PyTorch 1.10 or higher
- torchvision
- numpy
- matplotlib
- PIL
- Jupyter Notebook

You can install the required packages using pip:

```bash
pip install torch torchvision numpy matplotlib pillow jupyter
```


## Dataset

The CelebA dataset is used for training and testing the VAE model. Ensure you have the dataset in the celeba_gan/ directory, and the list_eval_partition.txt file should be in the same directory. You can download the dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

**Note**: For training, a subset of 10,000 images is used to manage computational resources efficiently.


## File Structure

- `VAE_CelebA.ipynb`: Jupyter Notebook containing the VAE implementation, training, and evaluation code.
- `celeba_gan/`: Directory containing the CelebA dataset and partition file.


## Usage

### 1. Data Preparation
Ensure that you have the CelebA dataset in the `celeba_gan/` directory with the `list_eval_partition.txt` file.

### 2. Run the Notebook
Open the Jupyter Notebook and run the cells to start training and evaluating the VAE model:

```bash
jupyter notebook VAE_CelebA.ipynb
```

This notebook will:

- Load and preprocess the dataset.
- Use 10,000 images for training, with additional images for validation and testing.
- Define and train the VAE model.
- Save the trained model to vae.pt.
- Plot training and validation loss curves.
- Display images: real, reconstructed, generated, and interpolated.


## Model Architecture

- **Encoder:**

    - Convolutional layers for feature extraction
    - Fully connected layers for latent variables (mu and logvar)
    
- **Decoder:**
    
    - Fully connected layer to transform latent vectors into feature maps
    - Transposed convolutional layers to reconstruct images

- **VariationalAutoEncoder (VAE):**
    - Encoder: Encodes images into latent space.
    - Decoder: Decodes latent vectors into images.
    - Reparameterization Trick: Samples from the latent space during training.


## Training and Evaluation

- **Loss Function:**
    - Reconstruction Loss: Binary cross-entropy
    - KL Divergence: Measures deviation from a standard normal distribution

- **Hyperparameters:**
    - `img_size`: 128x128
    - `capacity`: Number of channels in convolutional layers
    - `latent_dims`: 32
    - `learning_rate`: 0.001
    - `n_epochs`: 10
    - `variational_beta`: 1
    - `batch_size`: 32

- **Training Images:** 10,000 images are used for training to balance computational efficiency and model performance.


## Results
- **Real Images:** Grid of images from the training dataset.
- **Reconstructed Images:** Images reconstructed by the VAE.
- **Generated Images:** Images generated from random latent vectors.
- **Interpolation:** Visualizes linear interpolation between latent vectors of two images.


## Contributing
Feel free to fork the repository and submit pull requests. For issues or suggestions, please open an issue on GitHub.


## Acknowledgments
- CelebA dataset: MMLAB
- PyTorch and torchvision libraries
