
# VAE and CVAE Implementation

This repository contains a PyTorch implementation of Variational Autoencoders (VAE) and Conditional Variational Autoencoders (CVAE). The code is designed to be flexible and extendable, suitable for both academic research and practical applications.

## Features

- **Variational Autoencoder (VAE)**: A generative model that learns to encode data into a latent space and reconstruct it with a probabilistic approach.
- **Conditional Variational Autoencoder (CVAE)**: An extension of the VAE that incorporates label information to condition the generation process.
- **KL Annealing Schedule**: Customizable KL divergence annealing schedule, allowing for controlled regularization during training.
- **Flexible Architecture**: Easily modifiable encoder and decoder architectures with customizable hidden layers and latent dimensions.
- **Training with PyTorch Ignite**: Integrated support for training with PyTorch Ignite, including early stopping, checkpointing, and evaluation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/vae-cvae.git
   cd vae-cvae
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training a VAE

You can train a VAE using the provided `VAE` class. Below is an example of how to use it:

```python
from vae import VAE
from torch.utils.data import DataLoader
import torch

# Define your dataset and dataloader
train_loader = DataLoader(your_dataset, batch_size=64, shuffle=True)

# Initialize the VAE model
vae = VAE(n_point_features=your_input_dim, n_label_features=0, n_latent_dims=8)

# Train the model
losses, loss_terms = vae.train(train_loader, n_epochs=100, learning_rate=1e-3)
```

### 2. Training a CVAE

The `CVAE` class extends the `VAE` class to support conditional training:

```python
from vae import CVAE
from torch.utils.data import DataLoader

# Define your dataset and dataloader
train_loader = DataLoader(your_dataset, batch_size=64, shuffle=True)

# Initialize the CVAE model
cvae = CVAE(n_point_features=your_input_dim, n_label_features=your_label_dim, n_latent_dims=8)

# Train the model
losses, loss_terms = cvae.train(train_loader, n_epochs=100, learning_rate=1e-3)
```

### 3. KL Annealing

To use a custom KL annealing schedule, you can use the `give_weights` function:

```python
from vae import give_weights

# Define KL annealing schedule
weight_schedule = give_weights(n_epochs=100, start=0.5, stop=0.9, final_ratio=0.1, normalize=True)

# Pass the schedule to the training function
losses, loss_terms = vae.train(train_loader, n_epochs=100, weight_schedule=weight_schedule)
```

### 4. Sampling from the VAE/CVAE

You can generate new samples from the trained VAE/CVAE models:

```python
# Sample from the trained model
samples = vae.sample(n_samples=10)

# For CVAE with conditioning
conditioned_samples = cvae.sample(n_samples=10, y=your_labels)
```

## Demo

A complete demo of the VAE and CVAE models is available in the `vae_demo.ipynb` notebook. This notebook walks you through the entire process of defining, training, and evaluating the models, as well as visualizing the results.

To run the demo:

1. Open the notebook:

   ```bash
   jupyter notebook vae_demo.ipynb
   ```

2. Follow the instructions in the notebook to train the VAE and CVAE models on your dataset.

## References

1. Bishop, C. M. & Bishop, H. Deep Learning: Foundations and Concepts. (Springer International Publishing, Cham, 2024). doi:10.1007/978-3-031-45468-4.
2. Doersch, C. Tutorial on Variational Autoencoders. Preprint at https://doi.org/10.48550/arXiv.1606.05908 (2021).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Christoph Scherer, TU Berlin, 2024
