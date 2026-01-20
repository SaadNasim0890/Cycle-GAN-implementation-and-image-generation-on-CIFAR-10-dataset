
This repository contains the implementation for Assignment 2 of the Generative AI course. The project is divided into three main questions, covering CycleGANs, and Image Generation using Transformers & Diffusion.

## Section 1: CycleGAN (Q1)
**Location:** `Q1/cycle-GAN.ipynb`

This section implements a **CycleGAN** (Cycle-Consistent Generative Adversarial Network) for unpaired image-to-image translation. 
- **Goal:** To translate images from one domain to another (e.g., horses to zebras, summer to winter) without paired training examples.
- **Key Components:**
  - Two Generators ($G_{AB}$, $G_{BA}$)
  - Two Discriminators ($D_A$, $D_B$)
  - Cycle Consistency Loss to ensure the original image is recoverable after a round-trip translation.

## Section 3: Image Generation with SiT (Q3)
**Location:** `Q3/Q3A/genAI-Q3.ipynb` & `Q3/Q3B/genAI-Q3_B.ipynb`

This section implements a **Scalable Image Transformer (SiT)** combined with a diffusion-based approach for generating images of cats and dogs.

- **Model:** `SiT_REG` (Scalable Image Transformer with Regularized sampling).
- **Method:**
  - Uses a transformer-based architecture instead of the strictly U-Net based backbone common in many latent diffusion models.
  - Implements **Velocity Prediction** (v-prediction) for the diffusion process.
  - Features **Rotational Equilibrium Guidance (REG)** or similar class-guidance mechanisms to control generation (e.g., choosing between 'cat' and 'dog').
- **Dataset:** A subset of **CIFAR-10** (Classes: Cat & Dog).
- **Sampling:** Uses the **Euler Discrete Scheduler** for generating images from pure noise over ~250 timesteps.

---

## Dependencies

To run the notebooks in this project, you will need the following Python libraries:

```bash
pip install tensorflow keras torch torchvision diffusers timm accelerate numpy matplotlib
```

*Note:Q3 relies on PyTorch and Hugging Face Diffusers.*
