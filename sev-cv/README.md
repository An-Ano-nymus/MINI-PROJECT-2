SEV-CV (TensorFlow) - Self-Evolutionary Generative Transformers for Computer Vision

This is a minimal TensorFlow skeleton for Phase 0 (CIFAR-10, 32x32) with:
- ViT-like generator (SEV-G)
- Hybrid conv-transformer discriminator (SEV-D)
- Evolution controller stub
- Simple training loop with Hinge GAN losses

Quickstart (Windows cmd):
1) Create venv and install deps (TensorFlow CPU or GPU):
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install tensorflow tensorflow-datasets numpy
2) Run training (Phase 0):
   python -m sevcv.scripts.train_cifar10_tf --steps 1000 --batch 128

Use a mid-size dataset under 20GB (e.g., COCO 2017 images + annotations ~18GB):
   python -m sevcv.scripts.train_cifar10_tf --dataset coco2017 --img 128 --batch 64

Note: This is a starting point. Evolutionary fitness uses a simple diversity proxy for now.
