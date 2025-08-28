# SEV-CV Mathematics (Concise)

- Adversarial (hinge):
  - D: L_D = E_x[max(0, 1 - D(x))] + E_z[max(0, 1 + D(G(z)))]
  - G: L_G = -E_z[D(G(z))]
- Optional R1: L_R1 = (γ/2) E_x[||∇_x D(x)||^2]
- Path length (generator): encourages consistent latent-to-image Jacobian norm.
- Wavelet loss: L_wave = ||W(x_real) - W(x_fake)||_1 (W: wavelet transform)
- Contrastive alignment (conditional): InfoNCE between image and text/class embeddings.

Transformer block (per token t):
- t' = t + MHA(LN(t))
- t'' = t' + MLP(LN(t'))

Evolutionary controller (simplified):
- Individual θ = {policy (lr, ema), micro (heads, depth), prompt/z seeds}
- Mutate: lr ← lr·10^{N(0,σ)}, ema ← clip(ema + N(0,σ)), heads/depth ±1
- Evaluate: fitness f(θ) (proxy here; later FID/KID/PR)
- Select: keep top-k, elitism; repeat.

Training loop:
1) K adversarial steps (update D, then G)
2) Mutate population, evaluate small batches, select
3) Curriculum: adjust resolution/augmentation by controller

References: Hinge GAN, R1 regularization, ViT/Swin attention, Pareto selection, successive halving.