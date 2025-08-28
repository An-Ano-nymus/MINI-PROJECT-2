from __future__ import annotations
import argparse
from sevcv.train.trainer import train_phase0
from sevcv.evolution.controller import EvolutionController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--z", type=int, default=128)
    parser.add_argument("--img", type=int, default=32)
    parser.add_argument("--evolve", action="store_true")
    args = parser.parse_args()

    evo = EvolutionController(pop_size=4) if args.evolve else None
    result = train_phase0(steps=args.steps, batch_size=args.batch, img_size=args.img, z_dim=args.z, evo=evo)
    print({k: v for k, v in result.items() if k != "generator" and k != "discriminator"})


if __name__ == "__main__":
    main()
