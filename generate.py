import argparse
import random

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent, randomize_noise):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            samples = []
            for _ in range(args.split):
                sample_z = torch.randn(args.sample // args.split, args.latent, device=device)

                sample, _ = g_ema(
                        [sample_z], truncation=args.truncation, truncation_latent=mean_latent,
                    randomize_noise=randomize_noise
                )
                samples.extend(sample)

            utils.save_image(
                samples,
                f"{args.output_dir}/{str(i).zfill(6)}.{args.ext}",
                nrow=args.ncol,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-dir", '-o', type=str, required=True)

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="number of samples to be generated for each image",
    )
    parser.add_argument("--ncol", type=int, default=10)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--ext", type=str, default='png')
    parser.add_argument(
        "--pics", type=int, default=1, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument("--additional_multiplier", type=int, default=1)
    parser.add_argument("--load_latent_vec", action='store_true')
    parser.add_argument("--no-randomize-noise", dest='randomize_noise', action='store_false')
    parser.add_argument("--n_mlp", type=int, default=8)

    args = parser.parse_args()

    seed = random.randint(0, 2**31-1) if args.seed == -1 else args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args.latent = 512 * args.additional_multiplier

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        additional_multiplier=args.additional_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"], strict=True)

    if not args.load_latent_vec:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = checkpoint['latent_avg'].to(device)

    generate(args, g_ema, device, mean_latent, randomize_noise=args.randomize_noise)
