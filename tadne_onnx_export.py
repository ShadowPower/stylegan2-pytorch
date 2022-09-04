import torch

from model import Generator

if __name__ == '__main__':
    # network structure
    size = 512
    additional_multiplier = 2
    latent = 512 * additional_multiplier
    n_mlp = 4
    channel_multiplier = 2
    batch_size = 1
    device = 'cuda'

    # input and output files

    # check ckpt file with:
    # generate.py -o "output" --ckpt "checkpoint/aydao-anime-danbooru2019s-512-5268480.pt" --size 512 --n_mlp 4 --additional_multiplier 2 --load_latent_vec --no-randomize-noise --truncation 1 --pics 1 --sample 1 --ncol 1 --split 1 --ext jpg

    ckpt = r'checkpoint/aydao-anime-danbooru2019s-512-5268480.pt'
    output_file = r'checkpoint/aydao-anime-danbooru2019s-512-5268480.onnx'

    g_ema = Generator(
        size, latent, n_mlp,
        channel_multiplier=channel_multiplier,
        additional_multiplier=additional_multiplier
    ).to(device)
    checkpoint = torch.load(ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"], strict=True)
    g_ema.eval()

    sample_z = torch.randn(batch_size, latent, device=device)
    mean_latent = checkpoint['latent_avg'].to(device)
    truncation = torch.tensor(0.6, dtype=torch.float32)

    dummy_input = (
        [sample_z],
        truncation,
        mean_latent,
        False,
        None,
        False,
        None,
        False,
    )
    torch.onnx.export(g_ema, dummy_input, output_file, verbose=True,
                      input_names=['styles', 'truncation', 'mean_latent'], output_names=['image'],
                      opset_version=10)