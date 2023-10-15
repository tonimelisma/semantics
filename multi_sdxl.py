#!python3

import os
import argparse
from datetime import datetime
from diffusers import DiffusionPipeline
import torch


def generate_and_save_image(prompt):
    mps_device = torch.device("mps")

    # Load base model
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    base.to(mps_device)

    # Compile base model for performance
    base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

    # Load refiner model
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to(mps_device)

    # Compile refiner model for performance
    refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    # Define the number of steps and high_noise_frac
    n_steps = 40
    high_noise_frac = 0.8

    # Run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    # Saving the image
    subdir = "generated_images"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    current_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    shortened_prompt = prompt[:80].replace(" ", "_").replace("/", "-")
    filename = f"{current_time}_{shortened_prompt}.png"

    image_path = os.path.join(subdir, filename)
    image.save(image_path)

    print(f"Image saved to {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image based on a prompt.")
    parser.add_argument(
        "prompt", type=str, help="The text prompt for generating the image."
    )

    args = parser.parse_args()

    if not args.prompt:
        print("A text prompt is required.")
        exit(1)

    generate_and_save_image(args.prompt)
