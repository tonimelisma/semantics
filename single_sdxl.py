#!python3

import os
import argparse
from datetime import datetime
from diffusers import DiffusionPipeline
import torch


def generate_and_save_image(prompt):
    mps_device = torch.device("mps")

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to(mps_device)

    # Compile the model for performance
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    images = pipe(prompt=prompt).images[0]

    subdir = "generated_images"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    current_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    shortened_prompt = prompt[:80].replace(" ", "_").replace("/", "-")
    filename = f"{current_time}_{shortened_prompt}.png"

    image_path = os.path.join(subdir, filename)
    images.save(image_path)

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
