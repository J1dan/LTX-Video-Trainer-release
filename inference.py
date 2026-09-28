import torch
import os
import argparse
import pathlib
import imageio
import numpy as np
import PIL.Image
from copy import deepcopy
from torchvision.transforms import functional as F
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.model_loader import (
    load_scheduler,
    load_tokenizer,
    load_text_encoder,
    load_vae,
    load_transformer
)
from ltxv_trainer.utils import open_image_as_srgb

def export_video_for_vscode(frames, output_path, fps=24):
    """
    Exports video using imageio with libx264 and yuv420p.
    This guarantees playback compatibility with VSCode and most standard players.
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=8)
    for frame in frames:
        if isinstance(frame, PIL.Image.Image):
            frame = np.array(frame)
        elif isinstance(frame, torch.Tensor):
            if frame.dim() == 3:
                frame = frame.permute(1, 2, 0).cpu().numpy()
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()

def parse_motion_direction(prompt):
    """
    Looks for common phrases to determine if motion is left or right.
    """
    p = prompt.lower()
    left_phrases = ["turn left", "dolly left", "make a left turn", "pan left", "move left", "steer left"]
    right_phrases = ["turn right", "dolly right", "make a right turn", "pan right", "move right", "steer right"]
    
    for phrase in left_phrases:
        if phrase in p: return "left"
    for phrase in right_phrases:
        if phrase in p: return "right"
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Run ImagiNav LTX-Video inference")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use. If not specified, uses default CUDA device.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a LoRA directory (AC-MoE), LoRA file (Unified LoRA), or Model file (Unified Full Model)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance_scale", type=float, default=3.8, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    args = parser.parse_args()

    # Set device
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = "../samples/generated_videos/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    MODEL_SOURCE = "LTXV_2B_0.9.6_DEV"
    if args.model_path and "13b" in args.model_path.lower():
        MODEL_SOURCE = "LTXV_13B_097_DEV"
        print("Detected 13B model path. Switching base model to LTXV_13B_097_DEV.")

    transformer_source = MODEL_SOURCE
    lora_to_load = None

    if args.model_path:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Provided model_path does not exist: {args.model_path}")

        # Check if it's a directory (AC-MoE) or a file (Unified)
        if os.path.isdir(args.model_path):
            print("Detected AC-MoE Strategy (model_path is a directory).")
            direction = parse_motion_direction(args.prompt)
            if direction == "left":
                lora_files = [f for f in os.listdir(args.model_path) if "left" in f.lower() and f.endswith(".safetensors")]
                if lora_files:
                    lora_to_load = os.path.join(args.model_path, lora_files[0])
                else:
                    raise FileNotFoundError(f"AC-MoE Strategy Error: 'left' LoRA not found in directory {args.model_path}")
            elif direction == "right":
                lora_files = [f for f in os.listdir(args.model_path) if "right" in f.lower() and f.endswith(".safetensors")]
                if lora_files:
                    lora_to_load = os.path.join(args.model_path, lora_files[0])
                else:
                    raise FileNotFoundError(f"AC-MoE Strategy Error: 'right' LoRA not found in directory {args.model_path}")
            
            if lora_to_load:
                print(f"Motion '{direction}' detected in prompt. Selected LoRA: {lora_to_load}")
            else:
                print("INFO: Cannot determine motion direction (left/right) from prompt using common phrases (e.g. 'turn left', 'dolly right'). Proceeding without MoE LoRA.")
        elif os.path.isfile(args.model_path):
            if not args.model_path.endswith(".safetensors") and not args.model_path.endswith(".bin"):
                raise ValueError(f"Invalid model file format: {args.model_path}. Expected a checkpoint file (.safetensors).")
            if "lora" in os.path.basename(args.model_path).lower():
                print("Detected Unified LoRA Strategy.")
                lora_to_load = args.model_path
            else:
                print("Detected Unified Full Model Strategy.")
                transformer_source = args.model_path

    NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
    WIDTH, HEIGHT, FRAMES = 480, 256, 121

    print(f"Loading base components for {MODEL_SOURCE}...")
    scheduler = load_scheduler()
    tokenizer = load_tokenizer()
    text_encoder = load_text_encoder(load_in_8bit=False)
    vae = load_vae(MODEL_SOURCE, dtype=torch.bfloat16)
    
    print(f"Loading Transformer from {transformer_source}...")
    transformer = load_transformer(transformer_source, dtype=torch.bfloat16)

    pipeline = LTXConditionPipeline(
        scheduler=deepcopy(scheduler),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    pipeline = pipeline.to(device)

    if lora_to_load:
        print(f"Loading LoRA from {lora_to_load}")
        pipeline.load_lora_weights(lora_to_load, adapter_name="default")
        model_name_tag = os.path.basename(lora_to_load).split('.')[0]
    elif transformer_source != MODEL_SOURCE:
        model_name_tag = os.path.basename(transformer_source).split('.')[0]
    else:
        model_name_tag = "base_model"

    print(f"Prompt: {args.prompt}")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    pipeline_inputs = {
        "prompt": args.prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": FRAMES,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "output_reference_comparison": True,
    }
    
    print(f"Loading image: {args.image}")
    image = open_image_as_srgb(args.image)
    pipeline_inputs["image"] = image
    
    print(f"Generating video...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipeline(**pipeline_inputs)
        videos = result.frames
    
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    safe_prompt = "".join([c if c.isalnum() else "_" for c in args.prompt])[:50]
    
    for idx, video in enumerate(videos):
        suffix = f"_{idx}" if len(videos) > 1 else ""
        output_filename = f"{image_name}-{safe_prompt}-{model_name_tag}{suffix}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        export_video_for_vscode(video, output_path, fps=24)
        print(f"Saved to {output_path}")

    print("Generation complete!")

if __name__ == "__main__":
    main()
