import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, EulerDiscreteScheduler
#! from instantid
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
import cv2
import torch
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.pipelines.pipeline_animation_instantid import AnimationINSTANTIDPipeline, draw_kps
from animatediff.utils.util import save_videos_grid, load_weights

from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import numpy as np


@torch.no_grad()
def main(args):
	# breakpoint()
	*_, func_args = inspect.getargvalues(inspect.currentframe())
	func_args = dict(func_args)
	
	time_str = datetime.datetime.now().strftime("%Y-%m-%d")
	
	savedir = f"sample/{Path(args.exp_config).stem}_{args.H}_{args.W}-{time_str}"
	os.makedirs(savedir, exist_ok=True)
	
	# Load Config
	exp_config	= OmegaConf.load(args.exp_config)
	config = OmegaConf.load(args.base_config)
	config = OmegaConf.merge(config, exp_config)

	if config.get('base_model_path', '') != '':
		args.pretrained_model_path = config.base_model_path
	
	# Load Component
	tokenizer	 = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
	vae			 = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
	tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer_2")
	text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="text_encoder_2")

	# init unet model
	unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))

	# Enable memory efficient attention
	if is_xformers_available() and args.xformers:
		# breakpoint()
		unet.enable_xformers_memory_efficient_attention()

	scheduler = EulerDiscreteScheduler(timestep_spacing='leading', steps_offset=1,	**config.noise_scheduler_kwargs)

	if args.instantid:
		# prepare 'antelopev2' under ./models
		app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
		app.prepare(ctx_id=0, det_size=(640, 640))

		# prepare models under ./checkpoints
		face_adapter = f'/home/nas4_user/taewoongkang/repos/2d/AnimateDiff/checkpoints/ip-adapter.bin'
		controlnet_path = f'/home/nas4_user/taewoongkang/repos/2d/AnimateDiff/checkpoints/ControlNetModel'

		controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to("cuda")


		pipeline = AnimationINSTANTIDPipeline(
		  unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
		  text_encoder_2 = text_encoder_two, tokenizer_2=tokenizer_two, controlnet=controlnet
		).to("cuda")

		pipeline.load_ip_adapter_instantid(face_adapter)
	else:
		pipeline = AnimationPipeline(
			unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
			text_encoder_2 = text_encoder_two, tokenizer_2=tokenizer_two
		).to("cuda")

	# Load model weights
	pipeline = load_weights(
		pipeline = pipeline,
		motion_module_path = config.get("motion_module_path", ""),
		ckpt_path = config.get("ckpt_path", ""),
		lora_path = config.get("lora_path", ""),
		lora_alpha = config.get("lora_alpha", 0.8)
	)

	pipeline.unet = pipeline.unet.half()
	pipeline.text_encoder = pipeline.text_encoder.half()
	pipeline.text_encoder_2 = pipeline.text_encoder_2.half()
	pipeline.enable_model_cpu_offload()
	pipeline.enable_vae_slicing()

	prompts	   = config.prompt
	n_prompts  = config.n_prompt

	random_seeds = config.get("seed", [-1])
	random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
	random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
	seeds = []
	samples = []

	with torch.inference_mode():
		for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
			# manually set random seed for reproduction
			if random_seed != -1: torch.manual_seed(random_seed)
			else: torch.seed()
			seeds.append(torch.initial_seed())
			print(f"current seed: {torch.initial_seed()}")
			print(f"sampling {prompt} ...")

			if args.instantid:
				# face_image = load_image("./examples/yann-lecun_resize.jpg")
				# face_image = load_image("./examples/musk_resize.jpeg")
				face_image = load_image(args.face_dir)
				face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
				face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
				face_emb = face_info['embedding']
				face_kps = draw_kps(face_image, face_info['kps'])
				sample = pipeline(
					prompt,
					negative_prompt	  = n_prompt,
					num_inference_steps = config.get('steps', 100),
					guidance_scale	  = config.get('guidance_scale', 10),
					width				  = args.W,
					height			  = args.H,
					single_model_length = args.L,
					image_embeds=face_emb,
					imageinput=face_kps,
					controlnet_conditioning_scale=0.8,
					ip_adapter_scale=0.8,
				).videos

			else:
				sample = pipeline(
					prompt,
					negative_prompt	  = n_prompt,
					num_inference_steps = config.get('steps', 100),
					guidance_scale	  = config.get('guidance_scale', 10),
					width				  = args.W,
					height			  = args.H,
					single_model_length = args.L,
				).videos
			samples.append(sample)
			prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
			prompt = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

			# save video
			save_videos_grid(sample, f"{savedir}/sample/{prompt}.mp4")
			print(f"save to {savedir}/sample/{prompt}.mp4")

	samples = torch.concat(samples)
	save_videos_grid(samples, f"{savedir}/sample-{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.mp4", n_rows=4)
	config.seed = seeds
	OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/",)
	parser.add_argument("--pretrained_model_path", type=str, default="/home/nas4_dataset/vision/pretrained_models/stable-diffusion-xl-base-1.0",)
	parser.add_argument("--base_config",	  type=str, default="configs/inference/inference.yaml")    
	parser.add_argument("--exp_config",		 type=str, required=True)

	parser.add_argument("--L", type=int, default=16 )
	parser.add_argument("--W", type=int, default=1024)
	parser.add_argument("--H", type=int, default=1024)
	
	parser.add_argument("--xformers", action="store_true")

	parser.add_argument("--instantid", type=bool, default=False)
	parser.add_argument("--face_dir", type=str, default="/home/nas4_user/taewoongkang/repos/2d/AnimateDiff/examples/kaifu_resize.png")

	args = parser.parse_args()
	main(args)
