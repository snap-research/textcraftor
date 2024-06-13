import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from pipelines.pipeline_stable_diffusion import StableDiffusionPipelineTC as StableDiffusionPipeline
from pipelines.scheduling_ddim import DDIMScheduler

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.schedulers import DDPMScheduler, \
    DEISMultistepScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    PNDMScheduler, EulerAncestralDiscreteScheduler
import csv
import json
from pprint import pprint
import pandas as pd
import logging
from PIL import Image
from collections import OrderedDict, defaultdict
import random
import argparse

from transformers import pipeline
import hpsv2


def parser_spec():
    parser = argparse.ArgumentParser()
    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument(
        "--ckpt_path",
        default="",
        help=
        ("pytorch checkpoint"
         ))
    parser.add_argument(
        "--unet_path",
        default=None,
        help=
        ("unet checkpoint"
         ))
    parser.add_argument(
        "--text_path",
        default=None,
        help=
        ("text checkpoint"
         ))
    parser.add_argument(
        "--prompt_eng",
        default=False,
        action="store_true",
        help=
        ("whether use prompt engineering"
         ))
    parser.add_argument(
        "--output_path",
        default="tests/paper_ckpt_hpsv2",
        help=
        ("output path for saving images"
         ))

    parser.add_argument('--seed', default=2023, type=int, help='seed')
    parser.add_argument('--guidance', default=7.5, type=float, help='cfg guidance')
    parser.add_argument('--step', default=25, type=int, help='numer of steps')
    parser.add_argument('--grad_steps', default=25, type=int, help='numer of text aug steps')
    parser.add_argument('--prompt_weighting', default=1.0, type=float, help='prompt weighting')
    parser.add_argument('--scheduler', default='DDIM', type=str, choices=['DDIM', 'DPMSingle', 'Eulr'],
                        help='scheduler')
    return parser


def main(args):
    '''
    settings there ############################################################################################
    '''
    args.gpu = 0
    seed = args.seed
    guidance = args.guidance
    ckpt_path = args.ckpt_path
    output_path_name = args.scheduler + args.ckpt_path.split('/')[0] + 'St{:.0f}C{:.1f}/'.format(
        args.step, args.guidance)
    output_path = os.path.join(args.output_path, output_path_name)

    '''
    settings end #############################################################################################
    '''

    generator_s = torch.Generator("cuda").manual_seed(seed)
    pipe_textcraftor = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")

    scheduler_config = pipe_textcraftor.scheduler.config
    # scheduler_config['prediction_type'] = 'v_prediction'
    # scheduler_config['prediction_type'] = 'sample'
    scheduler_config['prediction_type'] = 'epsilon'

    if args.scheduler == 'DDIM':
        noise_scheduler = DDIMScheduler.from_config(scheduler_config)
    elif args.scheduler == 'DPMSingle':
        noise_scheduler = DPMSolverSinglestepScheduler.from_config(scheduler_config)
    elif args.scheduler == 'Eulr':
        noise_scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

    pipe_textcraftor.scheduler = noise_scheduler
    pipe_textcraftor.__setattr__('text_encoder_origin', copy.deepcopy(pipe_textcraftor.text_encoder))

    if args.text_path is not None:
        ckpt = torch.load(args.text_path, map_location="cpu")
        new_ckpt = {}
        for item in ckpt:
            new_ckpt[item[7:]] = ckpt[item]
        pipe_textcraftor.text_encoder.load_state_dict(new_ckpt, strict=False)
        print('text weight load success: ', args.text_path)

    if args.unet_path is not None:
        ckpt = torch.load(args.unet_path, map_location="cpu")
        new_ckpt = {}
        for item in ckpt:
            new_ckpt[item[7:]] = ckpt[item]
        pipe_textcraftor.unet.load_state_dict(new_ckpt, strict=True)
        print('unet weight load success: ', args.unet_path)

    if args.prompt_eng:
        text_pipe = pipeline('text-generation', model='daspartho/prompt-extend')

    # hpsv2 score #################################################################
    from typing import Union
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    hps_model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device="cuda",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    checkpoint = torch.load("./hpsv2/HPS_v2_compressed.pt", map_location="cpu")
    hps_model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    hps_model = hps_model.to("cuda")
    hps_model.eval()

    pipe_textcraftor.unet.eval()
    pipe_textcraftor.vae.eval()
    pipe_textcraftor.text_encoder.eval()

    pipe_textcraftor = pipe_textcraftor.to('cuda')
    pipe_textcraftor.text_encoder_origin = pipe_textcraftor.text_encoder_origin.to("cuda")

    os.makedirs("tests", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts('all')

    # Iterate over the benchmark prompts to generate images
    for style, prompts in all_prompts.items():
        os.makedirs(os.path.join(output_path, style), exist_ok=True)
        for idx, prompt in enumerate(prompts):
            # prompt engineering
            if args.prompt_eng:
                enhanced_prompts = []
                extended_prompt = text_pipe(prompt, num_return_sequences=1)
                for item in extended_prompt:
                    enhanced_prompts.append(item["generated_text"])
                print("finish prompt enhancement:", len(enhanced_prompts))
                prompt = enhanced_prompts

            output = pipe_textcraftor(prompt, num_inference_steps=args.step, guidance_scale=guidance,
                                      num_images_per_prompt=1,
                                      generator=generator_s,
                                      grad_steps=args.grad_steps,
                                      prompt_weighting=args.prompt_weighting,
                                      )
            image = output.images[0]
            # TextToImageModel is the model you want to evaluate
            image.save(os.path.join(output_path, style, f"{idx:05d}.jpg"))
            # <image_path> is the folder path to store generated images, as the input of hpsv2.evaluate().

    hpsv2.evaluate(output_path)
    exit()


if __name__ == '__main__':
    parser = parser_spec()
    args = parser.parse_args()

    main(args)
