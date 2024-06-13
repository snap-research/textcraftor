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


def standard_process(image):
    output = torch.nn.functional.adaptive_avg_pool2d(image, 224)
    output = torchvision.transforms.functional.normalize(output,
                                                         (0.48145466, 0.4578275, 0.40821073),
                                                         (0.26862954, 0.26130258, 0.27577711), )
    return output


def parser_spec():
    parser = argparse.ArgumentParser()
    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument(
        "--ckpt_path",
        default="/data_laion/yli12/paper_ckpts/student_unet_cfgloss_getty.pth",
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
        default="tests/paper_ckpt_parti",
        help=
        ("output path for saving images"
         ))

    parser.add_argument('--seed', default=2023, type=int, help='seed')
    parser.add_argument('--guidance', default=7.5, type=float, help='cfg guidance')
    parser.add_argument('--step', default=25, type=int, help='numer of steps')
    parser.add_argument('--grad_steps', default=25, type=int, help='numer of text aug steps')
    parser.add_argument('--prompt_weighting', default=1.0, type=float, help='prompt weighting')
    parser.add_argument('--scheduler', default='DDIM', type=str, choices=['DDIM', 'DPMSingle', 'Eulr', "PNDM"],
                        help='scheduler')
    parser.add_argument('--prompts', default='./gen_back_500_prompts.csv', type=str, help='path for prompts')
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

    # human parti prompts
    from datasets import load_dataset
    data_prompts = load_dataset("nateraw/parti-prompts")
    prompts = []
    for i in range(len(data_prompts["train"])):
        # if data_prompts["train"]["Category"][i] == "People":
        prompts.append(data_prompts["train"]["Prompt"][i])
    print('total benchmark prompts: ', len(prompts))

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
    elif args.scheduler == "PNDM":
        noise_scheduler = pipe_textcraftor.scheduler

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

    # score models ######################################################################
    import open_clip
    clip_model, *_ = open_clip.create_model_and_transforms(
        'ViT-g-14',
        pretrained='laion2b_s34b_b88k',
    )
    clip_model = clip_model.to("cuda")

    # aes score ############################################
    from aesthetic import load_models
    model_aes = load_models()

    # pick score##############################################################
    from transformers import AutoProcessor, AutoModel
    pick_model = AutoModel.from_pretrained("./pickscore/pickmodel").eval().to("cuda")

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

    img_per_inf = 4
    clip_meter = AverageMeter()
    aes_meter = AverageMeter()
    pick_meter = AverageMeter()
    hps_meter = AverageMeter()

    for i in range(len(prompts) // img_per_inf + int(len(prompts) % img_per_inf > 0)):
        print("generating batch:", i)
        try:
            prompt_chunk = prompts[(img_per_inf * i):(img_per_inf * (i + 1))]
        except:
            prompt_chunk = prompts[(img_per_inf * i):]
        # prompt_enhanced_chunk = enhanced_prompts[(img_per_inf * i):(img_per_inf * (i + 1))]
        n_prompts = len(prompt_chunk)

        # prompt engineering
        if args.prompt_eng:
            enhanced_prompts = []
            extended_prompt = text_pipe(prompt_chunk, num_return_sequences=1)
            for item in extended_prompt:
                enhanced_prompts.append(item[0]["generated_text"])
            print("finish prompt enhancement:", len(enhanced_prompts))
            prompt_chunk = enhanced_prompts

        output = pipe_textcraftor(prompt_chunk, num_inference_steps=args.step, guidance_scale=guidance,
                                  num_images_per_prompt=1,
                                  generator=generator_s,
                                  grad_steps=args.grad_steps,
                                  prompt_weighting=args.prompt_weighting,
                                  )
        images = output.images
        decoder_out = output.decoder_out

        for j, image in enumerate(images):
            image.save(output_path + "{:03d}".format(img_per_inf * i + j) + ".png")

        with torch.no_grad():
            image_score = standard_process(decoder_out)
            # clip score loss  ##################################################################################
            image_clip_features = clip_model.encode_image(image_score)
            text_inputs = pipe_textcraftor.tokenizer(
                prompt_chunk,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_features_clip = clip_model.encode_text(text_input_ids.cuda())
            clip_scores = torch.sum((image_clip_features / image_clip_features.norm(dim=-1, keepdim=True)) *
                                    (text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)), dim=1)
            # print("clip scores: ", clip_scores)
            clip_meter.update(clip_scores.mean().item(), n=n_prompts)
            print("Average clip scores: ", clip_meter.avg)

            # print aesthetic scores #########################################################################
            image_features_aes = model_aes['clip_model'].encode_image(image_score)
            im_emb = image_features_aes / torch.linalg.norm(image_features_aes, ord=2, dim=-1, keepdim=True)
            prediction = model_aes['classifier'](im_emb)
            # print("aes scores: ", prediction)
            aes_meter.update(prediction.mean().item(), n=n_prompts)
            print("Average aes scores: ", aes_meter.avg)

            # pick score #####################################################################################
            image_embs = pick_model.get_image_features(pixel_values=image_score)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = pick_model.get_text_features(input_ids=text_input_ids.to("cuda"))
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores = pick_model.logit_scale.exp() * (text_embs @ image_embs.T)
            pick_meter.update(scores.mean().item(), n=n_prompts)
            print("Average pick scores: ", pick_meter.avg)

            # hps score ######################################################################################
            outputs = hps_model(image_score, text_input_ids.to("cuda"))
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image)
            hps_meter.update(hps_score.mean().item(), n=n_prompts)
            print("Average hps scores: ", hps_meter.avg)

    print('finish generation!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print("Average clip scores: ", clip_meter.avg)
    print("Average aes scores: ", aes_meter.avg)
    print("Average pick scores: ", pick_meter.avg)
    print("Average hpsv2 scores: ", hps_meter.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    parser = parser_spec()
    args = parser.parse_args()

    main(args)
