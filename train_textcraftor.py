import os
import argparse
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import numpy as np

from pipelines.pipeline_stable_diffusion import StableDiffusionPipelineTC
from pipelines.scheduling_ddim import DDIMScheduler

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0, Attention

from aesthetic import load_models
import open_clip

import csv
import json
from pprint import pprint
import pandas as pd
import logging
from PIL import Image


class CsvDataset(Dataset):
    def __init__(self,
                 input_filename='openprompts.csv',
                 ):
        print("Reading data from {}, this may take a while!".format(input_filename))
        self.df = pd.read_csv(input_filename, keep_default_na=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.df.iloc[idx, 0]
        return prompt


def set_processors(attentions):
    for attn in attentions:
        attn.set_processor(AttnProcessor2_0())


def set_torch_2_attn(unet):
    optim_count = 0
    for name, module in unet.named_modules():
        if "attn1" or "attn2" == name.split(".")[-1]:
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def standard_process(image):
    output = torch.nn.functional.adaptive_avg_pool2d(image, 224)
    output = torchvision.transforms.functional.normalize(output,
                                                         (0.48145466, 0.4578275, 0.40821073),
                                                         (0.26862954, 0.26130258, 0.27577711), )
    return output


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


def parse():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                        metavar='LR', help='Initial learning rate')
    parser.add_argument('--lr_unet', default=1e-6, type=float,
                        help='unet learning rate')
    parser.add_argument('--lr_text', default=1e-6, type=float,
                        help='text learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--output-path', default='dummy', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--hpsv2', default=1., type=float, )
    parser.add_argument('--pickscore', default=1., type=float, )
    parser.add_argument('--clip', default=1., type=float, )
    parser.add_argument('--aesthetic', default=1., type=float, )
    parser.add_argument('--grad_steps', default=5, type=int,
                        help='truncate backpropagation')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling sync BN.')
    args = parser.parse_args()
    return args


def main():
    args = parse()
    prompt = "a photo of an astronaut riding a horse on mars"
    pick_reward = args.pickscore > 0
    hps_reward = args.hpsv2 > 0
    clip_reward = args.clip > 0
    aes_reward = args.aesthetic > 0

    # Enable tensor-core
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cudnn.benchmark = True
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    # #################### data and sampler ###################################################
    open_prompt_data = CsvDataset()

    train_sampler = None
    # val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(open_prompt_data)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    dataloader = DataLoader(open_prompt_data, batch_size=args.batch_size,
                            shuffle=(train_sampler is None), num_workers=args.workers,
                            pin_memory=True, sampler=train_sampler)

    if args.local_rank == 0:
        os.makedirs("outputs", exist_ok=True)
        os.makedirs(args.output_path, exist_ok=True)

    # ########################### model creation stuff ############################################
    generator_s = torch.Generator("cuda").manual_seed(93)
    pipe_textcraftor = StableDiffusionPipelineTC.from_pretrained("./stable-diffusion-v1-5")

    if hasattr(F, "scaled_dot_product_attention"):
        set_torch_2_attn(pipe_textcraftor.unet)

    scheduler_config = pipe_textcraftor.scheduler.config
    # scheduler_config['prediction_type'] = 'v_prediction'
    # scheduler_config['prediction_type'] = 'sample'
    scheduler_config['prediction_type'] = 'epsilon'
    noise_scheduler = DDIMScheduler.from_config(scheduler_config)
    noise_scheduler.set_timesteps(25)
    pipe_textcraftor.scheduler = noise_scheduler
    print("prediction type: ", noise_scheduler.config.prediction_type)

    if clip_reward:
        clip_model, *_ = open_clip.create_model_and_transforms(
            'ViT-g-14',
            pretrained='laion2b_s34b_b88k',
        )
        clip_model = clip_model.to("cuda")
    if aes_reward:
        model_aes = load_models()
    if pick_reward:
        from transformers import AutoProcessor, AutoModel
        # load model
        pick_model = AutoModel.from_pretrained("./pickscore/pickmodel").eval().to("cuda")
    if hps_reward:
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
        checkpoint = torch.load("hpsv2/HPS_v2_compressed.pt", map_location="cpu")
        hps_model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer('ViT-H-14')
        hps_model = hps_model.to("cuda")
        hps_model.eval()
    # ########################### end model creation stuff ############################################

    # pipe_teacher.text_encoder = pipe_teacher.text_encoder.to("cuda")
    pipe_textcraftor.__setattr__('text_encoder_origin', copy.deepcopy(pipe_textcraftor.text_encoder))
    pipe_textcraftor = pipe_textcraftor.to("cuda")
    pipe_textcraftor.text_encoder_origin = pipe_textcraftor.text_encoder_origin.to("cuda")

    if args.lr_text > 0.:
        pipe_textcraftor.text_encoder = torch.nn.parallel.DistributedDataParallel(pipe_textcraftor.text_encoder,
                                                                                  device_ids=[args.gpu],
                                                                                  broadcast_buffers=False)
        pipe_textcraftor.text_encoder.train()
    else:
        pipe_textcraftor.text_encoder.eval()

    if args.lr_unet > 0.:
        pipe_textcraftor.unet = torch.nn.parallel.DistributedDataParallel(pipe_textcraftor.unet,
                                                                          device_ids=[args.gpu],
                                                                          broadcast_buffers=False)
        pipe_textcraftor.unet.train()
    else:
        pipe_textcraftor.unet.eval()

    # ############ auto resume in case the training is interrupted ###############
    try:
        pipe_textcraftor.text_encoder.load_state_dict(torch.load(os.path.join(args.output_path, 'student_text.pth'),
                                                                 map_location='cpu'),
                                                      strict=True)

        print('##### Restarting, loaded latest Text ckpt from current directory ######')
    except:
        print('!!!!! relaunch without any pretrain Text, dangerous !!!!!')
    try:
        pipe_textcraftor.unet.load_state_dict(torch.load(os.path.join(args.output_path, 'student_unet.pth'),
                                                         map_location='cpu'),
                                              strict=True)

        print('##### Restarting, loaded latest UNet ckpt from current directory #####')
    except:
        print('!!!!! relaunch without any pretrain UNet, dangerous !!!!!')

    pipe_textcraftor.vae.eval()
    pipe_textcraftor.text_encoder_origin.eval()

    pipe_textcraftor.vae.requires_grad_(False)

    meters = {}
    if clip_reward:
        meters['loss_clip_meter'] = AverageMeter()
    if aes_reward:
        meters['loss_aes_meter'] = AverageMeter()
    if hps_reward:
        meters['loss_hps_meter'] = AverageMeter()
    if pick_reward:
        meters['loss_pick_meter'] = AverageMeter()

    optimizer_params = []
    if args.lr_unet > 0.:
        optimizer_params.append({'params': pipe_textcraftor.unet.parameters(), 'lr': args.lr_unet})
    if args.lr_text > 0.:
        optimizer_params.append({'params': pipe_textcraftor.text_encoder.parameters(), 'lr': args.lr_text})
    optimizer = optim.AdamW(optimizer_params)

    scaler = torch.cuda.amp.GradScaler()

    # ############ Training starts !!! ###############
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print('epoch: ', epoch)
        for mini_batch, row in enumerate(dataloader):
            data = row
            latents = torch.randn(len(data), 4, 64, 64).cuda()

            with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
                prompt_embeds_s, text_input_ids = pipe_textcraftor._encode_prompt(data, args.gpu, 1,
                                                                                  do_classifier_free_guidance=True,
                                                                                  get_text_inputs=True, )
                with torch.no_grad():
                    prompt_embeds_t = pipe_textcraftor._encode_prompt(data, args.gpu, 1,
                                                                      do_classifier_free_guidance=True,
                                                                      use_origin=True)

                extra_step_kwargs = pipe_textcraftor.prepare_extra_step_kwargs(generator_s, 0.0)
                latents = pipe_textcraftor.loop_unet_cus(latents, prompt_embeds_s,
                                                         extra_step_kwargs,
                                                         guidance_scale=7.5,
                                                         do_classifier_free_guidance=True,
                                                         freeze_prompt_embeds=prompt_embeds_t,
                                                         grad_steps=args.grad_steps,
                                                         )
                image = pipe_textcraftor.decode_latents(latents)

                losses = []
                # clip constraint ###################################################################################
                if clip_reward:
                    image_clip = standard_process(image)
                    image_clip_features = clip_model.encode_image(image_clip)
                    with torch.no_grad():
                        text_features = clip_model.encode_text(text_input_ids.to(image_clip.device))
                    loss_clip = -100.0 * args.clip * torch.mean(
                        torch.sum((image_clip_features / image_clip_features.norm(dim=-1, keepdim=True)) *
                                  (text_features / text_features.norm(dim=-1, keepdim=True)), dim=1))
                    losses.append(loss_clip)
                    meters['loss_clip_meter'].update(loss_clip.item() / (-100.0 * args.clip))

                # pick score loss #################################################################################
                if pick_reward:
                    image_pick = standard_process(image)
                    image_embs = pick_model.get_image_features(pixel_values=image_pick)
                    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                    with torch.no_grad():
                        text_embs = pick_model.get_text_features(input_ids=text_input_ids.to("cuda"))
                        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                    # score
                    scores = pick_model.logit_scale.exp() * (text_embs @ image_embs.T)
                    loss_pick = -1.0 * args.pickscore * torch.mean(scores)
                    losses.append(loss_pick)
                    meters['loss_pick_meter'].update(loss_pick.item() / (-1.0 * args.pickscore))

                # hpsv2 score loss ################################################################################
                if hps_reward:
                    image_hps = standard_process(image)
                    outputs = hps_model(image_hps, text_input_ids.to("cuda"))
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                    loss_hps = -100.0 * args.hpsv2 * torch.mean(hps_score)
                    losses.append(loss_hps)
                    meters['loss_hps_meter'].update(loss_hps.item() / (-100.0 * args.hpsv2))

                # aesthetic score loss ############################################################################
                if aes_reward:
                    image_aes = standard_process(image)
                    image_features = model_aes['clip_model'].encode_image(image_aes)
                    im_emb = image_features / torch.linalg.norm(image_features, ord=2, dim=-1, keepdim=True)
                    prediction = model_aes['classifier'](im_emb)
                    loss_aes = -3.0 * args.aesthetic * torch.mean(prediction)
                    losses.append(loss_aes)
                    meters['loss_aes_meter'].update(loss_aes.item() / (-3.0 * args.aesthetic))
                # #################################################################################################

            loss = sum(losses)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            """ >>> gradient clipping >>> """
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(pipeline.text_encoder.parameters(), 0.1)
            """ <<< gradient clipping <<< """
            scaler.step(optimizer)
            scaler.update()

            # ##################### rest is logging, validation and saving stuff ##################################
            if mini_batch % 10 == 0 and args.local_rank == 0:
                print('iteration: ', mini_batch, "rewards: ",
                      [meters[item].avg for item in meters])

            if mini_batch % 100 == 0 and args.local_rank == 0:
                pipe_textcraftor.text_encoder.eval()
                pipe_textcraftor.unet.eval()
                image = pipe_textcraftor(prompt, num_inference_steps=25, generator=generator_s,
                                         guidance_scale=7.5,
                                         # freeze_prompt_embeds=example_prompt_emb_t,
                                         grad_steps=15,
                                         ).images[0]
                if args.lr_text > 0.:
                    pipe_textcraftor.text_encoder.train()
                if args.lr_unet > 0.:
                    pipe_textcraftor.unet.train()
                image.save(
                    os.path.join(args.output_path, "astronaut_horse_" + str(epoch) +
                                 "_iter_" + str(mini_batch) + ".png"))

                if mini_batch % 500 == 0:
                    if args.lr_text > 0.:
                        torch.save(pipe_textcraftor.text_encoder.state_dict(),
                                   os.path.join(args.output_path, 'tc_text.pth'))
                    if args.lr_unet > 0.:
                        torch.save(pipe_textcraftor.unet.state_dict(),
                                   os.path.join(args.output_path, 'tc_unet.pth'))

                if mini_batch % 5000 == 0:
                    if args.lr_text > 0.:
                        torch.save(pipe_textcraftor.text_encoder.state_dict(),
                                   os.path.join(args.output_path,
                                                'tc_text_e_' + str(epoch) + '_iter_' + str(
                                                    mini_batch) + '.pth'))
                    if args.lr_unet > 0.:
                        torch.save(pipe_textcraftor.unet.state_dict(),
                                   os.path.join(args.output_path,
                                                'tc_unet_e_' + str(epoch) + '_iter_' + str(
                                                    mini_batch) + '.pth'))

        for meter in meters:
            meters[meter].reset()


if __name__ == '__main__':
    main()
