## TextCraftor<br><sub>Your Text Encoder Can be Image Quality Controller</sub>
[Webpage](https://snap-research.github.io/textcraftor/) | [arXiv](https://arxiv.org/abs/2403.18978)

## Prerequisites

### Installation

`conda` virtual environment with Python 3.8+, PyTorch 2.0+ is recommended. In your venv, run 
```
pip install -r requirements.txt
```

[//]: # (Please note that we make external libraries &#40;HPSv2, PickScore, Aesthetic score&#41; as built-in functions, so they are installation-free. )

### Data preparation

This repo provides the script of prompt-based finetuning, so only prompt data is needed for training. In this work we use [open-prompts by krea-ai](https://github.com/krea-ai/open-prompts), which can be downloaded [here](https://github.com/krea-ai/open-prompts?tab=readme-ov-file#csv-file). The data (`openprompts.csv`) takes a few GBs in disk. 

For testing (Parti-Prompts and HPSv2 benchmark), the data will be gathered on-the-fly, no preparation is needed. See Testing section below. 

### Download necessary models
Due to GitHub file size limit, we do not hold model weights in this repo. You need to
1. Follow instructions [here](https://huggingface.co/docs/diffusers/quicktour#local-pipeline) to download Stable Diffusion v1.5. You can soft-link your model into this directory if you already have one. It should appear in this directory as `./stable-diffusion-v1-5`.
2. Download [HPSv2 weights](https://drive.google.com/file/d/1T4e6WqsS5lcs92HdmzQYonrfDH1Ub53T/view?usp=sharing) and put it here: `hpsv2/HPS_v2_compressed.pt`. 
3. Download [PickScore model weights](https://drive.google.com/file/d/1UhR0zFXiEI-spt2QdX67FY9a0dcqa9xy/view?usp=sharing) and put it here: `pickscore/pickmodel/model.safetensors`. 

### Double check if everything is all set
```
|-- textcraftor/
    |-- hpsv2
        |-- HPS_v2_compressed.pt
    |-- pickscore
        |-- pickmodel
            |-- config.json
            |-- model.safetensors
    |-- stable-diffusion-v1-5
    |-- openprompts.csv
    |-- ...
```

## Training 
You can simply train TextCraftor with the following script: 
```
sh run_textcraftor.sh
```
As discussed in the paper, you have a lot of freedom choosing the reward combinations, 
and different combinations lead to different styles. 
```
--clip 0.5 --aesthetic 1. --pickscore 1. --hpsv2 1.
```
By default, we backpropagate through the last 5 steps and test the finetuned text encoder on the last 15 steps, as discussed in Section.4.2. 

In addition, Unet finetuning is also integrated in this code base. You can switch to Unet finetuning by simply swapping the learning rate and freeze the text encoder:
```
--lr_text 0. --lr_unet 1e-6
```
Please note that to avoid domain shift and preserve the generalization property, it is not recommended to finetune text encoder and Unet together: 
```
--lr_text 1e-6 --lr_unet 1e-6
```
though there would not be any bugs preventing you doing so. The recommended way is to finetune a text encoder first, 
then load and freeze the text model and perform TextCraftor+Unet finetuning. 

Based on our observations, on an 8 x A100 80G node, 
1. You can use `--batch-size 4` per GPU for TextCraftor training. 
2. You can use `--batch-size 1` per GPU for Unet finetuning. 
3. You can observe score and visual improvements in 500-1000 iterations. 


## Testing 
Taking the trained text encoder in our paper as an example (Table 1&2), 
download [our checkpoint](https://drive.google.com/file/d/1CoiFGD60AZiDV_JlXRlxf-lLaHHNtVfp/view?usp=sharing) and put it here: `checkpoints/tc_text_e_0_iter_10000.pth`. 
Then you can test on PartiPrompts by the following script: 
```
sh test_parti.sh
```
or test on HPSv2 benchmark by the following script: 
```
sh test_hpsv2.sh
```
The final scores will be logged when generation is finished, and generated images will be saved at `tests/`. 

## Acknowledgement
We thank [runwayml and Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5) for open sourcing and maintaining Stable Diffusion models. 

We thank [HPSv2](https://github.com/tgxs002/HPSv2), [PickScore](https://github.com/yuvalkirstain/PickScore) and [Aesthetic](https://laion.ai/blog/laion-aesthetics/) for the reward models. 

## Citation
If you find this work helpful, please cite our paper:
```BibTeX
@article{li2024textcraftor,
  title={TextCraftor: Your Text Encoder Can be Image Quality Controller},
  author={Li, Yanyu and Liu, Xian and Kag, Anil and Hu, Ju and Idelbayev, Yerlan and Sagar, Dhritiman and Wang, Yanzhi and Tulyakov, Sergey and Ren, Jian},
  journal={arXiv preprint arXiv:2403.18978},
  year={2024}
}
```