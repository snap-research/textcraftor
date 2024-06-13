export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 train_textcraftor.py --batch-size 4 \
--clip 0.5 --aesthetic 1. --pickscore 1. --hpsv2 1. \
--grad_steps 5 \
--lr_text 1e-6 --lr_unet 0. \
--output-path outputs/code-test