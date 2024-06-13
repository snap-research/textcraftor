export CUDA_VISIBLE_DEVICES=0
python test_parti.py --grad_steps 15 \
--scheduler DDIM --step 25 \
--text_path checkpoints/tc_text_e_0_iter_10000.pth \
--output_path ./tests/paper_ckpt_parti
