#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    exit 1
fi

GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python -m scripts.animate --exp_config configs/prompts/3-DreamShaper.yaml --H 1024 --W 1024 --L 16 --xformers \
--instantid True --face_dir /home/nas4_user/taewoongkang/repos/2d/AnimateDiff/examples/images.jpeg 

CUDA_VISIBLE_DEVICES=$GPU_ID python -m scripts.animate --exp_config configs/prompts/3-DreamShaper.yaml --H 1024 --W 1024 --L 16 --xformers \
--instantid True --face_dir /home/nas4_user/taewoongkang/repos/2d/AnimateDiff/examples/kaifu_resize.png

CUDA_VISIBLE_DEVICES=$GPU_ID python -m scripts.animate --exp_config configs/prompts/3-DreamShaper.yaml --H 1024 --W 1024 --L 16 --xformers \
--instantid True --face_dir /home/nas4_user/taewoongkang/repos/2d/AnimateDiff/examples/sam_resize.png

CUDA_VISIBLE_DEVICES=$GPU_ID python -m scripts.animate --exp_config configs/prompts/3-DreamShaper.yaml --H 1024 --W 1024 --L 16 --xformers \
--instantid True --face_dir /home/nas4_user/taewoongkang/repos/2d/AnimateDiff/examples/schmidhuber_resize.png

# CUDA_VISIBLE_DEVICES=$GPU_ID python -m scripts.animate --exp_config configs/prompts/3-DreamShaper.yaml --H 1024 --W 1024 --L 16 --xformers \
# --instantid True --face_dir /home/nas4_user/taewoongkang/repos/2d/AnimateDiff/examples/Ye.png

