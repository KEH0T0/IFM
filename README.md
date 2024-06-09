
# Implementation of IFM

## Inference Example

Inference at recommended resolution of 16 frames usually requires ~13GB VRAM.
### Step-1: Prepare Environment

```
git clone https://github.com/KEH0T0/IFM.git
cd IFM
git checkout sdxl


conda env create -f environment.yaml
conda activate animatediff_xl
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install xformers==0.0.22.post4+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Step-2: Download Base T2I & Motion Module Checkpoints
We provide a beta version of motion module on SDXL. You can download the base model of SDXL 1.0 and Motion Module following instructions below.
```
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 models/StableDiffusion/

bash download_bashscripts/0-MotionModule.sh
```
You may also directly download the motion module checkpoints from [Google Drive](https://drive.google.com/file/d/1EK_D9hDOPfJdK4z8YDB8JYvPracNx2SX/view?usp=share_link
) / [HuggingFace](https://huggingface.co/guoyww/animatediff/blob/main/mm_sdxl_v10_beta.ckpt
) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules), then put them in `models/Motion_Module/` folder.

###  Step-3: Download Personalized SDXL (you can skip this if generating videos on the original SDXL)
You may run the following bash scripts to download the LoRA checkpoint from CivitAI.
```
bash download_bashscripts/1-DynaVision.sh
bash download_bashscripts/2-DreamShaper.sh
bash download_bashscripts/3-DeepBlue.sh
```
You also can download the InstantID model in python script:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="InstantX/InstantID", filename="checkpoints/ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="checkpoints/ip-adapter.bin", local_dir="./checkpoints")
```

### Step-4: Generate Videos
Run the following commands to generate videos of **original SDXL**. 
```
python -m scripts.animate --exp_config configs/prompts/1-original_sdxl.yaml --H 1024 --W 1024 --L 16 --xformers \
--instantid True --face_dir /home/nas4_user/taewoongkang/repos/2d/AnimateDiff/examples/images.jpeg 
```
or
```
./test.sh 1 
```
Run the following commands to generate videos of **personalized SDXL**. DO NOT skip Step-3.
```
python -m scripts.animate --config configs/prompts/2-DynaVision.yaml --H 1024 --W 1024 --L 16 --xformers
python -m scripts.animate --config configs/prompts/3-DreamShaper.yaml --H 1024 --W 1024 --L 16 --xformers
python -m scripts.animate --config configs/prompts/4-DeepBlue.yaml --H 1024 --W 1024 --L 16 --xformers
```
or
```
./test_all.sh
```
The results will automatically be saved to `sample/` folder.

**Before generate videos, make sure that changing the config file with your saved "motion_module_path", "pretrained_model_path".**

**Make sure that the "face_adapter_path" and "controlnet_path" changed with your saved path.**

## Model Zoo
<details open>
<summary>Motion Modules</summary>

  | Name                 | Parameter | Storage Space |
  |----------------------|-----------|---------------|
  | mm_sdxl_v10_beta.ckpt      | 238 M     | 0.9 GB        |

</details>

<details open>
<summary>Recommended Resolution</summary>

  | Resolution                 | Aspect Ratio | 
  |----------------------|-----------|
  | 768x1344      | 9:16     |
  | 832x1216      | 2:3     |
  | 1024x1024     | 1:1     |
  | 1216x832      | 3:2     |
  | 1344x768      | 16:9     |

</details>
