<h2 align="center">
  <b>Trace3D: Consistent Segmentation Lifting via</b>
  <br>
  <b>Gaussian Instance Tracing</b>

  <b><i>ICCV 2025 </i></b>
</h2>

<p align="center">
    <a href="https://github.com/shy2000">Hongyu Shen </a><sup>1,2,*</sup>
    <a href="https://dali-jack.github.io/Junfeng-Ni/">Junfeng Ni</a><sup>2,3,*</sup>,
    <a href="https://yixchen.github.io/">Yixin Chen<sup>✉</sup></a><sup>2</sup>,
    <a href="">Weishuo Li</a><sup>2</sup>,
    <a href="https://peimingtao.github.io/">Mingtao Pei</a><sup>1</sup>,
    <a href="https://siyuanhuang.com/">Siyuan Huang<sup>✉</sup></a><sup>2</sup>
    <br>
    <a style="font-size: 0.9em; padding: 0.5em 0;"><sup>✉</sup> indicates corresponding author</a> &nbsp&nbsp 
  <a style="font-size: 0.9em; padding: 0.5em 0;"><sup>*</sup> these authors contributed equally to this work</a> &nbsp&nbsp 
    <sup>1</sup>Beijing Institute of Technology
    <br>
    <sup>2</sup>State Key Laboratory of General Artificial Intelligence, BIGAI &nbsp&nbsp 
    <sup>3</sup>Tsinghua University
</p>

<p align="center">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Video-green?style=plastic&logo=arXiv&logoColor=green' alt='Video Demo'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

<p align="center">
    <img src="assets/teaser.jpg" width=90%>
</p>

Trace3D leverages the proposed Gaussian Instance Tracing to enhance multi-view consistency and reduce ambiguous Gaussians, resulting in high-quality 3D instance segmentation.

## Installation
- Tested System: Ubuntu 22.04, CUDA 11.8
- Tested GPUs: RTX4090

1. Basic environment
```bash
conda create -n trace3d python=3.10
conda activate trace3d
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
2. SAM for segmentation
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
## Data
```bash
    data
    ├── nerf_llff_data  # Link: https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7
    │   └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
    │       ├── [sparse/0] (colmap results)
    │       └── [images|images_2|images_4|images_8]
    │
    └── replica		# Link: https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0
        └── [office_0|room_0|...]
            ├── traj_w_c.txt
            ├── [sparse/0] (colmap results)
            └── [rgb|depth|sam|]
```
## Training
Get SAM masks
```bash
python get_sam_masks.py --sam_checkpoint {SAM_CKPT_PATH} --file_path {IMAGE_FOLDER}
```
Before running: please specify the information in the scripts (e.g. `replica.sh`). More options can be found in `conf/` and `arguments/` and you can them adjusted in config file.
```bash
#--- Edit the config file replica.sh
dataset=replica_900
path=./data/${dataset}
scene='room_0' 
```
Scene reconstruction
```bash
bash replica.sh train_rgb
```

Merge patch masks 
```bash
bash replica.sh merge_patches
```
Delete Ambiguous Gaussians
```bash
bash replica.sh remove_ab_gaus
```
Contrastive lifting
```bash
bash replica.sh train_contra
```
<!-- ### 2.Merge patch masks  

### 3.Delete Ambiguous Gaussians -->


## Evaluation
3D Object Extraction
```bash
bash replica.sh eval_3d
```
Novel View 2D Instance Segmentation
```bash
bash replica.sh eval         
```

## Acknowledgements
Some codes are borrowed from  [Egolifter](https://github.com/facebookresearch/egolifter), [SA3D](https://github.com/Jumpat/SegmentAnythingin3D), [Omniseg3D](https://github.com/THU-luvision/OmniSeg3D), [FlashSplat](https://github.com/florinshen/FlashSplat) and [Gaussian-Editor](https://github.com/buaacyw/GaussianEditor). We thank all the authors for their great work. 

## Citation

```bibtex
@inproceedings{shen2025trace3d,
  title={Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing},
  author={Shen, Hongyu and Ni, Junfeng, and Chen, Yixin and Li, Weishuo and Pei, Mingtao and Huang, Siyuan},
  booktitle=ICCV,
  year={2025}
}
```