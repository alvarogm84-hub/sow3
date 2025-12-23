OV-DINO Fine-Tuning on the Milestone Dataset

This repository provides instructions for installing, preparing data, fine-tuning, and evaluating OV-DINO on a custom dataset (Milestone) using the COCO format.

Table of Contents

Project Structure

Installation

Data Preparation

Pretrained Model

Fine-Tuning

Evaluation

Project Structure
OV-DINO
├── configs
├── datas
│   └── milestone
│       ├── annotations
│       ├── train
│       ├── val
│       └── test
├── demo
├── detectron2-717ab9
├── detrex
├── docs
├── inits
│   └── ovdino
├── ovdino
└── wkdrs

Installation
1. Clone the Repository
git clone https://github.com/alvarogm84-hub/sow3.git
cd OV-DINO
export ROOT_DIR=$(realpath .)
cd $ROOT_DIR/ovdino

2. CUDA Setup (CUDA 11.6 Required)

OV-DINO is built and tested with CUDA 11.6.
If your system CUDA version differs, explicitly set CUDA_HOME:

export CUDA_HOME=/path/to/cuda-11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc -V

3. Create Conda Environment
conda create -n ovdino -y
conda activate ovdino

4. Install Dependencies
PyTorch (CUDA 11.6)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --index-url https://download.pytorch.org/whl/cu116

Optional: GCC 9 (Recommended for Compilation)
conda install gcc=9 gxx=9 -c conda-forge -y

Install Detectron2 and OV-DINO
python -m pip install -e detectron2-717ab9
pip install -e ./

Fix NumPy Compatibility
pip uninstall -y numpy
pip install "numpy<1.25"

Data Preparation
Milestone Dataset Format

Convert your dataset from YOLO format to COCO format using a YOLO-to-COCO converter.

Expected directory structure:

milestone
├── annotations
├── train
├── val
└── test

Configure Dataset Paths

Update the dataset paths in:

ovdino/configs/common/data/milestone_big_ovd.py


Ensure that train, val, and test paths correctly point to your Milestone dataset directories.

Adjust Batch Size

Modify batch size based on your GPU memory in:

ovdino/projects/ovdino/configs/
ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep.py


Example:

# Recommended values: 64 / 32 / 16 / 8 / 4
dataloader.train.total_batch_size = 4

Pretrained Model

Download the official OV-DINO pretrained checkpoint:

https://huggingface.co/hao9610/OV-DINO/resolve/main/
ovdino_swint_og-coco50.6_lvismv39.4_lvis32.2.pth


Place it in:

inits/ovdino/

Fine-Tuning
Fine-Tuning on the Milestone Dataset

Example training command:

python ./tools/train_net.py \
  --config-file ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep.py \
  --resume \
  train.init_checkpoint=inits/ovdino/ovdino_swint_og-coco50.6_lvismv39.4_lvis32.2.pth \
  train.output_dir=wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep \
  dataloader.evaluator.output_dir=wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep/eval_milestone_big_YYYYMMDD-HHMMSS

Evaluation
Evaluating a Fine-Tuned Model

Select the best checkpoint based on validation metrics in the training logs
(e.g., model_0009999.pth).

python ./tools/train_net.py \
  --config-file ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep.py \
  --eval-only \
  --resume \
  train.init_checkpoint=wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep/model_0009999.pth \
  train.output_dir=wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep \
  dataloader.evaluator.output_dir=wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep/eval_mil
