
### 1. Project Structure
```
OV-DINO
├── configs
├── datas
│   ├── milestone
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   │   └── test
├── demo
├── detectron2-717ab9
├── detrex
├── docs
├── inits
├── ovdino
├── wkdrs

```

### 2. Installation
```bash
# clone this project
git clone thttps://github.com/alvarogm84-hub/sow3.git
cd OV-DINO
export root_dir=$(realpath ./)
cd $root_dir/ovdino

# Optional: set CUDA_HOME for cuda11.6.
# OV-DINO utilizes the cuda11.6 default, if your cuda is not cuda11.6, you need first export CUDA_HOME env manually.
export CUDA_HOME="your_cuda11.6_path"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo -e "$log_format cuda version:\n$(nvcc -V)"

# create conda env for ov-dino
conda create -n ovdino -y
conda activate ovdino
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
conda install gcc=9 gxx=9 -c conda-forge -y # Optional: install gcc9
python -m pip install -e detectron2-717ab9
pip install -e ./

pip uninstall -y numpy
pip install "numpy<1.25"

### 3. Data Preparing
#### Milestone
Follow Yolo-to-COCO-format-converter instructions in order to prepare Milestone dataset in the COCO format:
```
OV-DINO
├── datas
│   ├── milestone
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   │   └── test
```
#### Set routes (train/val/test) to Milestone dataset in
-OV-DINO-main/ovdino/configs/common/data/milestone_big_ovd.py

#### Modify the batch size accordding to GPU capabilities in
-OV-DINO-main/ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep.py
# dataloader.train.total_batch_size = 4 (64/32/16/8/4)

#### Pretrained Model 
Download model: https://huggingface.co/hao9610/OV-DINO/resolve/main/ovdino_swint_og-#coco50.6_lvismv39.4_lvis32.2.pth

And put it on "inits/ovdino directory"

### 4. Fine-Tuning
#### Fine-Tuning on Custom Dataset (Milestone Dataset)
(Example) python ./tools/train_net.py --config-file /..../OV-DINO-main/ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep.py --resume train.init_checkpoint=/..../OV-DINO-main/inits/ovdino/ovdino_swint_og-coco50.6_lvismv39.4_lvis32.2.pth train.output_dir=/..../OV-DINO-main/wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep dataloader.evaluator.output_dir="/..../OV-DINO-main/wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep/eval_milestone_big_2025mmdd-HHMMSS" 


### 5. Evaluate Fine-Tuning model
Chose the best validation model accoring the log results, (eg:model_0009999.pth)

(Example) python ./tools/train_net.py --config-file /..../OV-DINO-main/ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep.py --eval-only --resume train.init_checkpoint=/..../OV-DINO-main/wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep/model_0009999.pth train.output_dir=/..../OV-DINO-main/wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep dataloader.evaluator.output_dir="/..../OV-DINO-main/wkdrs/ovdino_swin_tiny224_bert_base_ft_milestone_big_24ep/eval_milestone_big_2025mmdd-HHMMSS" 


