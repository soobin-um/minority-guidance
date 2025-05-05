# Don't Play Favorites: Minority Guidance for Diffusion Models (ICLR 2024)

[Soobin Um](https://soobin-um.github.io/), [Suhyeon Lee](https://scholar.google.com/citations?user=V9rMrFQAAAAJ&hl=en), and [Jong Chul Ye](https://bispl.weebly.com/professor.html)

This repository contains the official code for the paper "Don't Play Favorites: Minority Guidance for Diffusion Models" (ICLR 2024).

## 1. Environment setup
This instruction explains how to use our code base specifically focusing on LSUN-Bedrooms. Application on other datasets based on this example would be straightforward. Our implementation is heavily based on the [codebase](https://github.com/openai/guided-diffusion) for ["Diffusion Models Beat GANS on Image Synthesis"](https://arxiv.org/abs/2105.05233).

### 1) Clone the repository
```
git clone https://github.com/soobin-um/minority-guidance
cd minority-guidance
```

### 2) Install dependencies
Here's a summary of the dependencies you'll need to install:
- Python 3.11
- PyTorch 2.0.1
- CUDA 11.7

To this end, we recommend using conda to install all of the dependencies.
```
conda env create -f environment.yaml
```
If you have troubles with the above command, you can manually install the required packages by running the following commands:
```
conda create -n mg python=3.11.4
conda activate mg
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
conda install -c conda-forge mpi4py mpich
pip install lpips
pip install blobfile
pip install scikit-learn
```


## 2. Download a dataset
LSUN-Bedrooms is a huge dataset, so downloading all of it could be painful. Instead, you may want to download a smaller version [here](https://www.kaggle.com/datasets/jhoward/lsun_bedroom) where only a piece of the dataset is provided.

Put it into the place wherever you want. We will refer below the name of the folder as ```[your_data_dir]```.


## 3. Download pre-trained checkpoints
There are two pre-trained checkpoints used in our LSUN-Bedrooms experiments:
- A backbone diffusion model (i.e., ADM LSUN-Bedroom)
- A feature extractor (i.e., ADM ImageNet classifier)

You can download the diffusion model checkpoint in this [link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt). Place the model in the folder you want. This checkpoint will be refered as ```[your_model_path]```.

For the feature extractor, refer to this [link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt). The folder of this checkpoint will be refered as ```[your_fe_path]```.


## 4. Construct a labeled dataset with minority score
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
MS_FLAGS="--batch_size 50 --n_iter 5 --perturb_t 60 --num_m_classes 100 --val_ratio 0.05"
python3 construct_ms_dataset.py $MODEL_FLAGS $MS_FLAGS --model_path [your_model_path] --data_dir [your_data_dir] --output_dir [your_output_dir]
```
where:
- ```n_iter``` is the number of iterations per sample;
- ```perturb_t``` corresponds to the perturbation timestep (i.e., $t$ in Eq. 7);
- ```num_m_classes``` denotes the number of minority classes;
-  ```val_ratio``` is the validation set ratio;
- ```[your_output_dir]``` indicates a directory where the computed minority scores and the resulting paired dataset are stored.

This process would end up with the training set included in ```[your_output_dir]/train``` and the validation set contained in ```[your_output_dir]/val```. The samples will be formed as: ```[class_id]_[sample_id].jpg```.



## 5. Train a minority classifier with the labeled dataset
```
TRAIN_FLAGS="--iterations 60000 --anneal_lr True --batch_size 256 --lr 3e-4 --log_interval 100 --save_interval 1000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--latent_size 8 --in_channels 512 --out_channels 100 --classifier_attention_resolutions 8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown False --classifier_use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
python classifier_train.py $TRAIN_FLAGS $CLASSIFIER_FLAGS $DIFFUSION_FLAGS --data_dir [your_output_dir]/train --val_data_dir [your_output_dir]/val --f_extractor_path [your_fe_path]
```
where:
- ```latent_size``` and ```in_channels``` represent the input dimension of the minority classifier, i.e., 8 and 512, respectively;
- ```out_channels``` is equal to the number of minority classes (i.e., ```num_m_classes``` in the above).

We attach the pre-trained checkpoint of the minority classifier used in our experiments. Please refer to ```./ckpt/mc_lsun.pt```.


## 6. Sampling with minority guidance
```
SAMPLE_FLAGS="--batch_size 50 --num_samples 50 --timestep_respacing 250 --classifier_scale 3.5 --use_manual_class True --manual_class_id 99"
python classifier_sample.py $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS --model_path [your_model_path] --classifier_path [your_clf_path] --f_extractor_path [your_fe_path]
```
where:
- ```classifier_scale``` determines the strength of minority guidance;
- ```manual_class_id``` is the target minority class;

Note that ```[your_clf_path]``` should be the checkpoint name of the minority classifier that you want to use for minority guidance.

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{um2024dont,
  title={Don't Play Favorites: Minority Guidance for Diffusion Models},
  author={Soobin Um and Suhyeon Lee and Jong Chul Ye},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=3NmO9lY4Jn}
}
```
