import argparse
import os
import sys

import numpy as np
import torch as th

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from guided_diffusion.image_datasets import _list_image_files_recursively, ImageDataset
import blobfile as bf
from torch.utils.data import DataLoader
import lpips
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

def main():
    args = create_argparser().parse_args()
    # settle the total number of step as 100
    args.timestep_respacing = "100"

    dist_util.setup_dist()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    print("creating data loader...")

    all_files = _list_image_files_recursively(args.data_dir)
    classes = None
    if args.class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        
    dataset = ImageDataset(
        args.image_size,
        all_files,
        classes=classes,
        random_crop=False,
        random_flip=False,
    )
    loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
            )
    
    # Compute minority score
    print("computing minority score of given data...")
    loss_fn = lpips.LPIPS(net='vgg').to(dist_util.dev())
    ms_list = []
    
    perturb_t = args.perturb_t

    with th.no_grad():
        for batch in tqdm(loader):
            batch = batch[0].to(dist_util.dev())
            batch_losses = th.zeros(len(batch), args.n_iter)
            t = (perturb_t * th.ones((len(batch), ))).long().to(dist_util.dev())
            for i in range(args.n_iter):
                batch_per = diffusion.q_sample(batch, t)
                batch_recon = diffusion.p_mean_variance(model, batch_per.to(dist_util.dev()), t.to(dist_util.dev()))['pred_xstart'].to(dist_util.dev())
                LPIPS_loss = loss_fn(batch.to(dist_util.dev()), batch_recon.to(dist_util.dev()))
                batch_losses[:, i] = LPIPS_loss.view(-1)
            ms_list.append(batch_losses.mean(axis=1))

    ms = th.cat(ms_list).cpu()
    os.makedirs(args.output_dir, exist_ok=True)
    th.save(ms, os.path.join(args.output_dir, 'ms_values.pt'))
    
    if args.ms_compute_only:
        print("minority score computation done")
        sys.exit()
    
    print("constructing dataset labeled with minority scores...")
    
    q_th = th.zeros(args.num_m_classes)
    
    # construct quantile-based thresholds
    for i in range(len(q_th)):
        q_th[i] = th.quantile(ms, 1 / args.num_m_classes * (i+1))
        
    ms_labels = th.zeros_like(ms).long()
    
    # labeling
    for i in range(len(ms)):
        current = ms[i]
        for j in range(len(q_th)):
            if j == 0:
                if current <= q_th[j]:
                    ms_labels[i] = j
            else:
                if current > q_th[j-1] and current <= q_th[j]:
                    ms_labels[i] = j

    # saving data
    data_indices = th.arange(len(ms_labels))
    train_indicies, val_indices, y_train, y_val = train_test_split(data_indices, ms_labels, test_size=args.val_ratio, stratify=ms_labels)
    train_indicies.shape, val_indices.shape, y_train.shape, y_val.shape
    
    output_path_train = os.path.join(args.output_dir, "train")
    output_path_val = os.path.join(args.output_dir, "val")
    os.makedirs(output_path_train, exist_ok=True)
    os.makedirs(output_path_val, exist_ok=True)
    
    save_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )

    for index, batch in enumerate(tqdm(save_loader)):
        data = batch[0]
        file_name = all_files[index].split("/")[-1]
        
        
        if index in train_indicies:
            label_index = np.where(train_indicies==index)[0].item()
            label = y_train[label_index]
            data = (data * 0.5 + 0.5).clamp_(0.0, 1.0)
            # attach minority-score-labels in front of filenames
            save_image(data, os.path.join(output_path_train, f'{label:04d}_' + file_name))
        else:
            label_index = np.where(val_indices==index)[0].item()
            label = y_val[label_index]
            data = (data * 0.5 + 0.5).clamp_(0.0, 1.0)
            # attach minority-score-labels in front of filenames
            save_image(data, os.path.join(output_path_val, f'{label:04d}_' + file_name))
            
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=10,
        num_workers=4,
        use_ddim=False,
        model_path="",
        data_dir="",
        output_dir="",
        ms_compute_only=False,
        val_ratio=0.05,
        n_iter=5,
        num_m_classes=100,
        perturb_t=60
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
