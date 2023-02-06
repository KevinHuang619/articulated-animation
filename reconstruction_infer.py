"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
import cv2
from pathlib import Path

def reconstruction(config, generator, region_predictor, bg_predictor, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, region_predictor=region_predictor, bg_predictor=bg_predictor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []

    generator.eval()
    region_predictor.eval()
    bg_predictor.eval()

    fvr_data_dir = Path("/home/kevinhuang/github/articulated-animation/data/cherrypicks")
    fvr_list = list(fvr_data_dir.rglob("*.png"))

    cnt = 0
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            source_region_params = region_predictor(x['video'][:, :, 0])
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]

                source = cv2.imread(fvr_list[cnt].as_posix())
                source = cv2.resize(source, (256, 256))
                source = (source / 255.).astype(np.float32)
                source = torch.from_numpy(source).permute(2, 0, 1)
                source = source.unsqueeze(0)
                source = source.cuda()

                driving = x['video'][:, :, frame_idx]
                driving_region_params = region_predictor(driving)

                bg_params = bg_predictor(source, driving)
                out = generator(source, source_region_params=source_region_params,
                                driving_region_params=driving_region_params, bg_params=bg_params)

                out['source_region_params'] = source_region_params
                out['driving_region_params'] = driving_region_params

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                visualization = visualization[:, 256*3:256*4, :]
                visualizations.append(visualization)
                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            for i, prediction in enumerate(predictions):
                imageio.imsave(os.path.join(png_dir, x['name'][0] + f"_{i}" + '.png'), (255 * prediction).astype(np.uint8))

            # predictions = np.concatenate(predictions, axis=1)
            # imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            # image_name = x['name'][0] + config['reconstruction_params']['format']
            # imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
        if cnt == 245:
            break
        else:
            cnt += 1

    print("L1 reconstruction loss: %s" % np.mean(loss_list))
