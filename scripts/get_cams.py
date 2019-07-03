import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import os

from imageio import imsave
import pandas as pd
import cv2
import torch
import numpy as np

from dataset import TASK_SEQUENCES
from cams import GradCAM, EnsembleCAM
from cams import GuidedBackPropagation
from saver import ModelSaver
import util
from util import image_util
from args import TestArgParser
from dataset import get_loader, get_eval_loaders

from models import EnsembleClassifier
from dataset.constants import IMAGENET_MEAN, IMAGENET_STD, CXR_PULM_TB_MEAN, CXR_PULM_TB_STD


def save_grad_cams(args, loader, model, output_dir, only_competition=False, only_top_task=False, probabilities_csv=None):
    """Save grad cams for all examples in a loader."""

    # 'study_level' determined if the loader is returning
    # studies or individual images
    study_level = loader.dataset.study_level

    if hasattr(model, "task2model_dicts"):
        grad_cam = EnsembleCAM(model, args.device)
        task_sequence = model.module.task_sequence
    elif isinstance(model, EnsembleClassifier):
        grad_cam = EnsembleCAM(model, args.device)
        task_sequence = model.models[0].module.task_sequence
    else:
        task_sequence = model.module.task_sequence
        grad_cam = GradCAM(model, args.device)

    # By keeping track of the example id
    # we can name each folder using the example_id.
    counter = 0
    original_dir = os.path.join(output_dir, 'original')
    cam_dir = os.path.join(output_dir, 'cam')
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    if study_level:
        # for inputs_batch, labels_batch, masks_batch in loader:
        for inputs_batch, labels_batch, info_batch, masks_batch in loader:
            key = list(info_batch.keys())[0]
            for i, (input_study, label_study, mask_study) in enumerate(zip(inputs_batch, labels_batch, masks_batch)):

                directory = f'{output_dir}/{counter}'
                # Loop over the views in a studyo
                view_id = 0
                for input_, mask_val in zip(input_study, mask_study):
                    # Skip this image if it is just a 'padded' image
                    if mask_val == 0:
                        continue

                    write_grad_cams(input_, label_study, grad_cam, directory,
                                    task_sequence,
                                    args.transform_args.normalization,
                                    info_batch[key][i],
                                    original_dir,
                                    cam_dir,
                                    only_competition=only_competition,
                                    view_id=view_id,
                                    probabilities_csv=probabilities_csv)
                    view_id = view_id + 1

                # Write label to txt and save to same folder
                # to make inspecting the cams easier
                label = np.reshape(label_study.numpy(), (1, -1))
                label_df = pd.DataFrame(label, columns=list(task_sequence))
                label_df["Path"] = info_batch['paths'][i]
                label_df["Counter"] = counter
                label_df.to_csv(f'{directory}/groundtruth.txt', index=False)

                counter = counter + 1

    else:
        for data in loader:
            if len(data) == 4:
                inputs, labels, ids, covars = data
            else:
                inputs, labels, ids = data
            key = list(ids.keys())[0]
            for i in range(len(inputs)):
                print(ids[key][i])
                directory = output_dir
                if len(data) == 4 and not isinstance(covars[i],str):
                    write_grad_cams(inputs[i], labels[i], grad_cam, directory,
                        task_sequence, args.transform_args.normalization, ids[key][i],
                        original_dir, cam_dir, covars=covars[i].unsqueeze(0),
                        probabilities_csv=probabilities_csv)
                elif len(data) == 4:
                    write_grad_cams(inputs[i], labels[i], grad_cam, directory,
                        task_sequence, args.transform_args.normalization, ids[key][i], 
                        original_dir, cam_dir, covars=[covars[i]],
                        probabilities_csv=probabilities_csv)
                else:
                    write_grad_cams(inputs[i], labels[i], grad_cam, directory,
                        task_sequence, args.transform_args.normalization, ids[key][i],
                        original_dir, cam_dir, probabilities_csv=probabilities_csv)
       

def write_grad_cams(input_, label, grad_cam, directory, task_sequence,
    normalization, subj_id, original_dir, cam_dir, only_competition=False, 
    only_top_task=False, view_id=None, covars=None, report_prob=False,
    probabilities_csv=None):

    """Creates a CAM for each image.

        Args:
            input: Image tensor with shape (3 x h x h)
            grad_cam: EnsembleCam Object wrapped around GradCam objects, which are wrapped around models.
            directory: the output folder for these set of cams
            task_sequence:
    """

    if normalization == 'cxr_pulm_tb_norm':
         normalization_mean, normalization_std = CXR_PULM_TB_MEAN, CXR_PULM_TB_STD
    else :
         normalization_mean, normalization_std = IMAGENET_MEAN, IMAGENET_STD

    if only_competition:
        COMPETITION_TASKS = TASK_SEQUENCES['competition']

    # Get the original image by
    # unnormalizing (img pixels will be between 0 and 1)
    # img shape: c, h, w
    img = image_util.un_normalize(input_, normalization_mean, normalization_std)

    # move rgb chanel to last
    img = np.moveaxis(img, 0, 2)

    # Add the batch dimension
    # as the model requires it.
    input_ = input_.unsqueeze(0)
    _, channels, height, width = input_.shape
    num_tasks = len(task_sequence)

    # Create the directory for cams for this specific example
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # If CAMs should be weighted by probabilities from a file.
    if probabilities_csv:
        print(f'Weighting CAMs by probabilities from {probabilities_csv}')
        probs_df = pd.read_csv(probabilities_csv)

    with torch.set_grad_enabled(True):

        for task_id in range(num_tasks):
            task_name = list(task_sequence)[task_id]
            if only_competition:
                if task_name not in COMPETITION_TASKS:
                    continue

            task = task_name.lower()
            task = task.replace(' ', '_')
            task_label = int(label[task_id].item())

            probs, idx, cam = grad_cam.get_cam(input_, task_id, task_name, covars)

            # using task, prob and groundtruth in file name
            prob = probs[idx==task_id].item()

            if probabilities_csv:
                subj_id_col = probs_df['subjectID'].unique()
                if str(subj_id) not in subj_id_col:
                    print(f'Failed to find {subj_id} in {probabilities_csv}')
                else:
                    prob_from_file = probs_df.loc[probs_df['subjectID'] == subj_id].iloc[0][task_name + '_prob']
                    cam = prob_from_file * cam

            # Resize cam and overlay on image
            resized_cam = cv2.resize(cam, (height, width))
            # We don't normalize since the grad cam class has already taken care of that
            img_with_cam = util.add_heat_map(img, resized_cam, normalize=False)

            # Save a cam for this task and image
            if report_prob:
                filename = f'{subj_id}_{task}_p={prob:.3f}.jpg'
            else:
                filename = f'{subj_id}_{task}.jpg'

            output_path = os.path.join(cam_dir, filename)
            imsave(output_path, img_with_cam)


    # Save the original image in the same folder
    output_path = os.path.join(original_dir, f'{subj_id}.jpg')
    img = np.uint8(img * 255)
    imsave(output_path, img)

