import os
import argparse
import time
import json
import numpy as np
import warnings
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from model import RegressionModel, SphericalDistanceLoss, MSELossWithRegularization, SphericalDotProductLoss
from repogen.view_regressor.data_processing import load_data_from_coco_file, process_keypoints, c2s, s2c, angular_distance, randomly_occlude_keypoints
from visualizations import plot_heatmap, visualize_pose

torch.autograd.set_detect_anomaly(True)

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument('--views-filename', type=str, default="views.json",
                        help='Filename of the views file')
    parser.add_argument('--coco-filename', type=str, default="person_keypoints_val2017.json",
                        help='Filename of the coco annotations file')
    parser.add_argument('--workdir', type=str, default="logs",
                        help='Workdir where to save the model and the logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name of the subfolder where to save the model and the logs')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay for the optimizer')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--net-depth', type=int, default=3)
    parser.add_argument('--net-width', type=int, default=128)
    parser.add_argument('--test-interval', type=int, default=10)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--spherical-output', action="store_true", default=False,
                        help='If True, will train the regressor on spherical coordinates with the radius')
    parser.add_argument('--flat-output', action=argparse.BooleanOptionalAction, default=True,
                        help='If True, will train the regressor on spherical coordinates ignoring the radius')
    parser.add_argument('--loss', type=str, default="MSE+REG",
                        help='Loss function. Known values: MSE, L1, Spherical, MSE+Reg')
    parser.add_argument('--distance', type=str, default="Euclidean",
                        help='Distance function. Known values: Euclidean, Spherical. If Spherical and 3d output, will ignore the radius.')
    parser.add_argument('--load', action="store_true", default=False,
                        help='If True, will load the model from the checkpoint file')
    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Will force CPU computation')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Will print loss to the console')
    parser.add_argument('--normalize-input', action=argparse.BooleanOptionalAction, default=True,
                        help='Will normalize the input keypoints by the bounding box size')
    parser.add_argument('--visibility-in-input', action="store_true", default=False,
                        help='Will add the visibility of the keypoints to the input')
    parser.add_argument('--bbox-in-input', action=argparse.BooleanOptionalAction, default=True,
                        help='Will add the bounding box size to the input')
    parser.add_argument('--test-on-COCO', action=argparse.BooleanOptionalAction, default=True,
                        help='Will test the model on the COCO dataset')
    parser.add_argument('--num-visible-keypoints', type=int, default=6,
                        help='Number of visible keypoints to use in the input')
    parser.add_argument('--remove-limbs', action="store_true", default=False,
                        help='Will remove the limbs from the input')
    parser.add_argument('--occlude-data', action=argparse.BooleanOptionalAction, default=True,
                        help='Will randomly occlude data as augmentation')

    
    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = time.strftime("%Y%m%d_%H%M%S")

    return args


def test_model(args, model, dataloader, device, criterion, epoch, writer):
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            y_pred = model(batch_x.to(device))
            
            # Log the loss
            loss = criterion(y_pred, batch_y.to(device))
            writer.add_scalar('Loss/test', loss.item(), epoch)

            # Log the distance
            if args.distance.upper() == "EUCLIDEAN":
                test_distance = np.linalg.norm(y_pred.cpu().numpy() - batch_y.cpu().numpy(), axis=1)
            elif args.distance.upper() == "SPHERICAL":
                test_distance = angular_distance(y_pred.cpu().numpy(), batch_y.cpu().numpy())
            writer.add_histogram(
                '04 Test Distance/test',
                test_distance,
                global_step = epoch,
            )

            # Log the histogram of radius
            if args.spherical_output:
                test_radius = y_pred[:, 0].cpu().numpy()
            else:
                test_radius = np.linalg.norm(y_pred.cpu().numpy(), axis=1)
            writer.add_histogram(
                '01 Test Radius/test',
                test_radius,
                global_step = epoch,
            )

            # Log the PDF (probability density function)
            test_pdf, test_theta, test_phi = plot_heatmap(y_pred.cpu().numpy(), args.spherical_output, return_img=True, return_angles=True)
            test_pdf = np.array(test_pdf).astype(np.uint8).transpose(2, 0, 1)
            writer.add_image(
                "Test PDF/test",
                test_pdf,
                global_step = epoch,
            )

            # Log the histograms of angles
            writer.add_histogram(
                '02 Test Theta/test',
                test_theta,
                global_step = epoch,
            )
            writer.add_histogram(
                '03 Test Phi/test',
                test_phi,
                global_step = epoch,
            )


def infere_model_on_COCO(args, model, dataloader, device, epoch, writer, string="COCO", image_ids=None, image_folder=None):
    with torch.no_grad():
        for batch_x in dataloader:

            batch_x = batch_x[0]
            y_pred = model(batch_x.to(device))

            # Log the histogram of radius
            if args.spherical_output:
                test_radius = y_pred[:, 0].cpu().numpy()
            else:
                test_radius = np.linalg.norm(y_pred.cpu().numpy(), axis=1)
            writer.add_histogram(
                '11 {:s} Radius/test'.format(string),
                test_radius,
                global_step = epoch,
            )

            # Log the PDF (probability density function)
            test_pdf, test_theta, test_phi = plot_heatmap(y_pred.cpu().numpy(), args.spherical_output, return_img=True, return_angles=True)
            test_pdf = np.array(test_pdf).astype(np.uint8).transpose(2, 0, 1)
            writer.add_image(
                "{:s} PDF/test".format(string),
                test_pdf,
                global_step = epoch,
            )

            # Log the histograms of angles
            writer.add_histogram(
                '12 {:s} Theta/test'.format(string),
                test_theta,
                global_step = epoch,
            )
            writer.add_histogram(
                '13 {:s} Phi/test'.format(string),
                test_phi,
                global_step = epoch,
            )

            # Select N 'most top' images
            if image_ids is not None and image_folder is not None:
                if y_pred.shape[1]  == 2:
                    tops =  np.stack([
                            np.ones((len(y_pred), 1)) * np.pi / 2,
                            np.ones((len(y_pred), 1)) * np.pi / 2,
                        ], axis=1).squeeze()
                else:
                    tops = np.stack([
                            np.ones((len(y_pred), 1)),
                            np.ones((len(y_pred), 1)) * np.pi / 2,
                            np.ones((len(y_pred), 1)) * np.pi / 2,
                        ], axis=1).squeeze()
                distance_from_top = angular_distance(
                    c2s(y_pred.cpu().numpy()),
                    tops
                ).squeeze()

                top_indices = np.argsort(distance_from_top)
                for i, idx in enumerate(top_indices[:5]):
                    idx = int(idx)
                    
                    in_img = cv2.imread(os.path.join(image_folder, "..", "val2017", "{:012d}.jpg".format(int(image_ids[idx])))),
                    if isinstance(in_img, tuple):
                        in_img = in_img[0]
                    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
                    in_img = in_img.astype(np.uint8).transpose(2, 0, 1)

                    writer.add_image(
                        "{:s} Top Images/image {:02d}".format(string, i),
                        in_img,
                        global_step = epoch,
                    )

                    writer.add_image(
                        "{:s} Top Poses/pose {:02d}".format(string, i),
                        visualize_pose(batch_x[idx, :].numpy().squeeze(), has_bbox=args.bbox_in_input).astype(np.uint8).transpose(2, 0, 1),
                        global_step = epoch,
                    )


def main(args):

    # Create the workdir
    args.workdir = os.path.join(args.workdir, args.experiment_name)
    os.makedirs(args.workdir, exist_ok=True)

    views_filepath = os.path.join(args.folder, args.views_filename)
    coco_filepath = os.path.join(args.folder, args.coco_filename)
    
    # Load the data
    keypoints, bboxes_xywh, image_ids, positions = load_data_from_coco_file(
        coco_filepath,
        views_filepath,
        remove_limbs=args.remove_limbs,
        num_visible_keypoints=args.num_visible_keypoints,
    )
    keypoints = process_keypoints(
        keypoints,
        bboxes_xywh,
        normalize=args.normalize_input,
        add_visibility=args.visibility_in_input,
        add_bboxes=args.bbox_in_input,
        remove_limbs=args.remove_limbs,
    )

    if args.spherical_output:
        positions = c2s(positions)
        if args.flat_output:
            positions = positions[:, 1:]
    elif args.flat_output:
        positions = c2s(positions)
        positions = positions[:, 1:]    # Remove the radius
        positions = s2c(positions)

    # If CUDA available, use it
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    input_size = keypoints.shape[1]
    output_size = positions.shape[1]

    # Create a DataLoader
    dataset = TensorDataset(
        torch.from_numpy(keypoints).float(),
        torch.from_numpy(positions).float(),
    )

    # Split the data to the training and testing sets
    train_size = int(args.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_size if args.batch_size <= 0 else args.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    # Load the COCO dataset if needed
    if args.test_on_COCO:
        test_filepaths = {
            "COCO": "/datagrid/personal/purkrmir/data/COCO/original/annotations/person_keypoints_val2017.json",
            "FRONT": "/datagrid/personal/purkrmir/data/pose_experiments/FRONT_views_test/annotations/person_keypoints_val2017.json",
            "TOP": "/datagrid/personal/purkrmir/data/pose_experiments/TOP_views_test/annotations/person_keypoints_val2017.json",
            "PERIMETER": "/datagrid/personal/purkrmir/data/pose_experiments/PERIMETER_views_test/annotations/person_keypoints_val2017.json",
            "TRAIN": "/datagrid/personal/purkrmir/data/SyntheticPose/pose_regressor_50k/annotations/person_keypoints_val2017.json",
        }
        test_dataloaders = {}
        for key, filepath in test_filepaths.items():
            coco_keypoints, coco_bboxes_xywh, coco_image_ids = load_data_from_coco_file(
                filepath,
                remove_limbs=args.remove_limbs,
                num_visible_keypoints=args.num_visible_keypoints,
            )
            print(coco_keypoints.shape, coco_bboxes_xywh.shape, coco_image_ids.shape)
            coco_keypoints = process_keypoints(
                coco_keypoints,
                coco_bboxes_xywh,
                normalize=args.normalize_input,
                add_visibility=args.visibility_in_input,
                add_bboxes=args.bbox_in_input,
                remove_limbs=args.remove_limbs,
            )
            coco_dataloader = DataLoader(
                TensorDataset(
                    torch.from_numpy(coco_keypoints).float(),
                    # torch.from_numpy(coco_image_ids).float(),
                ),
                batch_size=coco_keypoints.shape[0],
                shuffle=False
            )
            test_dataloaders[key] = {}
            test_dataloaders[key]["dataloader"] = coco_dataloader
            test_dataloaders[key]["image_ids"] = coco_image_ids
            test_dataloaders[key]["folder"] = os.path.dirname(filepath)

    # Define the model, loss function, and optimizer
    model = RegressionModel(
        input_size=input_size,
        output_size=output_size,
        width = args.net_width,
        depth = args.net_depth,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.loss.upper() == "MSE":
        criterion = nn.MSELoss()
    elif args.loss.upper() == "L1":
        criterion = nn.L1Loss()
    elif args.loss.upper() == "MSE+REG":
        criterion = MSELossWithRegularization(
            reg_weight=1.0,
            reg_type="exact" if args.flat_output else "min",
        )
    elif args.loss.upper() == "SPHERICAL":
        # criterion = SphericalDistanceLoss(
        #     is_cartesian=not args.spherical_output,
        # )
        criterion = SphericalDotProductLoss(is_spherical=args.spherical_output)
    else:
        raise ValueError("Unknown loss function: {}".format(args.loss))
    
    if not args.distance.upper() in ["EUCLIDEAN", "SPHERICAL"]:
        raise ValueError("Unknown distance function: {}".format(args.distance))
    
    # Print the number of parameters
    print('Number of parameters: {}'.format(model.count_parameters()))

    # Load the model if needed
    if args.load:
        model.load_state_dict(torch.load(args.load))
        print("Loaded model from {}".format(args.load))

    # Training loop
    writer = SummaryWriter(
        log_dir=args.workdir,
        comment=args.experiment_name,
    )
    start_time = time.time()
    for epoch in tqdm(range(args.epochs), ascii=True):
        losses = []
        max_params = []
        for batch_x, batch_y in train_dataloader:

            if args.occlude_data:
                batch_x = randomly_occlude_keypoints(
                    batch_x,
                    has_bbox=args.bbox_in_input,
                    min_num_keypoints=6,
                    rectangle_pst=0.3,
                    occlusion_pst=0.1,
                )
            
            # Forward pass
            y_pred = model(batch_x.to(device))
            assert not torch.any(torch.isnan(y_pred))
            # print(c2s(y_pred.cpu().detach().numpy())[0, :])
            loss = criterion(y_pred, batch_y.to(device))
            # print(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            max_params.append(model.get_biggest_parameter().item())

            # Save the loss for later averaging
            losses.append(loss.item())

        # Log the loss
        writer.add_scalar('Loss/train', np.mean(losses), epoch)
        # Log the max_params
        writer.add_scalar('Params/train', np.max(max_params), epoch)
        
        # Test the model
        if epoch % args.test_interval == 0:
            test_model(
                args,
                model,
                test_dataloader,
                device,
                criterion,
                epoch,
                writer,
            )

            if args.test_on_COCO:
                for key, coco_item in test_dataloaders.items():
                    infere_model_on_COCO(
                        args,
                        model,
                        coco_item["dataloader"],
                        device,
                        epoch,
                        writer,
                        string=key,
                        image_ids=coco_item["image_ids"],
                        image_folder=coco_item["folder"],
                    )
        
        # Print progress
        if args.verbose and (epoch) % args.test_interval == 0:
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (epoch+1) * (args.epochs - epoch - 1)
            print('Epoch [{:5d}/100]\tLoss: {:7.4f}\tElapsed: {:5.2f} s\tRemaining: {:5.2f} s'.format(epoch+1, loss.item(), elapsed_time, remaining_time))


    # Test the model
    test_model(
        args,
        model,
        test_dataloader,
        device,
        criterion,
        epoch,
        writer,
    )

    for key, coco_item in test_dataloaders.items():
        infere_model_on_COCO(
            args,
            model,
            coco_item["dataloader"],
            device,
            epoch,
            writer,
            string=key,
            image_ids=coco_item["image_ids"],
            image_folder=coco_item["folder"],
        )
            

    # Save the model
    model_filename = os.path.join(args.workdir, "view_regressor.pth")
    torch.save(model.cpu().state_dict(), model_filename)
    print("Model saved to {}".format(model_filename))

    # Save the arguments
    args_filename = os.path.join(args.workdir, "args.json")
    with open(args_filename, "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)