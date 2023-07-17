import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import RegressionModel, SphericalDistanceLoss
from repogen.view_regressor.data_processing import (
    load_data_from_coco_file,
    process_keypoints,
    c2s,
    s2c,
)
from visualizations import plot_training_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument(
        "--views-filename",
        type=str,
        default="views.json",
        help="Filename of the views file",
    )
    parser.add_argument(
        "--coco-filename",
        type=str,
        default="person_keypoints_val2017.json",
        help="Filename of the coco annotations file",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--spherical-input",
        action="store_true",
        default=False,
        help="If True, will train the regressor on spherical coordinates ignoring the radius",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        default=False,
        help="If True, will load the model from the checkpoint file",
    )

    return parser.parse_args()


def main(args):
    views_filepath = os.path.join(args.folder, args.views_filename)
    coco_filepath = os.path.join(args.folder, args.coco_filename)

    # Load the data
    keypoints, bboxes_xywh, image_ids, positions = load_data_from_coco_file(
        coco_filepath, views_filepath
    )
    keypoints = process_keypoints(keypoints, bboxes_xywh)

    if args.spherical_input:
        positions = c2s(positions)
        # positions = positions[:, 1:]  # Ignore the radius

    # If CUDA available, use it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Split into train and test
    train_idx = np.random.choice(
        len(keypoints), int(0.8 * len(keypoints)), replace=False
    )
    test_idx = np.setdiff1d(np.arange(len(keypoints)), train_idx)
    train_keypoints = torch.from_numpy(keypoints[train_idx, :]).type(torch.float32)
    train_positions = torch.from_numpy(positions[train_idx, :]).type(torch.float32)
    train_images = image_ids[train_idx]
    test_keypoints = torch.from_numpy(keypoints[test_idx, :]).type(torch.float32)
    test_positions = torch.from_numpy(positions[test_idx, :]).type(torch.float32)
    test_images = image_ids[test_idx]

    # Define the model, loss function, and optimizer
    input_size = train_keypoints.shape[1]
    model = RegressionModel(
        input_size=input_size,
        output_size=3 if args.spherical_input else 3,
    )

    if args.spherical_input:
        # criterion = SphericalDistanceLoss()
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    print("Number of parameters: {}".format(model.count_parameters()))
    print("Number of training samples: {}".format(len(train_keypoints)))
    print(
        "Ratio pf training samples to parameters: {:.2f}".format(
            len(train_keypoints) / model.count_parameters()
        )
    )
    print("Number of test samples: {}".format(len(test_keypoints)))

    num_epochs = args.epochs
    train_loss_log = []
    test_loss_log = []

    # Move the model and the data to the device
    model = model.to(device)
    train_keypoints = train_keypoints.to(device)
    train_positions = train_positions.to(device)
    test_keypoints = test_keypoints.to(device)
    test_positions = test_positions.to(device)
    # criterion = criterion.to(device)

    if args.load:
        model.load_state_dict(torch.load("regression_model.pt"))
    else:
        # Train the model
        training_start_time = time.time()
        for epoch in range(num_epochs):
            # Forward pass
            y_pred = model(train_keypoints)
            loss = criterion(y_pred, train_positions)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_log.append(loss.item())

            # Print progress
            if (epoch + 1) % int(num_epochs / 10) == 0 or epoch == 0:
                elapsed_time = time.time() - training_start_time
                time_per_epoch = elapsed_time / (epoch + 1)
                remaining_time = time_per_epoch * (num_epochs - epoch - 1)
                print("+---------------------------+")
                print("Epoch [{}/{}]".format(epoch + 1, num_epochs))
                print(
                    "Elapsed time: {:.2f} s ({:.2f} s per epoch)".format(
                        elapsed_time, time_per_epoch
                    )
                )
                print("Remaining time: {:.2f} s".format(remaining_time))
                print("Loss: {:.4f}".format(loss.item()))

                y_test_pred = model(test_keypoints)
                test_loss = criterion(y_test_pred, test_positions)
                print("---")
                print("Test loss: {:.4f}".format(test_loss.item()))
                test_loss_log.append(test_loss.item())

    # Test the model on new data
    print("=================================")
    y_test_pred = model(test_keypoints)
    test_loss = (
        y_test_pred.cpu().detach().numpy() - test_positions.cpu().detach().numpy()
    )
    test_dist = np.linalg.norm(test_loss, axis=1)
    print("Test dist:")
    print("min: {:.4f}".format(np.min(test_dist)))
    print("max: {:.4f}".format(np.max(test_dist)))
    print("mean: {:.4f}".format(np.mean(test_dist)))
    if args.spherical_input:
        angle_dist = np.linalg.norm(test_loss[:, 1:], axis=1)
        print("---\nTest dist (last two coordinates):")
        print("min: {:.4f}".format(np.min(angle_dist)))
        print("max: {:.4f}".format(np.max(angle_dist)))
        print("mean: {:.4f}".format(np.mean(angle_dist)))

    sort_idx = np.argsort(test_dist)
    sorted_test_dist = test_dist[sort_idx]
    sorted_test_images = test_images[sort_idx]

    # print("---\nBest images:")
    # for i in range(10):
    #     print("Image ID: {:d}, dist: {:.4f}".format(sorted_test_images[i], sorted_test_dist[i]))

    # print("---\nWorst images:")
    # for i in range(1, 11):
    #     print("Image ID: {:d}, dist: {:.4f}".format(sorted_test_images[-i], sorted_test_dist[-i]))

    plot_training_data(
        args.epochs,
        args.lr,
        train_loss_log,
        test_loss_log,
        test_positions.cpu().detach().numpy(),
        y_test_pred.cpu().detach().numpy(),
        args.spherical_input,
    )

    # Save the model
    model_filename = "regression_model.pt"
    torch.save(model.cpu().state_dict(), model_filename)


if __name__ == "__main__":
    args = parse_args()
    main(args)
