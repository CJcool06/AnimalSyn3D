import pickle
import torch
import os
import roma
import argparse
import numpy as np
import torch.nn as nn
from glob import glob
from tqdm import trange

from utils.dataset_utils import (
    load_data,
    load_deformingthings4d_data,
    save_data,
    compute_joints,
)
from utils.dataset_constants import ROOT_JOINT_LOCATIONS, IGNORE_JOINTS_FOR_OPTIM
from utils.optim_utils import (
    compute_mssd,
    mpjpe_func,
    recenter_root,
    calculate_all_bone_lengths,
    find_parents,
    with_zeros_torch,
    pack_torch,
)


# smoothness_weight = 0.05
# acceleration_weight = 0.07
# projection_weight = 0.0002


# Setup argument parser
parser = argparse.ArgumentParser(description="Process 3DLFM data.")

# Use like: python ik_optimiser.py -m foxZED elkML tigerD8H
parser.add_argument(
    "-m", "--model", nargs="+", help="The models to optimise.", required=True
)

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument(
    "-d", "--device", default=default_device, help="The hardware device.", required=True
)

parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of optimisation iterations."
)
parser.add_argument(
    "--smoothness_weight", type=float, default=0.02, help="Smoothness weight"
)
parser.add_argument(
    "--acceleration_weight", type=float, default=0.01, help="Acceleration weight"
)
parser.add_argument(
    "--resume", action="store_true", help="Show the skeleton connections."
)
args = parser.parse_args()

## Variables
models = args.model
device = args.device
epochs = args.epochs
smoothness_weight = args.smoothness_weight
acceleration_weight = args.acceleration_weight
resume = args.resume

print("Device:", device)
if resume:
    print("Previous files will not be overriden!")


### Model

class FKModelWithProjection(nn.Module):
    def __init__(
        self,
        parents,
        canon_kpts,
        num_frames,
    ):
        super(FKModelWithProjection, self).__init__()
        self.parents = parents
        self.canon_kpts = canon_kpts
        self.n_joints = len(parents)
        self.num_frames = num_frames
        self.poses = nn.Parameter(
            torch.zeros(num_frames, self.n_joints, 3, dtype=torch.float32)
        )

    def forward(self, batch_idx=None):
        poses = self.poses
        R = roma.rotvec_to_rotmat(poses)

        # Initialize the list of transformation matrices
        G_list = []

        # Initialize transformation matrix for the root joint
        G_root = torch.zeros(
            self.num_frames, 4, 4, dtype=torch.float32, device=poses.device
        )
        G_root[:, :3, :3] = R[:, 0]
        G_root[:, :3, 3] = self.canon_kpts[0]
        G_root[:, 3, 3] = 1.0
        G_list.append(G_root)

        t = (
            self.canon_kpts.unsqueeze(0) - self.canon_kpts[self.parents].unsqueeze(0)
        ).repeat(self.num_frames, 1, 1)

        for i in range(1, self.n_joints):
            parent = self.parents[i]
            Ri_ti = torch.cat((R[:, i], t[:, i].unsqueeze(-1)), dim=-1)
            Gi = with_zeros_torch(Ri_ti, self.num_frames)

            # Compute the new transformation matrix
            G_new = torch.matmul(G_list[parent], Gi)

            # Append the new transformation matrix to the list
            G_list.append(G_new)

        G = torch.stack(G_list, dim=1)

        # Concatenate canonical keypoints with zeros
        C_zeros = torch.cat(
            [
                self.canon_kpts,
                torch.zeros(self.n_joints, 1, device=self.canon_kpts.device),
            ],
            dim=1,
        )
        C_zeros = (
            C_zeros.unsqueeze(0).unsqueeze(-1).expand(self.num_frames, -1, -1, -1)
        )  # Shape: (num_frames, n_joints, 4, 1)

        # Perform matrix multiplication
        G_mult = torch.matmul(G, C_zeros)

        G_packed = pack_torch(G_mult)

        # Subtract the packed result from the original G
        G_result = G - G_packed

        # Final step: keypoints calculation
        C_ones = torch.cat(
            [
                self.canon_kpts,
                torch.ones(self.n_joints, 1, device=self.canon_kpts.device),
            ],
            dim=1,
        )
        C_ones = (
            C_ones.unsqueeze(0).unsqueeze(-1).expand(self.num_frames, -1, -1, -1)
        )  # Shape: (num_frames, n_joints, 4, 1)

        # Perform matrix multiplication to get keypoints
        keypoints_3d = torch.matmul(G_result, C_ones)[
            :, :, :3, 0
        ]  # Extract first three components

        return keypoints_3d


###############################################


def train(model, optimizer, scheduler, target, num_epochs):
    ## Train the model
    for epoch in trange(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        keypoints_3d = model(epoch)

        # Compute loss
        mpjpe_loss = mpjpe_func(keypoints_3d, target) * 1000

        # Smoothness loss (Velocity loss)
        smoothness_loss = torch.sum(
            torch.norm(model.poses[1:] - model.poses[:-1], dim=-1)
        ) + torch.sum(torch.norm(keypoints_3d[1:] - keypoints_3d[:-1], dim=-1))

        # Acceleration loss
        acceleration_loss = torch.sum(
            torch.norm(
                model.poses[2:] - 2 * model.poses[1:-1] + model.poses[:-2], dim=-1
            )
        ) + torch.sum(
            torch.norm(
                keypoints_3d[2:] - 2 * keypoints_3d[1:-1] + keypoints_3d[:-2], dim=-1
            )
        )

        loss = (
            mpjpe_loss
            + smoothness_weight * smoothness_loss
            + acceleration_weight * acceleration_loss
        )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # scheduler if reduce on plateau
        scheduler.step(loss)

    mpjpe3d = mpjpe_func(keypoints_3d, target) * 1e3  # type: ignore
    msse3d = compute_mssd(keypoints_3d.cpu().detach().numpy()) * 1e4  # type: ignore
    print("Final 3D: {} | Final MSSE: {}".format(mpjpe3d.item(), msse3d.item()))



for model in models:
    print(f"\nOptimising {model}...")
    file_paths = glob(os.path.join("data", model, "*.json"))
    for file_path in file_paths:
        if resume:
            save_path = os.path.join("optimised_data", model)
            save_path = os.path.join(save_path, file_path.split("/")[-1])
            if os.path.exists(save_path):
                continue

        print(f"File: {file_path}")
        data = load_data(file_path)
        model = data["model"]
        anim = data["animation"]
        rel_mesh_path = data["rel_mesh_path"]
        joint_group_idxs = data["joint_group_idxs"]
        joint_connections = np.array(data["connections"])

        dataset_path = "/datasets/deformingthings4d"
        anime_file = os.path.join(dataset_path, rel_mesh_path)
        def4d_data = load_deformingthings4d_data(anime_file)
        vertices = def4d_data["vertices"]
        offsets = def4d_data["offsets"]

        joints = compute_joints(vertices, offsets, joint_group_idxs)

        root_joint = ROOT_JOINT_LOCATIONS[model]
        parents_dict = find_parents(
            root=root_joint,
            points=[i for i in range(joints.shape[1])],
            connections=joint_connections,
        )

        ignored_joints = IGNORE_JOINTS_FOR_OPTIM[model]
        for joint in ignored_joints:
            del parents_dict[joint]

        # Permute the joints such that the parents list is legal.
        keys = np.array(list(parents_dict.keys()))
        vals = np.array(list(parents_dict.values()))
        sorted_parents = np.zeros_like(vals) - 1
        for i, k in enumerate(keys):
            sorted_parents[vals == k] = i
        sorted_parents = torch.tensor(sorted_parents)

        permuted_joints = joints[:, keys, :].copy()

        # Recenter the root joint
        final_3d_centered, root_joint_locations = recenter_root(permuted_joints)
        final_3d_centered = torch.tensor(
            final_3d_centered, dtype=torch.float32, device=device
        )
        X_target = final_3d_centered.clone()
        num_frames = X_target.shape[0]

        ## Initialize model, optimizer and scheduler
        fk_model = FKModelWithProjection(
            parents=sorted_parents,
            canon_kpts=final_3d_centered[0],
            num_frames=num_frames,
        ).to(device)

        optimizer = torch.optim.Adam(fk_model.parameters(), lr=1e-1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=100,
        )

        ## Calculate total params to be optimized
        total_params = sum(p.numel() for p in fk_model.parameters() if p.requires_grad)
        print("Total params: ", total_params)

        print(
            "MSSD before: {:.2f} mm".format(compute_mssd(X_target.cpu().numpy()) * 1e4)
        )

        train(
            model=fk_model,
            optimizer=optimizer,
            scheduler=scheduler,
            target=X_target,
            num_epochs=epochs,
        )

        output_3d = fk_model(0)

        bone_lengths = calculate_all_bone_lengths(
            X_target.cpu().detach().numpy(), sorted_parents.cpu().detach().numpy()
        )
        bone_diffs = (np.diff(bone_lengths, axis=0) ** 2) ** 0.5
        print("Mean bone diffs before:", bone_diffs.mean())

        bone_lengths = calculate_all_bone_lengths(
            output_3d.cpu().detach().numpy(), sorted_parents.cpu().detach().numpy()
        )
        bone_diffs = (np.diff(bone_lengths, axis=0) ** 2) ** 0.5
        print("Mean bone diffs after:", bone_diffs.mean())

        output_3d = output_3d.cpu().detach().numpy() + root_joint_locations
        X_target = X_target.cpu().detach().numpy() + root_joint_locations

        reverse_idx = np.argsort(keys)
        output_3d = output_3d[:, reverse_idx]
        X_target = X_target[:, reverse_idx]

        mask = np.ones(joints.shape[1], dtype=bool)
        mask[ignored_joints] = False
        all_output = np.empty_like(joints)
        all_output[:, mask] = output_3d
        all_output[:, ~mask] = joints[:, ~mask]

        data["optimised_joints"] = all_output.tolist()

        save_path = os.path.join("optimised_data", model)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, file_path.split("/")[-1])
        save_data(data, save_path)
        print(f"Saved data to {save_path}.")

        with open(
            f"optim_joints_{model}_{file_path.split('/')[-1].split('.')[0]}.pkl", "wb"
        ) as f:
            pickle.dump(all_output, f)
