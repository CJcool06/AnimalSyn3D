import torch
import numpy as np
from collections import deque


def compute_mssd(data):
    diff = np.diff(data, axis=0)
    squared_diff = np.square(diff)
    mssd = np.mean(squared_diff)
    return mssd


def mpjpe_func(pred, gt, masks=None):
    # If masks are provided, apply them to pred and gt
    if masks is not None:
        # Expand masks along the last dimension to match the shape of pred and gt
        masks_expanded = masks.unsqueeze(-1)
        pred = pred * masks_expanded
        gt = gt * masks_expanded
        error = ((pred - gt).norm(dim=2) * masks).sum(-1) / masks.sum(-1)
    else:
        # Compute the error
        error = (pred - gt).norm(dim=2).mean(-1)

    return error.mean()


def recenter_root(joint_locations):
    # Subtract the root joint coordinates from all joints in all frames
    root_joint_locations = joint_locations[:, 0, :][:, np.newaxis, :]
    centered_joint_locations = joint_locations - root_joint_locations
    return centered_joint_locations, root_joint_locations


def calculate_all_bone_lengths(all_frames, parents):
    return np.array([get_bone_lengths(frame, parents) for frame in all_frames])


def get_bone_lengths(joint_coords, parents):
    return np.array(
        [
            np.linalg.norm(joint_coords[i] - joint_coords[p]) if p != -1 else 0
            for i, p in enumerate(parents)
        ]
    )


def find_parents(root, points, connections):
    # Initialize the parent dictionary and the BFS queue
    parent = {root: -1}
    queue = deque([root])

    # Create an adjacency list from the connections
    adj_list = {point: [] for point in points}
    for u, v in connections:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Perform BFS traversal
    while queue:
        current = queue.popleft()
        for neighbor in adj_list[current]:
            if neighbor not in parent:  # If the neighbor hasn't been visited yet
                parent[neighbor] = current  # Set the parent
                queue.append(neighbor)

    return parent


def with_zeros_torch(x, num_frames):
    ones = (
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=x.device)
        .view(1, 1, 4)
        .repeat(num_frames, 1, 1)
    )
    return torch.cat((x, ones), dim=1)


# Pack the result
def pack_torch(x):
    zeros = torch.zeros(x.shape[0], x.shape[1], 4, 3, device=x.device)
    return torch.cat((zeros, x), dim=3)