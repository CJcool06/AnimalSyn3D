import json
import numpy as np


def load_deformingthings4d_data(mesh_file_path):
    with open(mesh_file_path, "rb") as f:
        # Load info
        nf = np.fromfile(f, dtype=np.int32, count=1)[0]
        nv = np.fromfile(f, dtype=np.int32, count=1)[0]
        nt = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Load mesh
        vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
        face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
        offset_data = np.fromfile(f, dtype=np.float32, count=-1)

    # Reshape
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))

    # Pad zero offset at first frame
    offset_data = np.insert(offset_data, 0, np.zeros((nv, 3)), axis=0)

    return {
        "num_frames": nf,
        "num_vertices": nv,
        "num_faces": nt,
        "vertices": vert_data,
        "faces": face_data,
        "offsets": offset_data,
    }


def load_data(data_path):
    with open(data_path, "rb") as f:
        data = json.load(f)
    return data


def save_data(data, data_path):
    with open(
        data_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def compute_joints(
    vertices: np.ndarray,
    offsets: np.ndarray,
    group_idxs: list,
):
    num_frames = offsets.shape[0]
    num_joints = len(group_idxs)
    joint_locs = np.zeros((num_frames, num_joints, 3))
    for frame_idx, offset in enumerate(offsets):
        frame_vertices = vertices + offset
        for joint_idx, point_group_idx in enumerate(group_idxs):
            picked_points = frame_vertices[point_group_idx]
            joint_loc = picked_points.mean(axis=0)
            joint_locs[frame_idx, joint_idx] = joint_loc

    return joint_locs


def do_projection_2d(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
):
    num_frames = points.shape[0]
    num_joints = points.shape[1]
    points_ = np.concatenate(
        (points, np.ones((num_frames, num_joints, 1))),
        axis=-1,
    )  # Nx4

    proj_mat = intrinsic @ extrinsic[:3]
    proj_mat = proj_mat[None, ...].repeat(num_frames, axis=0)
    proj_points = np.einsum("ftd,fjd->fjt", proj_mat, points_)  # type: ignore
    proj_points = proj_points / proj_points[:, :, 2:]

    return proj_points[..., :2]


def do_projection_from_cam(
    points: np.ndarray,
    intrinsic: np.ndarray,
):
    points_h = intrinsic[None, ...] @ points.transpose(0, 2, 1)
    proj = points_h.transpose(0, 2, 1)

    # Homogenous to cartesian
    proj[..., :2] = proj[..., :2] / proj[..., 2:]

    return proj[..., :2]


def do_transform_to_cam(
    points: np.ndarray,
    extrinsic: np.ndarray,
):
    num_frames = points.shape[0]
    num_joints = points.shape[1]
    points_ = np.concatenate(
        (points, np.ones((num_frames, num_joints, 1))),
        axis=-1,
    )  # Nx4

    # Transform points to camera coordinates
    points_c = extrinsic[None, ...].repeat(num_frames, axis=0) @ points_.transpose(
        0, 2, 1
    )

    return points_c.transpose(0, 2, 1)
