import math
import numpy as np
from typing import Any
from scipy.interpolate import interp1d


def rotation_matrix_x(angle_deg):
    angle_rad = angle_deg * (np.pi / 180)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle_deg):
    angle_rad = angle_deg * (np.pi / 180)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle_deg):
    angle_rad = angle_deg * (np.pi / 180)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotate_camera_around_center(camera_position, center, angle):
    relative_position = camera_position - center
    R = rotation_matrix_y(angle)
    rotated_position = R @ relative_position
    new_camera_position = rotated_position + center
    return new_camera_position


def rotate_camera_around_center_xyz(camera_position, center, angle_x, angle_y, angle_z):
    relative_position = camera_position - center
    R = rotation_matrix_x(angle_x) @ (
        rotation_matrix_y(angle_y) @ rotation_matrix_z(angle_z)
    )
    rotated_position = R @ relative_position
    new_camera_position = rotated_position + center
    return new_camera_position


def look_at(camera_position, target, up_vector=np.array([0, 1, 0])):
    forward = target - camera_position
    forward /= np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    return np.vstack([right, up, forward])


def calculate_fov(bounding_box, distance_to_center, sensor_size):
    bounding_box_size = np.linalg.norm(bounding_box[1] - bounding_box[0])
    fov = 2 * math.atan(bounding_box_size / (2 * distance_to_center))
    return math.degrees(fov)


def calculate_overall_camera_parameters(
    frames,
    scalar_value=1.5,
    sensor_size=36,
    focal_length=35,
    angle=0,
):
    all_points = np.vstack(frames)

    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    bounding_box_center = (min_coords + max_coords) / 2

    distance_to_center = np.linalg.norm(max_coords - min_coords) * scalar_value
    camera_position = bounding_box_center + np.array([distance_to_center, 0, 0])
    if isinstance(angle, list):
        camera_position = rotate_camera_around_center_xyz(
            camera_position, bounding_box_center, *angle
        )
    else:
        camera_position = rotate_camera_around_center(
            camera_position, bounding_box_center, angle
        )
    camera_orientation = look_at(
        camera_position, bounding_box_center, up_vector=np.array([0, 1, 0])
    )

    fov = calculate_fov((min_coords, max_coords), distance_to_center, sensor_size)
    # focal_length = sensor_size / (2 * math.tan(math.radians(fov) / 2))

    camera_intrinsics = {
        "focal_length": focal_length,
        "sensor_size": sensor_size,
        "fov": fov,
    }

    camera_extrinsics = {"position": camera_position, "orientation": camera_orientation}

    return {"intrinsics": camera_intrinsics, "extrinsics": camera_extrinsics}


def calculate_intrinsic_matrix(focal_length, sensor_size, image_size):
    fx = focal_length * (image_size[0] / sensor_size)
    fy = focal_length * (image_size[1] / sensor_size)
    cx = image_size[0] / 2
    cy = image_size[1] / 2

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def calculate_extrinsic_matrix(position, orientation):
    R = orientation
    T = -R @ position

    extrinsic = np.hstack((R, T.reshape(-1, 1)))
    return extrinsic


def calculate_projection_matrix(K, extrinsic):
    P = K @ extrinsic
    return P


def interpolate_frames(points, target_frames):
    T, N, _ = points.shape

    # Generate some example 3D data (T frames of N points)
    original_data = points

    # Original frame indices
    original_indices = np.arange(T)

    # New frame indices
    new_indices = np.linspace(0, T - 1, target_frames)

    # Interpolated data array
    interpolated_data = np.zeros((target_frames, N, 3))

    # Perform interpolation for each point and dimension
    for point in range(N):
        for dim in range(3):
            # Create the interpolation function for the current point and dimension
            interp_func = interp1d(
                original_indices, original_data[:, point, dim], kind="cubic"
            )
            # Apply the interpolation function to the new indices
            interpolated_data[:, point, dim] = interp_func(new_indices)

    return interpolated_data


class PixelSpaceScaling:

    def forward(self, points_2d, points_cam, root_joint_idx):
        proj_2d_c = points_2d - points_2d[:, root_joint_idx : root_joint_idx + 1, :]
        cam_3d_c = points_cam - points_cam[:, root_joint_idx : root_joint_idx + 1, :]
        scaling_factor = np.linalg.norm(
            np.linalg.norm(proj_2d_c, axis=1), axis=1
        ) / np.linalg.norm(np.linalg.norm(cam_3d_c[..., :2], axis=1), axis=1)
        scaled_3d = points_cam * scaling_factor[:, None, None]
        return scaled_3d, scaling_factor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class RelativeScaling:
    """
    Scale each frame relative to the first frame.
    """

    def forward(self, points_cam):
        points_cam_centered = points_cam - points_cam.mean(axis=-2, keepdims=True)
        scaling_factor = 1 / np.abs(points_cam_centered[0]).max()
        scaling_factor = scaling_factor.repeat(points_cam.shape[0])
        assert not (
            (points_cam_centered * scaling_factor[:, None, None])[0] > 1
        ).any(), "Scaled 3D has value greater than 1 for the first frame. This should not happen."

        scaled_3d = points_cam * scaling_factor[:, None, None]
        return scaled_3d, scaling_factor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class AbsoluteScaling:
    """
    Scale each frame.
    """

    def forward(self, points_cam):
        points_cam_centered = points_cam - points_cam.mean(axis=-2, keepdims=True)
        scaling_factor = 1 / np.abs(points_cam_centered).max(axis=-1).max(axis=-1)
        assert not (
            np.abs(points_cam_centered * scaling_factor[:, None, None]) > 1
        ).any(), "Scaled 3D has value greater than 1. This should not happen."

        scaled_3d = points_cam * scaling_factor[:, None, None]
        return scaled_3d, scaling_factor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)