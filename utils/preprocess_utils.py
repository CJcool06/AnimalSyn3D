import numpy as np


def abs_norm(points, height, width):
    norm_points = points.copy()
    norm_points[..., 0] = norm_points[..., 0] / width * 2 - 1
    norm_points[..., 1] = norm_points[..., 1] / height * 2 - 1
    # if norm_points.shape[-1] == 3:
    #     norm_points[..., 2] = norm_points[..., 2] / width
    return norm_points


def abs_norm_3d(points, height, width):
    norm_points = points.copy()
    norm_points[..., 0] = norm_points[..., 0] / width
    norm_points[..., 1] = norm_points[..., 1] / height
    if norm_points.shape[-1] == 3:
        norm_points[..., 2] = norm_points[..., 2] / width - 1
        # norm_points[..., 2] = norm_points[..., 2] / (width * 2)
    return norm_points


def relative_norm(self, points, scaling=False):
    mean = points.mean(axis=1, keepdims=True)
    translated_data = points - mean

    if scaling:
        # Compute the scaling factor
        max_abs_values = np.max(
            np.abs(translated_data), axis=-2, keepdims=True
        )  # Shape: [Batch_size x 1 x D]
        scaling_factor = np.max(
            max_abs_values, axis=-1, keepdims=True
        )  # Shape: [Batch_size x 1 x 1]

        # if any value in scaling_factor is 0,
        # add a small constant to avoid division by zero
        scaling_factor = np.where(
            scaling_factor == 0,
            1e-5,
            scaling_factor,
        )

        # Normalize the translated data
        normalized_data = translated_data / (scaling_factor)

        return normalized_data, (mean, scaling_factor)
    else:
        return translated_data, (mean, np.ones((points.shape[0], 1, 1)))