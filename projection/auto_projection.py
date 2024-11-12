import os
import math
import numpy as np
from glob import glob
from scipy.interpolate import interp1d

from utils.projection_utils import (
    interpolate_frames,
    calculate_overall_camera_parameters,
    calculate_intrinsic_matrix,
    calculate_extrinsic_matrix,
    PixelSpaceScaling,
    RelativeScaling,
    AbsoluteScaling,
)
from utils.dataset_utils import (
    do_projection_from_cam,
    do_transform_to_cam,
    load_data,
    load_deformingthings4d_data,
    compute_joints,
    save_data,
)
from utils.dataset_constants import ROOT_JOINT_LOCATIONS


np.random.seed(999)

IMAGE_SIZE = (1080, 1080)  # (width, height)
sequence_length = 48

projection_strategy = "perspective"
# projection_strategy = "weak_perspective"
# projection_strategy = "orthographic"

# scaling_strategy = None
scaling_strategy = PixelSpaceScaling()
# scaling_strategy = RelativeScaling()
# scaling_strategy = AbsoluteScaling()

data_dir = "optimised_data"
output_dir = "optimised_data_output"
models = [
    "foxZED",
    "pumaRW",
    "deerOMG",
    "elkML",
    "bear3EP",
    "bucksYJL",
    "bunnyQ",
    "moose6OK9",
    "rabbit7L6",
    "tigerD8H",
    "raccoonVGG",
    "doggieMN5",
    "chickenDC",
    # "dragonOF2",
    # "crocodileOPW",
    # "duck",
    # "hippoDG",
]

for model in models:
    os.makedirs(os.path.join(output_dir, model), exist_ok=True)

    file_paths = glob(os.path.join(data_dir, model, "*.json"))
    for file_path in file_paths:
        print(f"File: {file_path}")

        data = load_data(file_path)
        model = data["model"]
        anim = data["animation"]
        rel_mesh_path = data["rel_mesh_path"]
        joint_group_idxs = data["joint_group_idxs"]
        joint_connections = np.array(data["connections"])

        if "optimised_joints" in data:
            print("Using optimised points.")
            joints = np.array(data["optimised_joints"])
        else:
            dataset_path = "/datasets/deformingthings4d"
            anime_file = os.path.join(dataset_path, rel_mesh_path)
            def4d_data = load_deformingthings4d_data(anime_file)
            vertices = def4d_data["vertices"]
            offsets = def4d_data["offsets"]
            joints = compute_joints(vertices, offsets, joint_group_idxs)

        if joints.shape[0] % sequence_length != 0:
            before = joints.shape[0]
            target_frames = (
                math.ceil(joints.shape[0] / sequence_length) * sequence_length
            )
            joints = interpolate_frames(joints, target_frames)
            print(f"Interpolating more points: {before} -> {joints.shape[0]} frames.")

        if model == "dragonOF2":
            # Invert Y
            joints[:, :, 1] = -joints[:, :, 1]
        else:
            # Swap Y and Z
            temp = joints[:, :, 2].copy()
            joints[:, :, 2] = joints[:, :, 1]
            joints[:, :, 1] = temp

        random_angle = np.random.randint(0, 360)
        print(f"Finding optimal scaling value for angle {random_angle}...")

        found_scaling_val = None
        s_min = 1.0
        # s_min = 0.5
        s_max = 10.0
        s_step = 0.5
        for scaling_val in np.arange(s_min, s_max, s_step):

            # Get a single set of camera parameters for all frames
            camera_parameters = calculate_overall_camera_parameters(
                joints,
                scalar_value=scaling_val,
                sensor_size=36,
                focal_length=35,
                angle=random_angle,
            )

            # Get intrinsic matrix
            K = calculate_intrinsic_matrix(
                focal_length=camera_parameters["intrinsics"]["focal_length"],
                sensor_size=camera_parameters["intrinsics"]["sensor_size"],
                image_size=IMAGE_SIZE,
            )

            # Get extrinsic matrix
            extrinsic = calculate_extrinsic_matrix(
                position=camera_parameters["extrinsics"]["position"],
                orientation=camera_parameters["extrinsics"]["orientation"],
            )

            # Do transformation to camera as origin
            cam_3d = do_transform_to_cam(
                points=joints,
                extrinsic=extrinsic,
            )

            # Projection
            if projection_strategy == "perspective":
                proj_2d = do_projection_from_cam(
                    points=cam_3d,
                    intrinsic=K,
                )
            elif projection_strategy == "weak_perspective":
                focal_length = camera_parameters["intrinsics"]["focal_length"]
                depth_avg = cam_3d[..., -1].mean(axis=-1, keepdims=True)
                depth_avg = depth_avg * 1000  # Meters to millimeters
                scale = focal_length / depth_avg
                proj_2d = cam_3d[..., :2] * scale[..., None]
            else:
                raise Exception("Unsupported projection strategy.")

            # Scaling
            if scaling_strategy is None:
                proj_3d = cam_3d.copy()
                scaling_factor = np.ones((cam_3d.shape[0]))
            elif isinstance(scaling_strategy, PixelSpaceScaling):
                root_joint = ROOT_JOINT_LOCATIONS[model]
                proj_3d, scaling_factor = scaling_strategy(proj_2d, cam_3d, root_joint)
            elif isinstance(scaling_strategy, RelativeScaling):
                proj_3d, scaling_factor = scaling_strategy(cam_3d)
            elif isinstance(scaling_strategy, AbsoluteScaling):
                proj_3d, scaling_factor = scaling_strategy(cam_3d)

            # If the perspective projection is not valid (ie. points are outside view)
            # then try the whole process again but with a larger scaling value.
            if projection_strategy == "perspective":
                _max = proj_2d.max(axis=0).max(axis=0)
                _min = proj_2d.min(axis=0).min(axis=0)
                if (_min < 0).any() or _max[0] > IMAGE_SIZE[0] or _max[1] > IMAGE_SIZE[1]:
                    continue

            found_scaling_val = scaling_val
            break

        if found_scaling_val is not None:
            print(f"Found: {found_scaling_val}")
        else:
            raise Exception(
                "Did not find a suitable scaling value. Try increasing the maximum scaling value."
            )

        data["points"] = {
            "2d": proj_2d.tolist(),  # type: ignore
            "3d": proj_3d.tolist(),  # type: ignore
            "cam_3d": cam_3d.tolist(),  # type: ignore
            "scaling_factor": scaling_factor.tolist(),  # type: ignore
        }
        data["camera"]["intrinsic"] = K.tolist()  # type: ignore
        data["camera"]["extrinsic"] = extrinsic.tolist()  # type: ignore
        data["camera"]["width"] = IMAGE_SIZE[0]
        data["camera"]["height"] = IMAGE_SIZE[1]

        output_path = file_path.replace(data_dir, output_dir)
        save_data(data, output_path)

        print(
            f"Saved camera parameters, 3D positions, and 2D projections of joints to {file_path}."
        )
