import os
import math
import pickle
import numpy as np
from string import digits
from glob import glob

from utils.dataset_utils import load_data
from utils.dataset_constants import ROOT_JOINT_LOCATIONS
from utils.preprocess_utils import abs_norm, abs_norm_3d, relative_norm


np.random.seed(999)
remove_digits = str.maketrans('', '', digits)

sequence_length = 48
split_ratio = 0.8

data_dir = "optimised_data_output"
save_dir = "AnimalSyn3D"

# norm_type = ""
norm_type = "absolute"
# norm_type = "relative"

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
    "doggieMN5",
    "raccoonVGG",
    "chickenDC",
    # "dragonOF2",
    # "crocodileOPW",
    # "duck",
    # "hippoDG",
]

exclusions = [
    "data/pumaRW/Walk2.json",
    "data/elkML/Walk4.json",
]
whitelist = []


train_sum = 0
val_sum = 0
for model in models:
    train_data = {
        "sequence_length": sequence_length,
        "joints_2d": [],
        "joints_3d": [],
        "joints_3d_cam": [],
        "scaling_factor": [],
        "actions": [],
        "camera": {
            "height": [],
            "width": [],
            "intrinsic": [],
            "extrinsic": [],
        },
        "root_joint": ROOT_JOINT_LOCATIONS[model],
    }
    val_data = {
        "sequence_length": sequence_length,
        "joints_2d": [],
        "joints_3d": [],
        "joints_3d_cam": [],
        "scaling_factor": [],
        "actions": [],
        "camera": {
            "height": [],
            "width": [],
            "intrinsic": [],
            "extrinsic": [],
        },
        "root_joint": ROOT_JOINT_LOCATIONS[model],
    }

    print("\nModel:", model)

    num_joints = 0
    joint_connections = None
    file_paths_raw = glob(os.path.join(data_dir, model, "*.json"))
    file_paths = []
    for file_path in file_paths_raw:
        if file_path in exclusions:
            print("Skipping excluded file:", file_path)
            continue
        if len(whitelist) > 0 and file_path.split("/")[-1] not in whitelist:
            print("Skipping un-whitelisted file:", file_path)
            continue
        file_paths.append(file_path)

    train_idxs = np.random.choice(len(file_paths), size=math.ceil(len(file_paths) * split_ratio), replace=False)

    for idx, file_path in enumerate(file_paths):
        # print(file_path)

        data = load_data(file_path)
        if "points" not in data:
            print(f"Skipping {file_path}. No 2D points yet.")
            continue
        
        action = file_path.split("/")[-1].split(".")[0]
        camera_height = data["camera"]["height"]
        camera_width = data["camera"]["width"]
        joints_2d = np.array(data["points"]["2d"])
        joints_3d = np.array(data["points"]["3d"])
        joints_3d_cam = np.array(data["points"]["cam_3d"])
        scaling_factor = np.array(data["points"]["scaling_factor"])
        joint_connections = np.array(data["connections"])
        camera_intrinsic = np.array(data["camera"]["intrinsic"])
        camera_extrinsic = np.array(data["camera"]["extrinsic"])
        
        num_frames, num_joints, _ = joints_2d.shape

        if norm_type == "absolute":
            joints_2d = abs_norm(joints_2d, camera_height, camera_width)
            joints_3d = abs_norm_3d(joints_3d, camera_height, camera_width)

            _min = joints_2d.min(axis=1).min(axis=0)
            _max = joints_2d.max(axis=1).max(axis=0)
            if (_min < -1).any() or (_max > 1).any():
                print("Illegal 2D normalisation...")
                print(_min)
                print(_max)

            _min = joints_3d[0, :, :3].min(axis=0)
            _max = joints_3d[0, :, :3].max(axis=0)

            if (_min < -1.0001).any() or (_max > 1.0001).any():
                print(f"File: {file_path}")
                print("Illegal 3D normalisation...")
                print("Min:", _min)
                print("Max:", _max)

        if norm_type == "relative":
            joints_2d, _ = relative_norm(joints_2d, camera_height, camera_width)
            joints_3d, _ = relative_norm(joints_3d, camera_height, camera_width)
        
        num_sequences = num_frames // sequence_length
        if num_sequences == 0:
            print(f"Skipping {file_path} with {num_frames} frames.")
            continue

        cutoff = num_sequences * sequence_length
        if idx in train_idxs:
            train_sum += num_sequences
            train_joints_2d = joints_2d[:cutoff]
            train_joints_3d = joints_3d[:cutoff]
            train_joints_3d_cam = joints_3d_cam[:cutoff]
            train_scaling_factor = scaling_factor[:cutoff]
            train_data["actions"].extend([action] * num_sequences)
            train_data["joints_2d"].append(train_joints_2d)
            train_data["joints_3d"].append(train_joints_3d)
            train_data["joints_3d_cam"].append(train_joints_3d_cam)
            train_data["scaling_factor"].append(train_scaling_factor)
            train_data["camera"]["height"].extend([camera_height] * num_sequences)
            train_data["camera"]["width"].extend([camera_width] * num_sequences)
            train_data["camera"]["intrinsic"].extend([camera_intrinsic] * num_sequences)
            train_data["camera"]["extrinsic"].extend([camera_extrinsic] * num_sequences)
        else:
            val_sum += num_sequences
            val_joints_2d = joints_2d[:cutoff]
            val_joints_3d = joints_3d[:cutoff]
            val_joints_3d_cam = joints_3d_cam[:cutoff]
            val_scaling_factor = scaling_factor[:cutoff]
            val_data["actions"].extend([action] * num_sequences)
            val_data["joints_2d"].append(val_joints_2d)
            val_data["joints_3d"].append(val_joints_3d)
            val_data["joints_3d_cam"].append(val_joints_3d_cam)
            val_data["scaling_factor"].append(val_scaling_factor)
            val_data["camera"]["height"].extend([camera_height] * num_sequences)
            val_data["camera"]["width"].extend([camera_width] * num_sequences)
            val_data["camera"]["intrinsic"].extend([camera_intrinsic] * num_sequences)
            val_data["camera"]["extrinsic"].extend([camera_extrinsic] * num_sequences)

    try:
        train_data["actions"] = np.array(train_data["actions"])
        train_data["joints_2d"] = np.concatenate(train_data["joints_2d"], axis=0).reshape(-1, sequence_length, num_joints, 2)
        train_data["joints_3d"] = np.concatenate(train_data["joints_3d"], axis=0).reshape(-1, sequence_length, num_joints, 3)
        train_data["joints_3d_cam"] = np.concatenate(train_data["joints_3d_cam"], axis=0).reshape(-1, sequence_length, num_joints, 3)
        train_data["scaling_factor"] = np.concatenate(train_data["scaling_factor"], axis=0).reshape(-1, sequence_length)
        train_data["camera"]["height"] = np.array(train_data["camera"]["height"])
        train_data["camera"]["width"] = np.array(train_data["camera"]["width"])
        train_data["camera"]["intrinsic"] = np.stack(train_data["camera"]["intrinsic"], axis=0)
        train_data["camera"]["extrinsic"] = np.stack(train_data["camera"]["extrinsic"], axis=0)
        train_data["joint_connections"] = joint_connections
        train_data["sequence_length"] = sequence_length
        train_data["norm_type"] = norm_type

        print("Training set shape:", train_data["joints_3d"].shape)
        print("Training set cam shape:", train_data["joints_3d_cam"].shape)
        print("Training camera height/width shape:", train_data["camera"]["height"].shape)

        os.makedirs(f"{save_dir}/seq{sequence_length}/{norm_type}/{model}/train", exist_ok=True)
        with open(f"{save_dir}/seq{sequence_length}/{norm_type}/{model}/train/train.pkl", "wb") as f:
            pickle.dump(train_data, f)

        os.makedirs(f"{save_dir}_3dlfm/seq{sequence_length}/{norm_type}/{model}/train", exist_ok=True)
        with open(f"{save_dir}_3dlfm/seq{sequence_length}/{norm_type}/{model}/train/asim{model}.pkl", "wb") as f:
            pickle.dump(train_data, f)
    except:
        print(f"Skipping training set for {model}.")
    
    val_data["actions"] = np.array(val_data["actions"])
    val_data["joints_2d"] = np.concatenate(val_data["joints_2d"], axis=0).reshape(-1, sequence_length, num_joints, 2)
    val_data["joints_3d"] = np.concatenate(val_data["joints_3d"], axis=0).reshape(-1, sequence_length, num_joints, 3)
    val_data["joints_3d_cam"] = np.concatenate(val_data["joints_3d_cam"], axis=0).reshape(-1, sequence_length, num_joints, 3)
    val_data["scaling_factor"] = np.concatenate(val_data["scaling_factor"], axis=0).reshape(-1, sequence_length)
    val_data["camera"]["height"] = np.array(val_data["camera"]["height"])
    val_data["camera"]["width"] = np.array(val_data["camera"]["width"])
    val_data["camera"]["intrinsic"] = np.stack(val_data["camera"]["intrinsic"], axis=0)
    val_data["camera"]["extrinsic"] = np.stack(val_data["camera"]["extrinsic"], axis=0)
    val_data["joint_connections"] = joint_connections
    val_data["sequence_length"] = sequence_length
    val_data["norm_type"] = norm_type

    print("Val set shape:", val_data["joints_3d"].shape)
    print("Val set cam shape:", val_data["joints_3d_cam"].shape)
    print("Val camera height/width shape:", val_data["camera"]["height"].shape)

    os.makedirs(f"{save_dir}/seq{sequence_length}/{norm_type}/{model}/test", exist_ok=True)
    with open(f"{save_dir}/seq{sequence_length}/{norm_type}/{model}/test/test.pkl", "wb") as f:
        pickle.dump(val_data, f)

    os.makedirs(f"{save_dir}_3dlfm/seq{sequence_length}/{norm_type}/{model}/test", exist_ok=True)
    with open(f"{save_dir}_3dlfm/seq{sequence_length}/{norm_type}/{model}/test/asim{model}.pkl", "wb") as f:
        pickle.dump(val_data, f)

print("Output dir:", save_dir)
print("Train sequences:", train_sum)
print("Val sequences:", val_sum)