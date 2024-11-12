import open3d as o3d
import numpy as np
from glob import glob
import os
import argparse
import ast
from utils.dataset_utils import load_deformingthings4d_data, save_data
from utils.dataset_constants import JOINT_CONNECTIONS


def save_list(file_path, list_data):
    with open(file_path, "w") as f:
        for item in list_data:
            f.write(f"{item}\n")

def read_list(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        list_data = [ast.literal_eval(line.strip()) for line in f]
    return list_data

parser = argparse.ArgumentParser(description="Label some animal joints.")
parser.add_argument("dataset_path", type=str, help="The path to the dataset.")
parser.add_argument("model", type=str, help="The name of the animal model.")
parser.add_argument("--animation", type=int, default=0, help="The animation to view.")
parser.add_argument("--frame", type=int, default=0, help="The frame to view.")
parser.add_argument("--skeleton", action="store_true", help="Show the skeleton connections.")

args = parser.parse_args()
dataset_path = args.dataset_path
model = args.model
animation = args.animation
frame = args.frame
show_skeleton = args.skeleton

animal_dirs = glob(os.path.join(dataset_path, "animals", f"{model}*"))
anime_file = glob(os.path.join(animal_dirs[animation], "*.anime"))[0]
print(f"Animation file: {anime_file}")

def4d_data = load_deformingthings4d_data(anime_file)
vert_data = def4d_data["vertices"]
face_data = def4d_data["faces"]
offset_data = def4d_data["offsets"]

original_vertices = vert_data.copy()
vertices = vert_data.copy()
colours = np.ones_like(vertices) * [0, 0, 1]

vertices += offset_data[frame]

selected_groups_idx = []
buffer = read_list(f"points_buffer_{model}.txt")
if buffer is not None:
    selected_groups_idx = buffer

for points in selected_groups_idx:
    picked_points = vertices[points]
    joint_loc = picked_points.mean(axis=0)
    vertices = np.append(vertices, joint_loc.reshape(1, 3), axis=0)
    colours = np.append(colours, list([[1, 0, 0]]), axis=0)

while True:

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(vertices + offset_data[i])
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colours)

    # Create a visualizer
    vis = o3d.visualization.VisualizerWithEditing()  # type: ignore
    vis.create_window()

    print(
        "CONTROLS:",
        "\n[Q] - Quit",
    )

    # Add the point cloud to the visualizer
    if show_skeleton:
        line_set = o3d.geometry.LineSet()
        # pcd.points = o3d.utility.Vector3dVector(vertices + offset_data[i])
        line_set.points = o3d.utility.Vector3dVector(vertices[len(original_vertices):])
        line_set.lines = o3d.utility.Vector2iVector(JOINT_CONNECTIONS[model])
        line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(JOINT_CONNECTIONS[model]))])
        vis.add_geometry(line_set)
    else:
        vis.add_geometry(pcd)

    # Run the visualizer
    vis.run()  # User picks points by clicking
    vis.destroy_window()

    # Get the picked points
    picked_points = vis.get_picked_points()

    if len(picked_points) == 0:
        print("No points selected, exiting viewer.")
        break

    print("Selected point IDs:", picked_points)

    picked_vertices = vertices[np.array(picked_points)]  # type: ignore
    joint_loc = picked_vertices.mean(axis=0)

    vertices = np.append(vertices, joint_loc.reshape(1, 3), axis=0)
    colours = np.append(colours, list([[1, 0, 0]]), axis=0)

    selected_groups_idx.append(picked_points)  # type: ignore

    save_list(f"points_buffer_{model}.txt", selected_groups_idx)
    print("Saved selected points to buffer.")

os.makedirs(os.path.join(f"data/{model}"), exist_ok=True)
model_dirs = glob(os.path.join(dataset_path, "animals", f"{model}*"))
for model_dir in model_dirs:
    anime_file = glob(os.path.join(model_dir, "*.anime"))[0]
    model = anime_file.split("/")[-2].split("_")[0]
    animation_name = anime_file.split("/")[-2].split("_")[1]
    data = {
        "model": model,
        "animation": animation_name,
        "rel_mesh_path": "/".join(anime_file.split("/")[-3:]),
        "joint_group_idxs": selected_groups_idx,
        "connections": JOINT_CONNECTIONS[model],
        "camera": {
            "intrinsic": [],
            "extrinsic": [],
            "width": 0,
            "height": 0,
        },
    }
    save_data(data, os.path.join(f"data/{model}/{animation_name}.json"))
    print(f"Saved to {os.path.join(f'data/{model}/{animation_name}.json')}")
