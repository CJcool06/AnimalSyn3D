<div align="center">

  # AnimalSyn3D 

</div>

**AnimalSyn3D** is a synthetic dataset containing sparse 3D keypoints and skeleton configurations for a variety of different animals. Each animal has various animated movements that are split into 48-frame sequences with keypoint correspondences across frames.

<div align="center">
  <a href="https://github.com/CJcool06/AnimalSyn3D/releases">Download Dataset Here</a>
</div>

<!--
**\<Put main graphic here\>**  
-->

<br />
<div align="center" style="padding-top: 30px;">

  ## Dataset Overview

</div>

There are 13 animals in our dataset. Each animal may have a different number of joints, skeleton configuration, number of frames, and types of animations.

<br />

<div align="center">

  |  | Bear | Buck | Bunny | Chicken | Deer | Dog | Elk | Fox | Moose | Puma | Rabbit | Raccoon | Tiger | Total |
  | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
  | Animations | 67 | 42 | 45 | 7 | 56 | 65 | 67 | 37 | 59 | 68 | 45 | 54 | 66 | 678 |
  | Frames | 4,464 | 3,168 | 3,072 | 432 | 3,648 | 4,128 | 5,328 | 2,304 | 3,792 | 5,808 | 3,072 | 4,176 | 4,992 | 48,384 |
  | Joints | 21 | 27 | 25 | 19 | 29 | 22 | 26 | 26 | 29 | 26 | 25 | 28 | 27 | 330 |

</div>
<br />


<div align="center" style="padding-top: 30px;">

  ## Data Format 

</div>
<!-- <br /> -->

The dataset directory is organised by animal. Each animal has a set of `.json` files, each holding data for a specific animation sequence.

### <u>Folder Structure</u>
```
AnimalSyn3D/
└── <animal>/
    └── <animation>.json
```


### <u>Animation File</u>
```
KEY                 | DESCRIPTION
---------------------------------------------------------
model               | Name of the animal model.
animation           | Name of the animation sequence.
rel_mesh_path       | Relative path to the model mesh.
joint_group_idxs    | Indices used to label the joints.
connections         | Animal skeleton.
optimised_joints    | Joint locations after optimisation.
camera              | 
├── intrinsic       | The intrinsic matrix.
├── extrinsic       | Extrinsic matrix.
├── width           | Camera width.
└── height          | Camera height.
points              | 
├── 2d              | Projected 2D points.
├── 3d              | Scaled 3D points in camera space.
├── cam_3d          | Unscaled 3D points in camera space.
└── scaling_factor  | Scaling applied to 3D points.
```

<br />
<div align="center" style="padding-top: 30px;">

  ## BibTeX

</div>

Please cite us if you find AnimalSyn3D useful in your research.

```
@InProceedings{fusco2024objectagnostic3d,
    author    = {Christopher Fusco and Mosam Dabhi and Shin-Fang Ch'ng and Simon Lucey},
    title     = {Object Agnostic 3D Lifting in Space and Time},
    journal   = {arxiv},
    year      = {2024}
}
```

<br />
<div align="center" style="padding-top: 30px;">

  ## Acknowledgements

</div>

This dataset is entirely based on the models and animations provided by <a target="_black" href="https://github.com/rabbityl/DeformingThings4D?tab=readme-ov-file">DeformableThings4D</a>.
