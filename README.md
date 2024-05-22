# Winding Clearness for Differentiable Point Cloud Optimization

## Introduction
This document contains the source code for our TVCG submission titled “Winding Clearness for Differentiable Point Cloud Optimization”.

- The `./optimization` folder includes the code for single point cloud optimization proposed in Section 5.1.
- The `./learning` folder includes the code for incorporating winding clearness as a geometric constraint in the standard DDPM proposed in Section 5.2.



## Prerequisites:
The code execution requires the following dependencies. 
These versions are not specified. Other compatible versions may also work, provided that the torch versions align with the available cuda toolkit.
```
python==3.10
pytorch==2.0.0
torchvision==0.15.1
cudatoolkit==11.8
tqdm==4.32.1
open3d==0.16.0
trimesh=4.0.8
scipy==1.11.4
```
These dependencies are sufficient for the single point cloud optimization approach.
For the diffusion-based approach, PyTorchEMD is required and can be installed with the following commands:
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-310m-x86_64-linux-gnu.so .
```
Please ensure that the gcc version aligns with 
the system configuration as required in 
"learning/metrics/ChamferDistancePytorch/chamfer3D/setup.py"
and "learning/modules/functional/backend.py"

## Single Point Cloud Optimization
Execute the following commands to run the single point cloud optimization method:
```bash
python optimization_based_2D.py --dataroot ./input/2D --result_label 2D
python optimization_based_3D.py --dataroot ./input/3D --result_label 3D 
python optimization_based_3D.py --dataroot ./input/3D2 --result_label 3D2 --lambda_k 20.0
```
Results will be stored in the "output" folder.

## Diffusion-based Method
The baseline of the diffusion-based method is [PVD](https://alexzhou907.github.io/pvd).
All relevant data and pretrained models are accessible or can be trained as described in the GitHub page of PVD.
We include here the fine-tuned models of our method for the chair category and all the 55 classes at "./new_joint_train_model_chair" and "./new_joint_train_model_all", respectively.

The generated point clouds for "PVD" and "PVD+Ours" of the chair category is in "./results".

To run the train and test code, the ShapeNet point cloud are requires, which can be downloaded [here](https://github.com/stevenygd/PointFlow).
After putting pretrained model "chair_1799.pth" of PVD into "./ckpt/generation", we can run the following command to generate the results of "PVD":
```bash
python test_generation_chair.py
```
and the following command to generate the results of "PVD+Ours":
```bash
python test_generation_regularization_chair.py
```
The generated results will be located in "output/test_generation_regularization_chair".
The joint training strategy of our method can be executed by:

```bash
python train_generation_joint_chair.py
```