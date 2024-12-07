# Z1Env

## Installation

First, clone this package

```bash
git clone https://github.com/BolunDai0216/Z1Env.git
```

Then, install the package using the command below

```bash
cd Z1Env
python3 -m pip install .
```

## Usage

Shortcuts to control the sim screen
https://github.com/Gabo-Tor/pybullet-keyboard-shortcuts
On mac i needed to install proxsuite from source and 
since I have python with pyenv
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_PYTHON_INTERFACE=ON  -DCMAKE_INSTALL_PREFIX=$(pyenv prefix)


## How to generate the `.obj` files

- Go to `unitree_ros/robots/z1_description/meshes/visual`
- Load the corresponding `.dae` file in Blender
- Rotate the model by 90 degrees around the x-axis
- Export the model as a `.obj` file