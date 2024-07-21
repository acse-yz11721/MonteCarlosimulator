# Micromagnetic Simulation Tool
[![MIT License](https://img.shields.io/static/v1?label=license&message=MIT&color=orange)](https://opensource.org/licenses/mit-license.php)

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This is is a Python software that implements Monte Carlo simulation for micromagnetic model, with Jupyter Notebook postprocessing codes.

## Table of Contents

- [Background](#background)
- [Installation](#Installation)
- [Testing](#Testing)
- [Usage](#usage)
  - [Scenario](#scenario)
  - [Simulation](#simulation)
  - [Visualisation](#visualisation)
- [Author](#Author)
- [License](#license)

## Background

This software tool had the ability to implement a Monte Carlo simulation of the micromagnetic model on a 3D lattice. Magnetic free energy of the system is able to have Exchange energy, DMI energy, Zeeman energy and Anisotropy energy or any combination of these terms. The output can be  .npy files with the magnetisation vector field. In addition you can plot vector fields and visualize your vector field by generating a gif.

## Installation

Be sure to have Python 3.10 or newer installed on your machine. To install the module and any pre-requisites for the tool to function

```
conda env create -f environment.yml
conda activate micromagneticsmodel
```

or you can (conda) install the packages yourself

```
conda install --channel conda-forge ubermag
conda install imageio
conda install pytest
conda install jupyter
```

## Testing

To run the pytest test suite
```
python -m pytest tests
```

If you want to run test seperately

- run test for energy equations

  ```
  python -m pytest tests/test_energy.py
  ```

- run test for the simulator

  ```
  python -m pytest/test_simulator.py
  ```

## Usage

The basic usage of this software is explained in tutorial.ipynb file. We recommand you to create a new jupyter notebook follow the instructions mentioned in tutorial.ipynb file.

After you implement the simulation, the  'relax ' state of the magnetisation vector field will be saved in ./iterations/ directory if you choose to save. In addition, you can choose to plot the field on different planes. The function to plot the magnetisation vector field is written as:

```
field = plot_field(my_cool_mesh, my_cool_m, 'z', value=10-9, save_path="./plots/", save_name=save_name)
```

The mesh and magnetisation will be passed to plot_field function, firstly. Then we specify which component of the field you want to plot. For example, 'z' and value=10-9, it means that you get a plot of the field's z components at a vertical distance of 10e-9 from the origin. We recommand you to save the magnetisation vector field in ./iterations/ directory.

If you want to visualize the the magnetisation vector field you just saved. Yo need to make sure to close save_plot.ipynb file. After that  you could run visualization.ipynb. There is a example magnetisation vector field of size (30, 30, 2, 3).

![plane](https://github.com/ese-msc-2021/irp-yz11721/blob/main/pics/plane.gif)



## Author

   - Yuhang Zhang

## Lisense

This project is licensed under the MIT License  - see the [LICENSE.md](license.md) file for details
