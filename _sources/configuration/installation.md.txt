# Installation of Dependencies


## Installation of Ubermag
If you want to check the documentation of installation of Ubermag on the official site, please click [Here](https://ubermag.github.io/installation.html)

### Conda environment
It's recommended that you build a new conda virtual env here because it's suggested by Ubermag. In this case, the environment can be easier to make.
Note that the Python version should be larger or equal to 3.8.

#### Create a new conda env
```
$ conda create -n ubermag_env python=3.8
$ conda activate ubermag_env
```

#### Installing Ubermag

Now, you can install any PACKAGE by running the following command. We show the installation of the ubermag meta-package. To install individual packages, replace ubermag with the name of the package that you want to install.

```
$ conda install --channel conda-forge ubermag
```

### Testing of Ubermag Env
You can test the installation by running:
```
$ python -c "import ubermag; ubermag.test()"
```
Again, replace ubermag with the name of your package if you do not install the meta-package. If no errors are reported, the installation was successful.


## Installation of Other Dependencies
For our project, just run the following command, the packages can be easily installed.
```
$ conda install imageio
```