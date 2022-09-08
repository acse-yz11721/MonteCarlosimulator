# Run Pytest


## Pytest Instances

In ours project, the scripts of Pytest Instances are placed in the **tests** folder. The Pytest Package was automaticalliy installed when you were configuring the Ubermag Env.


## Try to test

Now let's open the terminal which is working in the top layer of the path of our project. Just type:
```
$ pytest tests/
```
It's certain you will get an AttributeError which indicates the miss finding of Module "continuous_model". That's because the python interpreter hasn't treat the "continuous_model" dir as a source code package yet.
Let's solve it.

## Configurate the project

The **pyproject.toml** is created in order to tell the interpreter how to solve the dependency relationship of the codes. The following snippet complains the src and the test path of the project.

```toml
[tool.pytest.ini_options]
pythonpath = [
    "continuous_model",
]
testpaths = [
    "tests",
]
```

Now typing a single pytest can lead to the correct testing process.
```
$ pytest
```