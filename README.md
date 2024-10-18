# gpuselect

Automatic selection of idle GPUs by modifying `CUDA_VISIBLE_DEVICES`.

The intended use for this package is to make GPU allocation simpler on shared servers with multiple GPUs in the absence of any more complex resource scheduling framework (and to help reduce the number of idling GPUs).

The idea is to avoid having to manually allocate GPUs to users and instead applying a few simple default rules to pick idle GPU(s) each time you run a script, updating the value of `CUDA_VISIBLE_DEVICES` to reflect the selected device(s).

The Python package contains a single core method named `gpuselect()`. Called with default parameters, this will:
  * enumerate all GPUs in the system through the [NVML API](https://docs.nvidia.com/deploy/nvml-api/index.html)
  * exclude any GPU with nonzero device or memory utilization
  * exclude any GPU with a nonzero process count
  * pick a single GPU at random from the remaining list
  * set `CUDA_VISIBLE_DEVICES=n` where `n` is the selected GPU's device ID

Adding this to an existing Python script should only require a couple of lines of code in most cases:
```python
# import the module and call gpuselect with default parameters
from gpuselect import gpuselect
gpuselect()
# CUDA_VISIBLE_DEVICES is now updated

import torch
...
```

Beyond the defaults, you can select multiple GPUs, select GPUs based on name or device ID, or pass in a custom filter function if you want more control.

There's also a CLI to act as a wrapper for launching other processes with a modified `CUDA_VISIBLE_DEVICES` value:

```shell
# sets CUDA_VISIBLE_DEVICES as described above, then runs the command 
# as a subprocess (i.e. in the same environment)
gpuselect -- python3 my_scripy.py arg1 arg2
```

See below for further examples.

## Installation

Clone the repo and run `pip install -r requirements.txt` in a virtualenv.

This will install:
  * A Python module called `gpuselect`
  * A CLI tool also called `gpuselect`

### Usage in Python scripts

#### Request any single idle GPU (default parameters)
The defaults assume you want to use a single GPU with zero utilization and zero processes active, of any available type. To do this you can just import the module and call `gpuselect` with default parameters:

```python
from gpuselect import gpuselect
gpuselect() 
# CUDA_VISIBLE_DEVICES is now updated. The method also returns
# its new value (e.g. "1").

# import other packages here
```

> **Note**: it's probably best to put this import at the top of your script, because some packages (`e.g. pytorch`) will query the value of `CUDA_VISIBLE_DEVICES` on import and so updating it afterwards has no effect. 


#### Request GPU(s) of a specific type

The `name` parameter defaults to `None`. If you set it to a string, it can be used to match partial or complete device names. GPUs that don't have matching names will be excluded.

```python
from gpuselect import gpuselect
# select an idle GPU with "A6000" in the name, e.g. "NVIDIA RTX 6000 Ada Generation"
gpuselect(name="A6000")

# Or to do the same thing with multiple GPUs:
gpuselect(count=2, name="A6000")
```

#### Changing default utilization thresholds

There are 3 utilization threshold filters that are all set to zero by default:
  * `util`: device utilization percentage
  * `mem_util`: device memory utilization percentage
  * `processes`: number of processes using the device

If you want to relax any of these thresholds, you can pass a new value to `gpuselect`:

```python
from gpuselect import gpuselect
# allow selection of GPUs with up to 10% device utilization and up to 1
# other active process
gpuselect(util=10, processes=1)
```

#### Requesting specific GPU(s)

If you need to select GPUs by device ID for any reason, you can set the `devices` parameter to one or more device IDs. 

```python
from gpuselect import gpuselect
# select device 0, if possible (this still applies the usual utilization filters)
gpuselect(devices=0)

# select devices 0,1 if possible (this still applies the usual utilization filters)
gpuselect(devices=[0, 1])
```

#### Custom selection

You can pass a `Callable` to `gpuselect` through the `selector` parameter. The object should take `gpuselect.GpuInfo` as a parameter and return `True` to include the GPU or `False` to exclude it. 

```python
from gpuselect import gpuselect, GpuInfo

def custom_select(gpu: GpuInfo) -> bool:
    # a GpuInfo object has fields:
    #   device (int): device ID
    #   name (str): device name
    #   util (int): device utilization percentage
    #   mem_util (int): device memory utilization percentage
    #   processes (int): device process count

    # match any device with "3090" in the name and <= 5% utilization
    return "3090" in gpu.name and gpu.util <= 5

# note that this effectively overrides all other filtering parameters except for `count`
gpuselect(selector=custom_select)
```

#### Handling errors
```python
# by default, this will throw an exception if the requested number of GPUs can't be found
try:
    gpuselect(count=4)
except Exception as e:
    print(f"Failed to select GPUs: {e}")

# if you want to suppress this, you can pass silent=True. in this case you can
# check if it failed to find devices by checking if the return value is empty
if gpuselect(count=4, silent=True) == "":
    print(f"Failed to select GPUs")
```

#### Debug logging

```python
import logging
from gpuselect import gpuselect
logging.getLogger("gpuselect").setLevel(logging.DEBUG)
gpuselect()
```

### Reading GPU state

The `gpuselect` module also exposes a function called `gpustatus` which you can use to retrieve GPU state in your own code. This can be used independently of the `gpuselect` functionality.

The function returns a list of `dicts`, one for each GPU. Each `dict` contains the following keys and values:

  * `device`: device ID (`int`)
  * `name`: device name (`str`)
  * `utilization`: device utilization % (`int`)
  * `mem_utilization`: device memory utilization % (`int`)
  * `processes`: processes using device (`int`)
  * `performance_state`: device performance state, 0-15 (`int`)
  * `power_usage`: device power draw in milliwatts (`int`)
  * `temperature`: device temperature in deg C (`int`)

By default, this will only report on the GPUs visible according to `CUDA_VISIBLE_DEVICES`. Depending on the context you're running the code in, you can also choose to view data on all GPUs:

```python
from gpuselect import gpustatus
# respect CUDA_VISIBLE_DEVICES
gpu_info = gpustatus()

# use all GPUs accessible to the NVML API
all_gpu_info = gpustatus(only_cvd=False)
```

### Running arbitrary commands

If you want to use `gpuselect` in a context where inserting it into a Python script is not possible, the CLI version supports most of the same functionality. 

```shell
# Update CUDA_VISIBLE_DEVICES in your current shell
gpuselect --count 1 --name 3090
```

More usefully, you can use it as a wrapper for arbitrary commands by appending extra arguments after a `--` delimiter, which allows `gpuselect` to update `CUDA_VISIBLE_DEVICES` before the new process starts:

```shell
# Run a command with default gpuselect parameters (i.e. any idle GPU)
gpuselect -- python foo.py

# The same thing but requesting 2 GPUs
gpuselect --count 2 -- python foo.py

# Requesting an A6000
gpuselect --name A6000 -- python foo.py

# Relaxed device and memory utilization percentages
gpuselect --util 5 --mem_util 10 -- python foo.py
```

### Embedding into shell commands

Another possible use case is running shell commands where you need a list of CUDA device IDs. For example, launching a Docker container:

```bash
#!/bin/bash
# launch a Docker container, assigning it a pair of idle 3090s 
# (the way Docker parses the --gpus value is a bit weird, which is the reason for the quoting)
docker run --gpus \"device=$(gpuselect --count 2 --name 3090)\" --rm -it gpu_container nvidia-smi
```
