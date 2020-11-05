# The D2L Generalized Policy Learner


## Installation

The whole pipeline runs in *Python3* and relies on some dependencies; most notably:

* The [FS planner](https://github.com/aig-upf/fs-private/).
* Some MaxSat solver such as [OpenWBO](http://sat.inesc-id.pt/open-wbo/).
* The [Tarski](https://github.com/aig-upf/tarski/) planning problem definition module.
* CMake

The provided [Dockerfile](containers/Dockerfile) recipe lists all necessary instructions to install D2L
on an Ubuntu box. You can either use it through Docker, or simply follow the commands
to install it locally.


## Usage

Individual experiments are on the [experiments](experiments) folder, grouped by domains.
See [experiments/gripper.py](experiments/gripper.py) for an example.
A file such as `gripper.py` contains different experiment configurations for learning in the Gripper domain.
We invoke the pipeline with `run.py <domain>:<experiment-name> <pipeline-steps-to-be-executed>`,
where the last parameter is an optional list of experiment step IDs (e.g.: 1 2 3). 
If no step ID is specified, the entire experiment is run.
Example invocations:

```shell script
  # Learn to clear a block
  ./run.py blocks:clear

  # Learn to stack two blocks
  ./run.py blocks:on

  # Learn to solve the standard Gripper domain
  ./run.py gripper
```

The configuration of each experiment can be inspected by looking at the experiment file.

## Using the D2L Docker image 
We can also run experiments from within the Docker image.
[TODO]

