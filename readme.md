# The D2L Generalized Policy Learner

## Installation

The entire D2L pipeline runs in Python3 and relies on the following dependencies:

* The [FS planner](https://github.com/aig-upf/fs-private/).
* Some MaxSat solver such as [OpenWBO](http://sat.inesc-id.pt/open-wbo/).
* The [Tarski](https://github.com/aig-upf/tarski/) planning problem definition module.
* CMake

The provided [Dockerfile](containers/Dockerfile) recipe lists all necessary instructions to install D2L
on an Ubuntu box. You can either use it through Docker, or simply follow the commands to install the pipeline
locally on your machine.


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
```

The configuration of each experiment can be inspected by looking at the experiment file.

## AAAI21 Paper Experiments
The following is a list of the concrete experiments used in the results table
of our AAAI'21 paper,
Guillem Francès, Blai Bonet, Hector Geffner,
[Learning General Policies from Small Examples Without Supervision](https://arxiv.org/abs/2101.00692).

```shell script
  ./run.py blocks:clear   # Q_clear
  ./run.py blocks:on      # Q_on
  ./run.py gripper:small  # Q_grip
  ./run.py reward:small   # Q_rew
  ./run.py delivery:small # Q_deliv
  ./run.py visitall:small # Q_visit
  ./run.py spanner:small  # Q_span
  ./run.py miconic:small  # Q_micon
  ./run.py blocks:all_at_5     # Q_bw
```

## Using the Docker image 
In order to use the provided Docker image, you need [a Docker installation](https://docs.docker.com/engine/installation)
on your machine.

### Building the image

Build the docker image with the following command from the repo root:
```shell script
sudo docker build -t d2l -f containers/Dockerfile .
```

### Running the image

You can open a terminal on the image for inspection or debugging by running
```shell script
sudo docker run --entrypoint bash -it  d2l
```

Or you can directly run a concrete experiment, e.g.:
```shell script
sudo docker run --rm d2l blocks:clear
```