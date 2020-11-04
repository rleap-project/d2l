
# SLTP/D2L: Docker Images

## Building the image


```lang-bash
(sudo) docker build -t sltp .
```


## Running the image

```lang-bash
# Run an interactive shell to inspect the image:
docker run -it d2l bash

# Run a concrete example:
docker run -it d2l /workspace/d2l/experiments/d2l/run.py blocks:clear
```