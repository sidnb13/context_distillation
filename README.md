# prompt_inject2

## installation

Install with conda (cpu, tpu, or gpu) or docker (gpu only).

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate prompt_inject2
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate prompt_inject2
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate prompt_inject2
python -m pip install --upgrade pip
python -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**install with docker (gpu only):**
* install docker and docker compose
* make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
docker compose build
docker compose run prompt_inject2
```

And then in the new container shell that pops up:

``` shell
cd prompt_inject2
```
