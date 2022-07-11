# g2p2022

This repo has the relevant files to recreate our entry for the SIGMORPHON 2022 G2P shared task.

## Three systems

We report three different systems in the paper.

1. "vanilla" transformer run on CMU-dict English data. This is the file `vanilla.py`.
1. OpenNMT transformer model run on the shared task data. The configuration files are located in the `nmt` directory. This model was run in a docker container and instructions for creating the image are below.
1. Phonetisaurus model run on the shared task data. Configuration files are in the `phone` directory. This model was also run in a docker container and instructions for creating the container are below.

## OpenNMT container

First, create a docker container:

```bash
docker run -it \
  --gpus all \
  --name nmt \
  -v /data/:/mhdata \
	-v /home/mhammond/sigmorphon2022:/mhsig \
	pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
```

Run these steps:

```bash
apt update
apt upgrade
apt install vim
apt install wget
```

Then install:

```bash
pip install OpenNMT-py
```

## Phonetisaurus container

First, create a docker container:

```bash
docker run -it \
  --gpus all \
  -p 8888:8888 \
  --name phone \
  -v /data/:/mhdata \
  -v /home/mhammond/sigmorphon2022/:/mhsig \
  ubuntu:20.04
```

Next clone the Phonetisaurus repo:

https://github.com/AdolfVonKleist/Phonetisaurus

Do the steps from the `readme.md` file there adding these bits:

```bash
apt install wget
apt install python3-pip
apt install vim
...
vim test.wlist
```
