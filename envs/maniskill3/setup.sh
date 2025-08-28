#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

conda env create -f ./envs/maniskill3/environment.yaml
conda activate maniskill3_env
