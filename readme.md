# DreamWaQ

## Description
This repo contains implementation of the paper [Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
  
## Installation
This repo requires the following packages:
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- Legged Gym
- RSL-RL

To install Isaac Gym, go to the link and follow the instructions on the page.

1. clone this repo

2. 
```bash
  cd rsl_rl-1.0.2
  pip install -e .
  cd ..
```
3. 
```bash
  cd legged_gym
  pip install -e .
  cd ..
```

## Usage
To train your robot run
```bash
    python3 legged_gym/scripts/train.py --task=[robot name]  
```  

To evaluate the trained policy run
```bash
    python3 legged_gym/scripts/play.py --task=[robot name]
```  
  
![Go1](DwaQ_stairs.gif)!

## Configuration
Requires python 3.8 and numpy version<=1.24.
