#!/bin/bash
cd /mnt/data/docker/tts || exit
source .venv/bin/activate
python train.py
