# AutoSleepScorerDev
Please find the package here https://github.com/skjerns/AutoSleepScorer


This repository contains all the files for training and preprocessing. It's more a file dump for me than a usable repository. Most files are absolutely undocumented.

I have created a small script that trains a model using the EDFx-database: `run_sample.py`.

## File description

|File Name   | Description   |
|---|---|
|main files||
|`run_sample.py`| Minimalistic script to download and train on the EDFx   |
|`keras_utils.py`| Contains the CV and training routine, generators and checkpoints. Everything that uses keras   |
|`tools.py`|  Helper functions   |
|`models.py`| All of the models used during my thesis  |
|`sleeploader.py` | Creates sleep database from eeg files   |
|`edfx_database.py`|  Script for preparing the edfx database   |
|others| |
|`test_xxx.py` | Some costum scripts for training under different conditions  |