## Phyiosnets EDF-X database hypnograms

This directory includes the hypnograms from the edfx-sleep database from [Physionet](https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/). Use the file [convert_edfx_database.py](./convert_edfx_database.py) to extract the Fpz-Cz EEG channel, truncate the extensive pre- and post sleep Wake intervals, which can be done with the function `prepare_edfx_database()`. Note that this folder uses a slightly different format than AutoSleepScorer and might not be compatible.


The following conversion is used. One annotation is saved per second of data.

| EDF | Numerical |
|-----|-----------|
| W | 0 |
| S1 | 1 |
| S2 | 2 |
| S3 | 3 |
| S4 | 4 |
| REM |5 |
| Artefact | 6 |
| Movement | 7 |
| Unkown | 9 |

