# Vector Embedding Project for Echocardiogram

## Preparation for analysis
### Install envoroment
```
conda env create -f environment.yml
```
### Make local dir for analysis
```
    mve-echo-data/
    ├── csv/
    ├── dataset/
    └── result/
```
### Get data
    - (public) MIMIC-IV
    - (public) MIMIC-IV-ECHO
    - (consortium data) MIMIC-IV-ECHO Structured Measurement

### Get the foundation models
    1. EchoPrime (https://github.com/echonet/EchoPrime)
        - echo_prime_encoder.pt
        - view_classifier.ckpt

### Analysis process
    1. data_preprocessing
        - step00 ~ satep09
    2. ve_mve
        - step00: mve traioning and, ve generation and initial evaluation
        - step01 ~ step06: analysis