# Role-Separated Human Activity Recognition with Shared Boundary-Flux Evidence and Descriptor-Specific Statistical Modulation

<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/8edb01f3-751a-486c-bba6-c1dc8df2dc24" />

This repository implements the methodology proposed in the paper "Role-Separated Human Activity Recognition with Shared Boundary-Flux Evidence and Descriptor-Specific Statistical Modulation"


## Paper Overview

**Abstract**: Wearable inertial windows contain complementary temporally ordered patterns and compact statistical summaries, yet hybrid HAR architectures often vary input representations, temporal operators, and adaptive control simultaneously, obscuring their individual roles. We present Boundary-Flux-Aware HAR (BFA-HAR), a role-separated architecture in which a shared Boundary-Flux representation of signal level and first- and second-order temporal variation is processed by a Local Context CNN and a complete-window attention encoder. In parallel, a low-dimensional Statistical Summary Representation (SSR) summarizes mean, dispersion, and autocorrelation information. Adaptive modulation is restricted to the SSR pathway and resolved independently for its three descriptor groups, while both temporal representations remain continuously available. BFA-HAR achieved Macro-F1 scores of 0.9710, 0.9819, 0.9684, and 0.9993 on UCI-HAR, WISDM, MotionSense, and MHEALTH, respectively. A controlled four-way temporal-input comparison showed that the shared Boundary-Flux configuration consistently achieved the highest Macro-F1, while a complementary factorial comparison showed that restricting adaptive control to the SSR pathway further improved performance. Parameter-matched comparisons further showed that groupwise SSR modulation outperformed shared-scalar modulation. Coefficient analysis showed distinct modulation patterns, while forced-control interventions revealed descriptor-specific sensitivities. Together, these results validate a role-separated design in which heterogeneous temporal encoders share complete Boundary-Flux evidence and descriptor-specific adaptive control is confined to the statistical pathway, with practical execution demonstrated on Raspberry Pi devices.


## Dataset
This repository does not include datasets. Please download them from the official sources below and configure the dataset path accordingly.
- **UCI-HAR** dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **MotionSense** dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **MHEALTH** dataset is available at https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- **WISDM** dataset is available at https://www.cis.fordham.edu/wisdm/dataset.php

## Requirements
```
torch==2.6.0
numpy==2.3.4
scikit-learn==1.7.2
matplotlib==3.9.2
seaborn==0.13.2
pandas==2.3.3
```
To install all required packages:
```
pip install -r requirements.txt
```
## Codebase Overview
- [`model.py`](model.py) : Implementation of the proposed BFA-HAR architecture, including the complete-window attention encoder, Local Context CNN, SSR pathway, and descriptor-group-specific SSR modulation.
- [`Comparison/`](Comparison) : Baseline models used in the unified benchmark.


## Citing this Repository
If you use this code in your research, please cite:
```
@article{BFA-HAR,
  title   = {Role-Separated Human Activity Recognition with Shared Boundary-Flux Evidence and Descriptor-Specific Statistical Modulation},
  author  = {Jimin Kim and Myung-Kyu Yi},
  journal = {},
  volume  = {},
  number  = {},
  pages   = {},
  year    = {},
  publisher = {}
}
```

## License
This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please contact:
  - Jimin Kim: sispo3314@gmail.com
