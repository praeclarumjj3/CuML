# CuML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repo contains code for various ML/DL algorithms/models implemented using **CUDA** as a semester project for **CSN-221: Computer Architecture**

## Repository Structure

- [CuCNN](https://github.com/praeclarumjj3/CuML/CuCNN/): Contains a CNN implemented using CUDA on MNIST dataset.

## Setup

Steps to test the code on **Google Colab**.

- Mount google drive to the colab notebook:

```
from google.colab import drive
drive.mount('/content/gdrive')
```

- Clone the repo using:
```
! git clone https://github.com/praeclarumjj3/CuML.git
```

- Navigate to the folder of the algorithm you want to test using:
```
%cd /content/gdrive/My Drive/CuML/[Folder_Name]
```

- Compile the code using:

```
! make
```

- Run the executable using:
```
! ./[Algotithm_Name] 
```