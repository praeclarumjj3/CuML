# CuNaiveBayes

This folder contains the code for Naive Bayes algorithm implemented using CUDA.

## Setup

Follow the below instructions to test the code on google colab:

- Mount Google Drive to the colab notebook

```bash
from google.colab import drive
drive.mount('/content/gdrive')
``` 

- Clone the repo using 

```bash
! git clone https://github.com/praeclarumjj3/CuML.git
```

- Set GPU as the hardware accelerator using the following steps:

```bash
Runtime-> Change runtime type -> select GPU as hardware accelerator
```

- Navigate to the folder of the algorithm you wish to test

```bash
%cd /content/gdrive/My Drive/CuML/CuNaiveBayes
```

- Compile the code using

```bash
! nvcc -lcuda -lcublas *.cu -o CuNaiveBayes
```

- Run the executable using

```bash
! ./CuNaiveBayes
```

## Result

This result is evaluated for example test case - "review your password". 

<p align="center">
  <img src="https://github.com/praeclarumjj3/CuML/blob/master/CuNaiveBayes/images/result.png" width="500" />
</p>



