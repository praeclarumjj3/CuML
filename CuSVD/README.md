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
%cd /content/gdrive/My Drive/CuML/CuSVD
```

- Compile the code using:

```
! nvcc -Xcompiler="--std=c++0x" -lm -arch=sm_50 -std=c++14 main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca
```

- Run the executable using:
```
! ./pca <input filename> <retention>
```
- Example 
```
! ./pca testcases/testcase_4_4 50
```

