# CuSVM

This folder contains code for SVM Algorithm implemented using **CUDA** as a semester project for **CSN-221: Computer Architecture**

## Setup

Steps to **Run** it on **Windows**:

- Download & Install Nvidia CUDA Toolkit v11.1.1 by following the tutorial given below:
	
```
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#introduction
```

- Run the following command to create the .exe file:

```
nvcc kernel.cu -lcublas
```

- Double click on the a.exe file generated to run it.

Steps to **Run & Debug** it on **Windows**:

- Download & Install Microsoft Visual Studio 2019

- Before opening this project in Visual Studio, it is advisable to make sure Cuda is working properly by making a new Cuda project in Visual Studio.

- Create a new project in Visual studio and check in the available templates whether "CUDA xy.z Runtime" (xy.z refers to version) template is there or not. If not then the link given below might help in fixing it. If yes, then Cuda is properly configured with Visual Studio.

```
https://forums.developer.nvidia.com/t/no-project-templates-in-vs2017/69306
```

- Open the .sln/.vcxproj file in Visual Studio.

- Click on Build -> Build Solution and if the build is successful you can run it using Ctrl + F5.