#CuDecisonTree
It contains code for Decision Tree algorithm implmented using CUDA.

##Setup
Run the code using:
```
-! nvcc -arch=sm_50 project.cu timer.c DecisionTreeCuda.cpp -o project
```
```
-! ./project 200000 784
```
