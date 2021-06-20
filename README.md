# GraphPPIS  
GraphPPIS is a novel framework for structure-based protein-protein interaction site prediction using deep graph convolutional network via initial residual and identity mapping, which is able to capture information from high-order spatially neighboring amino acids.

# System requirement  
GraphPPIS is developed under Linux environment with:  
python  3.7.7  
numpy  1.19.1  
pandas  1.1.0  
torch  1.6.0  
scikit-learn  0.23.2  

# Dataset and Feature  
The datasets used in this study (Train_335, Test_60, Test_331 and UBtest_31) are stored in ./Dataset in python dictionary format:  
```
Dataset[ID] = [seq, label]
```
The distance maps(L * L) and normalized feature matrixes PSSM(L * 20), HMM(L * 20) and DSSP(L * 14) are stored in ./Feature in numpy format.  

# Running GraphPPIS  
Train the model with default parameters:  
```
python train.py
```  
Test the model you just trained on the three test sets:  
```
python test.py
```
You can adjust the parameters via GraphPPIS_model.py  
The pre-trained GraphPPIS model and the simplified version using BLOSUM62 can be found under ./Model  

# Web server and contact  
The GraphPPIS web server is freely available at [https://biomed.nscc-gz.cn:9094/apps/GraphPPIS](https://biomed.nscc-gz.cn:9094/apps/GraphPPIS)  
Contact:  
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)

