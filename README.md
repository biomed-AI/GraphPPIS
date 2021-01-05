# GraphPPIS  
GraphPPIS is a novel framework for protein-protein interacting site prediction using deep graph convolutional network via initial residual and identity mapping, which is able to capture information from high-order neighboring amino acids.

# System requirement  
GraphPPIS is developed under Linux environment with:  
python 3.7.7  
numpy 1.19.1  
pandas 1.1.0  
torch 1.6.0  
scikit-learn 0.23.2  

# Running GraphPPIS  
Train the model with default parameters:  
`python train.py`  
Test the model you just trained:  
`python test.py`  
You can adjust the parameters via GraphPPIS_model.py  
The pre-trained GraphPPIS model and the simplified version using BLOSUM62 can be found under ./Model  

# Web server and contact  
The GraphPPIS web server is freely available at [http://biomed.nscc-gz.cn](http://biomed.nscc-gz.cn)  
Contact:  
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)

