# GraphPPIS  
GraphPPIS is a novel framework for structure-based protein-protein interaction site prediction using deep graph convolutional network, which is able to capture information from high-order spatially neighboring amino acids. Here, the GraphPPIS source code is designed for high-throughput predictions, and does not have the limitation of one query protein per run. We recommend you to use the [web server](https://biomed.nscc-gz.cn:9094/apps/GraphPPIS) of GraphPPIS if your input is small.

# System requirement  
GraphPPIS is developed under Linux environment with:  
python  3.7.7  
numpy  1.19.1  
pandas  1.1.0  
torch  1.6.0  

# Software and database dependencies  
For the full & accurate version of GraphPPIS, you need to install the following three softwares and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[DSSP](https://github.com/cmbi/dssp)  
However, if you use the fast version of GraphPPIS, only DSSP is needed.

# Run GraphPPIS  
Running GraphPPIS for prediction:  
```
python GraphPPIS_predict.py -p PDBID.pdb
```

# Dataset, feature and model  
The datasets used in this study (Train_335, Test_60, Test_331 and UBtest_31) are stored in ./Dataset in python dictionary format:  
```
Dataset[ID] = [seq, label]
```
The distance maps(L * L) and normalized feature matrixes PSSM(L * 20), HMM(L * 20) and DSSP(L * 14) are stored in ./Feature in numpy format.  
The pre-trained GraphPPIS full model and the simplified version using BLOSUM62 can be found under ./Model  

# Web server and contact  
The GraphPPIS web server is freely available at [https://biomed.nscc-gz.cn:9094/apps/GraphPPIS](https://biomed.nscc-gz.cn:9094/apps/GraphPPIS)  
Contact:  
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)

