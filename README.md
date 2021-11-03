# Intro  
GraphPPIS is a novel framework for structure-based protein-protein interaction site prediction using deep graph convolutional network, which is able to capture information from high-order spatially neighboring amino acids. The GraphPPIS source code is designed for high-throughput predictions, and does not have the limitation of one query protein per run. We recommend you to use the [web server](https://biomed.nscc-gz.cn/apps/GraphPPIS) of GraphPPIS if your input is small.  
![GraphPPIS_framework](https://github.com/biomed-AI/GraphPPIS/blob/master/IMG/GraphPPIS_framework.png)  

# System requirement  
GraphPPIS is developed under Linux environment with:  
python  3.7.7  
numpy  1.19.1  
pandas  1.1.0  
torch  1.6.0  

# Software and database requirement  
To run the full & accurate version of GraphPPIS, you need to install the following three software and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[DSSP](https://github.com/cmbi/dssp)  
However, if you use the fast version of GraphPPIS, only DSSP is needed.  

# Build database and set path  
1. Use `makeblastdb` in BLAST+ to build UniRef90 ([guide](https://www.ncbi.nlm.nih.gov/books/NBK569841/)).  
2. Build Uniclust30 following [this guide](https://github.com/soedinglab/uniclust-pipeline).  
3. Set path variables `UR90`, `HHDB`, `PSIBLAST`, `HHBLITS` and `DSSP` in `GraphPPIS_predict.py`.  

# Run GraphPPIS for prediction  
For a protein chain in PDB:  
```
python GraphPPIS_predict.py -p PDB_ID -c chain_ID
```
For a user-custom PDB file:  
```
python GraphPPIS_predict.py -f XXX.pdb -c chain_ID
```
The program uses the fast model in default. If you want to use the slow & accurate mode, type as follows:  
```
python GraphPPIS_predict.py -p PDB_ID -c chain_ID -m slow
```

# Dataset, feature and model  
We provide the datasets, pre-computed features and the two pre-trained models here for those interested in reproducing our paper.  
The datasets used in this study (Train_335, Test_60, Test_315 and UBtest_31) are stored in ./Dataset in fasta format.  
The distance maps(L * L) and normalized feature matrixes PSSM(L * 20), HMM(L * 20) and DSSP(L * 14) are stored in ./Feature in numpy format.  
The pre-trained GraphPPIS full model and the simplified version using BLOSUM62 can be found under ./Model  

# Web server and contact  
The GraphPPIS web server is freely available at [https://biomed.nscc-gz.cn/apps/GraphPPIS](https://biomed.nscc-gz.cn/apps/GraphPPIS)  
Contact:  
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)

