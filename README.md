# The code of the paper "Multi-source discriminant subspace alignment for cross-corpus speech emotion recognition".

#The flowchart of MDSA. 
(https://github.com/shaokai1209/shaokai1209/blob/main/2023%20TASLP.png)

#If you want to test MDSA with datasets, you should create two files for each dataset in the "data" folder, including dataset.mat and dataset_label.mat.

#If you have any questions, please email the author (shaokai1209@gmail.com)

#Running multi_src_SER.m

#num_src_domain: choose the number of source domain

#split: choose train:test in target corpus

# Example:  

#num_src_domain = 4

#split = 0.7

#========= Input Multi-source ==========

#The Filename of Source Corpus 1 : IEMOCAP

#The Filename of Source Corpus 2 : CVE

#The Filename of Source Corpus 3 : EMOVO

#The Filename of Source Corpus 4 : TESS

#========= Input Target ==========

#The Filename of Target Corpus : berlin

#...

#  Cite us
#Cite this paper, if you find MDSA is helpful to your research publication.
```
@ARTICLE{10158502,
  author={Li, Shaokai and Song, Peng and Zheng, Wenming},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Multi-Source Discriminant Subspace Alignment for Cross-Domain Speech Emotion Recognition}, 
  year={2023},
  volume={31},
  number={},
  pages={2448-2460},
  doi={10.1109/TASLP.2023.3288415}}
```
