# The code of the paper "Multi-source discriminant subspace alignment for cross-corpus speech emotion recognition".

#The flowchart of MDSA.

![image](https://github.com/shaokai1209/shaokai1209/blob/main/2023%20TASLP.png)

#If you want to test MDSA with datasets, you should create two files for each dataset in the "data" folder, including dataset.mat and dataset_label.mat.

#If you have any questions, please email the author (shaokai1209@gmail.com)

#Running multi_src_SER.m

#num_src_domain: choose the number of source domain

#split: choose train:test in target corpus

#Special note: IEMOCAP and CVE require a license to obtain, the "data" folder has three public datasets for testing, only need to set num_src_domain=2.

# Example:  

#num_src_domain = 2

#split = 0.7

#========= Input Multi-source ==========

#The Filename of Source Corpus 1 : EMOVO

#The Filename of Source Corpus 2 : TESS

#========= Input Target ==========

#The Filename of Target Corpus : berlin

#...

#  Cite us
#Cite this paper, if you find MDSA is helpful to your research publication.
```
@article{li2023multi,
  title={Multi-source discriminant subspace alignment for cross-domain speech emotion recognition},
  author={Li, Shaokai and Song, Peng and Zheng, Wenming},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2448--2460},
  year={2023},
  publisher={IEEE}
}
```
