# The code of the paper "Multi-source discriminant subspace alignment for cross-corpus speech emotion recognition".

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
