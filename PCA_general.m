load("PCA_test.mat");
PCA_data = PCA_data ./ repmat(sum(PCA_data,2),1,size(PCA_data,2));
PD = zscore(PCA_data,1) ;clear PCA_data ;
save PCA_test_zsocre.mat PD;