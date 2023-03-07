% This make.m is used under Windows

% add -largeArrayDims on 64-bit machines

mex -O -c -largeArrayDims svm.cpp 
mex -O -c -largeArrayDims svm_model_matlab.c
mex -O -largeArrayDims svmtrain.c svm.obj svm_model_matlab.obj
mex -O -largeArrayDims svmpredict.c svm.obj svm_model_matlab.obj
mex -O -largeArrayDims libsvmread.c
mex -O -largeArrayDims libsvmwrite.c
