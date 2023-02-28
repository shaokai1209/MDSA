clear all
addpath('./libsvm-new');
addpath('./liblinear-2.1/matlab');
addpath('./data');
warning off;
%% data set path
db_path = './data';
num_src_domain = 4;
%% train:test in target corpus
split = 0.7;
trainset_path = cell(1,num_src_domain);
trainset_label_path = cell(1,num_src_domain);
disp("========= Input Multi-source ==========");
for i=1:num_src_domain
    msg = ['The Filename of Source Corpus ',num2str(i),' :'];
    m = input(msg,'s');
    trainset_path{i} = [db_path '/' num2str(m) '.mat'];
    trainset_label_path{i} = [db_path '/' num2str(m) '_label.mat'];
end
disp("========= Input Target ==========");
msg = ['The Filename of Target Corpus :'];
m = input(msg,'s');
testset_path = [db_path '/' num2str(m) '.mat'];
testset__label_path = [db_path '/' num2str(m) '_label.mat'];

disp("========= Feature Dimension Reduction: PCA ==========");
X_src = cell(1,num_src_domain);
X_src_label = cell(1,num_src_domain);
disp("========= Normalization ==========");
for i=1:num_src_domain
    disp(trainset_path{i});
    load(trainset_path{i});
    load(trainset_label_path{i});
    S=double(feature);
    S=normalization(S',1);
    X_src{i} = S';
    X_src_label{i} = [double(label)]; 
    %shuffle source_i
    X_ss= [X_src{i},X_src_label{i}];
    rowrank = randperm(size(X_ss, 1));
    X1 = X_ss(rowrank,:);  
    X_src{i} = X1(:,1:size(X_src{i},2));
    X_src_label{i} = X1(:,size(X_src{i},2)+1);
end
load(testset_path);
load(testset__label_path);
T = double(feature);
T=normalization(T',1);
X_tar = T';
X_tar_label = double(label);
disp("......... Normalization End ........");
disp("========= Multi-PCA ==========");
X=[];
trials = [];
for i=1:num_src_domain
    X = [X;X_src{i}];
end
X = [X;X_tar];
Options = [];
Options.ReducedDim =500;
msg = ['PCA feature from '  num2str(size(X,2)) ' to ' num2str(Options.ReducedDim)];
disp(msg);
[eigvector,eigvalue] = PCA1(X,Options);
X = X * eigvector;
n = 0;
n_f = 0;
for i=1:num_src_domain
    n_f = n+1;
    n = n+size(X_src{i},1);
    X_src{i} = X(n_f:n,:);
end
X_tar = X(n+1:end,:);
disp("......... Multi-PCA End ........");
disp("========= Split Target Corpus by Emotional Labels (Train : Test) ==========");
% for iii = 1:10
msg = ['According to split='  num2str(split) ' , the target_train:target_test is ' num2str(10*split) ':' num2str(10*(1-split))];
disp(msg);
class_num = max(X_tar_label);
split_list = cell(1,class_num);
c_e = 0;
c_s = 0;
X_tar_train = [];
X_tar_test = [];
X_tar_train_label = [];
X_tar_test_label = [];
for i = 1:class_num
    c_s = c_e+1;
    c_n = length(find(X_tar_label==i));
    c_e = c_e + c_n;
    c_p = round(split*c_n)+c_s-1;
    split_list{i} = [c_s,c_p,c_e];
    X_tar_train = [X_tar_train;X_tar(c_s:c_p,:)];
    X_tar_train_label = [X_tar_train_label;X_tar_label(c_s:c_p,:)];
    X_tar_test = [X_tar_test;X_tar(c_p+1:c_e,:)];
    X_tar_test_label = [X_tar_test_label;X_tar_label(c_p+1:c_e,:)];
end
data = [];
for jj=1:5
%shuffle target_train
X_tt= [X_tar_train,X_tar_train_label];
rowrank = randperm(size(X_tt, 1)); 
X1 = X_tt(rowrank,:);  
X_tar_train = X1(:,1:size(X_tar_train,2));
X_tar_train_label = X1(:,size(X_tar_train,2)+1);
%shuffle target_test
X_tt= [X_tar_test,X_tar_test_label];
rowrank = randperm(size(X_tt, 1)); 
X1 = X_tt(rowrank,:);  
X_tar_test = X1(:,1:size(X_tar_test,2));
X_tar_test_label = X1(:,size(X_tar_test,2)+1);
disp("......... Split End ........");
disp("========= Multi-Disciminant Subspace Alihnment (MDSA) START ==========");
ll = 0;
for g11= [1]
    ll=ll+1;
    options = [];
    options.beta = 8*10^3;%1~8
    options.gamma =1.5;%
    options.g1 =0.1;
    options.T = 1;
    options.src_n = num_src_domain;
    options.k = 250;% it can be selected in[100,150,200,250]  
    t1 = datestr(now,'HH:MM:SS.FFF');
    [obj,Pc,P,alpha] =MDSA(X_src,X_src_label,X_tar_train,options,X_tar_train_label);
    t2 = datestr(now,'HH:MM:SS.FFF');
    Zs = [];
    Zs_label = [];
    for i = 1:num_src_domain
       Zs = [Zs,P{i}*X_src{i}'];
       Zs_label = [Zs_label;X_src_label{i}];
    end
    Zs = Zs*diag(sparse(1./sqrt(sum(Zs.^2))));
    Zt = Pc*X_tar_test';
    Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
    Zt_label =X_tar_test_label;
    model= svmtrain(Zs_label,Zs','-s 0 -t 0 -c 1 -g 0.25 ');
    [pred_label, acc,~] = svmpredict(Zt_label,Zt',model);
    disp("Final test acc:");
    disp(acc(1));
    myacc(ll) = acc(1);
end
data(jj) = acc(1);
end
b = mean(data);
disp(b);
a = std(data,1);
disp(a);
% end
% Zss = [];
% lll =  cell(1,num_src_domain);
% for i = 1:num_src_domain
%        mm = P{i}*X_src{i}';
%        mm=normalization(mm,1);
%        Zss = [Zss,mm(:,1:300)];
%        lll{i} = X_src_label{i}(1:300,:);
%        trials  = [trials;300];
% end
% Zss = Zss*diag(sparse(1./sqrt(sum(Zss.^2))));
% 
% X =[Zss,Zt];
% % X=normalization(X,1);
% %mahalanobis  euclidean
%    Y = tsne(X','Algorithm','exact','Distance','cosine');%,'NumPCAComponents',10
%    Ys1=Y(1:trials(1),:);
%    Ys2=Y(trials(1)+1:trials(1)+trials(2),:);
%    Ys3=Y(trials(1)+trials(2)+1:trials(1)+trials(2)+trials(3),:);
%    Ys4=Y(trials(1)+trials(2)+trials(3)+1:trials(1)+trials(2)+trials(3)+trials(4),:);
%    Y2=Y(trials(1)+trials(2)+trials(3)+1:end,:);
%    figure;
%       %subplot(2,3,1);
% %        axis([-50,50,-50,50]);
%        
%      scatter(Ys1(lll{1}==1,1),Ys1(lll{1}==1,2),'*','r','LineWidth',1);
%      hold on
%      scatter(Ys1(lll{1}==2,1),Ys1(lll{1}==2,2),'*','b','LineWidth',1);
%         hold on;
%      scatter(Ys1(lll{1}==3,1),Ys1(lll{1}==3,2),'*','g','LineWidth',1);
%         hold on;
%      scatter(Ys1(lll{1}==4,1),Ys1(lll{1}==4,2),'*','y','LineWidth',1);
%         hold on;
% 
%    scatter(Ys2(lll{2}==1,1),Ys2(lll{2}==1,2),'+','r','LineWidth',1);
%      hold on
%      scatter(Ys2(lll{2}==2,1),Ys2(lll{2}==2,2),'+','b','LineWidth',1);
%         hold on;
%      scatter(Ys2(lll{2}==3,1),Ys2(lll{2}==3,2),'+','g','LineWidth',1);
%         hold on;
%      scatter(Ys2(lll{2}==4,1),Ys2(lll{2}==4,2),'+','y','LineWidth',1);
%         hold on;
%         
%    scatter(Ys3(lll{3}==1,1),Ys3(lll{3}==1,2),'o','r','LineWidth',1);
%      hold on
%      scatter(Ys3(lll{3}==2,1),Ys3(lll{3}==2,2),'o','b','LineWidth',1);
%         hold on;
%      scatter(Ys3(lll{3}==3,1),Ys3(lll{3}==3,2),'o','g','LineWidth',1);
%         hold on;
%      scatter(Ys3(lll{3}==4,1),Ys3(lll{3}==4,2),'o','y','LineWidth',1);
%         hold on;
%         
%         
%     scatter(Ys4(lll{4}==1,1),Ys4(lll{4}==1,2),'^','r','LineWidth',1);
%      hold on
%      scatter(Ys4(lll{4}==2,1),Ys4(lll{4}==2,2),'^','b','LineWidth',1);
%         hold on;
%      scatter(Ys4(lll{4}==3,1),Ys4(lll{4}==3,2),'^','g','LineWidth',1);
%         hold on;
%      scatter(Ys4(lll{4}==4,1),Ys4(lll{4}==4,2),'^','y','LineWidth',1);
%         hold on;
%         
%         
%      scatter(Y2(X_tar_test_label==1,1),Y2(X_tar_test_label==1,2),'d','r','LineWidth',1);
%      hold on
%      scatter(Y2(X_tar_test_label==2,1),Y2(X_tar_test_label==2,2),'d','b','LineWidth',1);
%         hold on;
%      scatter(Y2(X_tar_test_label==3,1),Y2(X_tar_test_label==3,2),'d','g','LineWidth',1);
%         hold on;
%      scatter(Y2(X_tar_test_label==4,1),Y2(X_tar_test_label==4,2),'d','y','LineWidth',1);
%         hold on;
     
%         box on;
%     view(-20,20);



%     act=Zt_label;
%     act1=act';
%     det=pred_label;
%     det1=det';
%     confusion_matrix1(act1,det1);