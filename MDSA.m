function [obj,Pc,P,alpha] =MDSA(X_src,X_src_label,X_tar_train,options,true_label )
T = options.T;
src_n = options.src_n;
k = options.k;
beta = options.beta;
gamma=options.gamma ;
g1 = options.g1;
Options_2 = [];
Options_2.T = 50;
Options_2.beta = beta;
Options_2.gamma = gamma;
Options_2.g1 = g1;
Options_2.src_n = src_n;
Options_2.k = k;
for i = 1:T
    [obj,P,Pc,alpha] = myMDSA(X_src,X_src_label,X_tar_train,Options_2);
    Zs = [];
    Zs_label = [];
    for i = 1:src_n
       Zs = [Zs,P{i}*X_src{i}'];
       Zs_label = [Zs_label;X_src_label{i}];
    end
    Zs = Zs*diag(sparse(1./sqrt(sum(Zs.^2))));
    Zt = Pc*X_tar_train';
    Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
    Zt_label =true_label;
    model= svmtrain(Zs_label,Zs','-s 0 -t 0 -c 1 -g 0.25 ');
    [pred_label, acc,~] = svmpredict(Zt_label,Zt',model);
    disp("Train acc:");
    disp(acc(1));
end

end

function [obj,P,Pc,alpha] =myMDSA(X_src,X_src_label,X_tar_train,options)
T = options.T;
src_n = options.src_n;
k = options.k;
beta = options.beta;
gamma=options.gamma ;
g1 = options.g1;
Xs = [];
Ys = [];
Xt = X_tar_train;
% initialize common projection Pc, multi-projection P{i} include src_n
% source projections and one target projection
d = size(X_src{1},2);
P = cell(1,src_n);
Options = [];
Options.ReducedDim =k;
for i = 1:src_n
    P{i}=PCA1(X_src{i},Options);
    P{i} = P{i}';
end
Z = cell(1,src_n);
for i = 1:src_n
    Z{i}=ones(size(X_src{i},1),size(X_tar_train,1));
end
%Multi scatter matrices
regu = 10^-5;
Sw_s = cell(1,src_n);
Sb_s = cell(1,src_n);
L =   cell(1,src_n);
for i = 1:src_n
    [Sw_s{i}, Sb_s{i}] = ScatterMat(X_src{i}',X_src_label{i});
    L{i} = Sw_s{i}-regu*Sb_s{i};
    Xs = [Xs;X_src{i}];
    Ys = [Ys;X_src_label{i}];
end
% Reconstruction matrix initialize
n = 0;
for i = 1:src_n
    n = n + size(X_src{i},1);
end
n = n+size(X_tar_train,1); 
X = [Xs;Xt]; % scale: n*d
Pc = PCA1(X,Options);
Pc = Pc';
alpha = ones(src_n,1)/src_n; % initialize all alpha(i) as src_num
alpha_obj = ones(src_n,1)/src_n;
for i = 1:T
    for v = 1:src_n
%         P{v} = (alpha(v)*Pc*Xt'*Z{v}'*X_src{v}+beta*Pc)*pinv(alpha(v)*L{v}+alpha(v)*X_src{v}'*Z{v}*Z{v}'*X_src{v}+beta*eye(d));
%         Z{v} = pinv(X_src{v}*P{v}'*P{v}*X_src{v}')*(X_src{v}*P{v}'*Pc*Xt');
%         Pc = (alpha(v)*P{v}*X_src{v}'*Z{v}*Xt+beta*P{v})*pinv(alpha(v)*Xt'*Xt+beta*eye(d));
%         alpha_obj(v) = trace(P{v}*L{v}*P{v}')+norm(P{v}*X_src{v}'*Z{v}-Pc*Xt','fro');
        P{v} = (alpha(v)*g1*Pc*Xt'*Z{v}'*X_src{v}+beta*Pc)/(alpha(v)*L{v}+alpha(v)*g1*X_src{v}'*Z{v}*Z{v}'*X_src{v}+beta*eye(d));
        Z{v} = (X_src{v}*P{v}'*P{v}*X_src{v}')\(X_src{v}*P{v}'*Pc*Xt');
        Pc = (alpha(v)*g1*P{v}*X_src{v}'*Z{v}*Xt+beta*P{v})/(alpha(v)*g1*Xt'*Xt+beta*eye(d));
        alpha_obj(v) = trace(P{v}*L{v}*P{v}')+g1*norm(P{v}*X_src{v}'*Z{v}-Pc*Xt','fro');
    end
    v_obj = alpha_obj/(2*gamma);
    v_obj = normalization(v_obj',3);
    v_obj = v_obj';
    alpha = EProjSimplex_new(v_obj, 1);%¸üÐÂ¦Á
    % obj
    sum1 = 0;
    for v = 1:src_n
        sum1 = sum1+alpha(v)*trace(P{v}*L{v}*P{v}')+alpha(v)*g1*norm(P{v}*X_src{v}'*Z{v}-Pc*Xt','fro')+beta*norm(P{v}-Pc,'fro');
    end
    sum2 = gamma*sum(alpha.*alpha,1);
    obj(i) = sum1+sum2;
    msg = ['iter ' num2str(i) ' over............'];
    disp(msg);
    if (i>=2) && (abs(obj(i)-obj(i-1))<1)
%     if (i>=50)
        msg = ['MDSA iter ' num2str(i) ' convergence..............' ', the obj_value is ' num2str(obj(i))];
        disp(msg);
        break;
    end
end
end