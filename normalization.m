function normal = normalization(x,kind)
% by Li Yang BNU MATH Email:farutoliyang@gmail.com QQ:516667408
% last modified 2009.2.24
%

if nargin < 2
    kind = 2;%kind = 1 or 2 表示第一类或第二类规范化
end

[m,n]  = size(x);
normal = zeros(m,n);
%% normalize the data x to [0,1]
if kind == 1  
    for i = 1:m
        if x(i,:) == zeros(1,n)
            normal(i,:) = x(i,:);
        else
            ma = max( x(i,:) );
            mi = min( x(i,:) );
            normal(i,:) = ( x(i,:)-mi )./( ma-mi );
        end
    end
end
%% normalize the data x to [-1,1]
if kind == 2
    for i = 1:m
        if x(i,:) == zeros(1,n)
            normal(i,:) = x(i,:);
        else       
            mea = mean( x(i,:) );
            va = var( x(i,:) );
            normal(i,:) = ( x(i,:)-mea )/va;
        end
    end
end

%% normalize the data x to [-1,1]
if kind == 3
    for i = 1:m
        if x(i,:) == zeros(1,n)
            normal(i,:) = x(i,:);
        else    
            normal(i,:) = x(i,:)/norm(x(i,:));
        end
    end
end

if kind == 4
    for i = 1:m
        if x(i,:) == zeros(1,n)
            normal(i,:) = x(i,:);
        else    
            normal(i,:) = x(i,:)/max(x(i,:));
        end
    end
end
