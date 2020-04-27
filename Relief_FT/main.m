% 计算特征重要性
tic
load X_train.mat
load y_train.mat
[N,T,n]=size(X_train);
X=zeros(N,n,T);
% reshape data
for i=1:N
    x=X_train(i,:,:);
    y=reshape(x,[n,T]);
    X(i,:,:)=y';
end
[ranks,weights] = relieff_tan(X,y_train,10);
bar(weights)
save weights.mat weights
toc