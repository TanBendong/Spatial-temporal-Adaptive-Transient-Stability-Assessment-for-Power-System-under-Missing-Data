%%
% Optimal PMU Clusters Searching Model with GA
tic
load weights.mat % feature importance
[P_Gen,Q_Gen,P_load,Q_load,V_f,V_theta,P_line,Q_line]=Matrix_Transform(weights);
rand('seed',0);
options = optimoptions('ga','ConstraintTolerance',1e-6,'PlotFcn', @gaplotbestf,'Generations',40,'PopulationSize',40);
for N=2:25
    IntCon=1:N*8;
    A=zeros(N,N*8);
    b=-ones(N,1);
    for i =1:N
        for j=1:N
            if i<=j && i>j-1
                A(i,(j-1)*8+1:(j*8))=-1;       
            end      
        end
    end
    [x,fval]=ga(@(x)fitness(x,P_Gen,Q_Gen,P_load,Q_load,V_f,V_theta,P_line,Q_line),N*8,A,b,[],[],zeros(N*8,1),ones(N*8,1),@circlecon,IntCon,options);   
    X=reshape(x,8,length(x)/8);
    X=X';
    file=[]; % you should fill the address where you save your file
    csvwrite(file, X)
end
toc
