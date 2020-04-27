function [P_Gen,Q_Gen,P_load,Q_load,V_f,V_theta,P_line,Q_line]=Matrix_Transform(Score)
% transform feature importance vector to matrix according to the topology of 39-bus power system
% Score£ºfeature importance vector
% P_Gen£º active power of generators
% Q_Gen£ºreactive power of generators
% P_load£ºactive power of loads
% Q_load£ºreactive power of loads
% V_f£ºvoltage magnitude
% V_theta£ºvoltage angle
% P_line£ºactiveer power of branches
% Q_line£ºactiveer of branches

% loads matrix
load_mat=[3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28,29,31,39];
% generators matrix
gen_mat=[30,31,32,33,34,35,36,37,38,39];
% branches matrix
branch_mat=[1,2; 1,39; 2,3; 2,25;...
                         3, 4; 3, 18; 4, 5;...
                         4, 14; 5, 6; 5, 8; 6, 7;...
                         6, 11; 7, 8; 8, 9;...
                         9, 39; 10, 11; 10, 13;...
                         13, 14; 14, 15;...
                         15, 16; 16, 17; 16, 19; 16, 21;...
                         16, 24; 17, 18; 17, 27;...
                         21, 22; 22, 23;...
                         23, 24; 25, 26;...
                         26, 27; 26, 28; 26, 29;...
                         28, 29];
% transformers matrix
tranformer_mat=[2, 30;
    6, 31;
    10, 32;
    11, 12;
    12, 13;
    19, 20;
    19, 33;
    20, 34;
    22, 35;
    23, 36;
    25, 37;
    29, 38];
% active power matrix of generators£¬1*39
P_Gen=zeros(1,39);
P_Gen(gen_mat)=Score(1:10);

% reactive power matrix of generators£¬1*39
Q_Gen=zeros(1,39);
Q_Gen(gen_mat)=Score(11:20);

% active power matrix of loads£¬1*39
P_load=zeros(1,39);
P_load(load_mat)=Score(191:209);

% reactive power matrix of loads£¬1*39
Q_load=zeros(1,39);
Q_load(load_mat)=Score(210:228);

% voltage magnitude matrix £¬1*39
V_f=zeros(1,39);
V_f=Score(21:59);

% voltage angle matrix£¬1*39
V_theta=zeros(1,39);
V_theta=Score(60:98);

% power matrix of branches£¬39*39
P_line=zeros(39,39);
P_line_Score=Score(99:132);
P_tran_Score=Score(167:178);
Q_line=zeros(39,39);
Q_line_Score=Score(133:166);
Q_tran_Score=Score(179:190);

[m,n]=size(branch_mat);
[m1,n1]=size(tranformer_mat);
L=zeros(39,39);
k=1;
for i=1:m
    P_line(branch_mat(i,1),branch_mat(i,2))=P_line_Score(k);
    P_line(branch_mat(i,2),branch_mat(i,1))=P_line_Score(k);
    Q_line(branch_mat(i,1),branch_mat(i,2))=Q_line_Score(k);
    Q_line(branch_mat(i,2),branch_mat(i,1))=Q_line_Score(k);
    k=k+1;
end
k=1;
for i=1:m1
    P_line(tranformer_mat(i,1),tranformer_mat(i,2))=P_tran_Score(k);
    P_line(tranformer_mat(i,2),tranformer_mat(i,1))=P_tran_Score(k);
    Q_line(tranformer_mat(i,1),tranformer_mat(i,2))=Q_tran_Score(k);
    Q_line(tranformer_mat(i,2),tranformer_mat(i,1))=Q_tran_Score(k);
    k=k+1;
end
end
