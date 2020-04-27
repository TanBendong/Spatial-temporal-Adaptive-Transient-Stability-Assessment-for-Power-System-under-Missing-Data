function Risk=fitness(x,P_Gen,Q_Gen,P_load,Q_load,V_f,V_theta,P_line,Q_line)
pmu_loc=[3,8,10,16,20,23,25,29];
X=reshape(x,8,length(x)/8);
X=X';
% feature importance calculation
score=[];
obv=ones(length(x)/8,39);
for i=1:length(x)/8 
    PMU = PMU_Place(pmu_loc(X(i,:)==1));
    [link_array, pmu_array, ZIB_array] = PMU.Construct_matrix();
    [V, I] = PMU.Observation(link_array, pmu_array, ZIB_array);
    [V_Bus, I_Line_Result, G, L]=PMU.Index(V, I,link_array);
    score=[score;(sum(sum(P_Gen.*V'))+sum(sum(Q_Gen.*V'))+sum(sum(P_load.*V'))+sum(sum(Q_load.*V'))+sum(sum(V_f.*V'))+sum(sum(V_theta.*V'))+sum(sum(P_line.*I))+sum(sum(Q_line.*I)))];
    obv(i,:)=V';
end
% risk calculation, and its definition can be seen in our paper
combine=1:8;
risk=[];
for m=1:8
    C = nchoosek(combine,m);
    [row,col]=size(C);    
    remain=0;
    for a=1:row
        value=score;
        value(sum(X(:,C(a,:)),2)>0)=0;
        remain=remain+sum(value)*(0.02)^m;        
    end 
    risk=[risk;remain];
end
Risk=-sum(risk)/sum(score);