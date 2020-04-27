function [c,ceq] = circlecon(x)
pmu_loc=[3,8,10,16,20,23,25,29];
X=reshape(x,8,length(x)/8);
X=X';
observation=zeros(length(x)/8,39);
for i=1:length(x)/8 
    PMU = PMU_Place(pmu_loc(X(i,:)==1));
    [link_array, pmu_array, ZIB_array] = PMU.Construct_matrix();
    [V, I] = PMU.Observation(link_array, pmu_array, ZIB_array);
    observation(i,:)=V';
end
yueshu=sum(observation,1);
c=-(yueshu'-ones(39,1));
combine=1:length(x)/8;
C = nchoosek(combine,2);
[row,col]=size(C);  
for i=1:row
    c=[c;-(sum(abs(X(C(i,1),:)-X(C(i,2),:)))-1)];
end
ceq=[];
