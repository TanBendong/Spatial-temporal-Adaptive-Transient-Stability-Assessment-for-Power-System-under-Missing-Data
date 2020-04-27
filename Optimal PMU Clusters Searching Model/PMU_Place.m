classdef PMU_Place % ∂®“Â¿‡
  properties
      bus_mat; 
      branch_mat;
      load_mat;
      gen_mat;
      pmu_loc;      
  end
  methods 
      function obj=PMU_Place(pmu_loc)
          % default power system is 39-bus power system
          % bus matrix
          obj.bus_mat=[1,2,3,4,5,6,7,8,9,10,11,12,13,...
                      14,15,16,17,18,19,20,21,22,23,24,25,26,...
                      27,28,29,30,31,32,33,34,35,36,37,38,39];
           % branch matrix
          obj.branch_mat=[1,2; 1,39; 2,3; 2,25;...
                         2, 30; 3, 4; 3, 18; 4, 5;...
                         4, 14; 5, 6; 5, 8; 6, 7;...
                         6, 11; 6, 31; 7, 8; 8, 9;...
                         9, 39; 10, 11; 10, 13; 10, 32;...
                         12, 11; 12, 13; 13, 14; 14, 15;...
                         15, 16; 16, 17; 16, 19; 16, 21;...
                         16, 24; 17, 18; 17, 27; 19, 20;...
                         19, 33; 20, 34; 21, 22; 22, 23;...
                         22, 35; 23, 24; 23, 36; 25, 26;...
                         25, 37; 26, 27; 26, 28; 26, 29;...
                         28, 29; 29, 38];
          % load matrix
          obj.load_mat=[3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28,29,31,39];
          % generator matrix
          obj.gen_mat=[30,31,32,33,34,35,36,37,38,39];
          % PMU location matrix
          obj.pmu_loc=pmu_loc;
      end
      function [link_array, pmu_array, ZIB_array]=Construct_matrix(obj)
        
        N  = length(obj.bus_mat);
        link_array = zeros(N, N);
        pmu_array = zeros(N, 1);
        ZIB_array = ones(N, 1);
        
        branch=obj.branch_mat;
        [BN,BM]=size(obj.branch_mat);
        for i = 1:BN
            link_array(branch(i, 1), branch(i, 2)) = 1;
            link_array(branch(i, 2), branch(i, 1)) = 1;
        end
        
        pmu_array(obj.pmu_loc) = 1;  
        
        ZIB_array(obj.load_mat) = 0;
        ZIB_array(obj.gen_mat) = 0;
      end
      function [V, I]=Observation(obj, link_array, pmu_array, ZIB_array)
        
        N  = length(obj.bus_mat);
        V = zeros(N, 1);  % voltage matrix
        I = zeros(N, N);  % current matrix
        G = zeros(length(obj.gen_mat), 1);  % generator matrix
        L = zeros(length(obj.load_mat), 1);  % load matrix

        % middle variables
        A = zeros(N, 1);
        B = zeros(N, 1);
        % describe observability according to rules
        for i =1:N
            % if pmu_array[i] == 1:
            for j =1:N
                if j ~= i
                    V(i) = pmu_array(i) || (pmu_array(j) && link_array(i, j));
                    I(i, j) = (pmu_array(i) || pmu_array(j)) && link_array(i, j);
                    I(j, i) = (pmu_array(i) || pmu_array(j)) && link_array(i, j);
                end
            end
        end
        
        for K =1:20  % iterate until convergence
            % Rule 1: The voltage and branch current of the bus where the PMU is installed are observable, and its incident bus is also observable.
            for i = 1:N
                for j = 1:N
                    if j ~= i
                        if V(i) == 1 && V(j) == 1 && I(i, j) == 0 && link_array(i, j) == 1
                            I(i, j) = V(i) && V(j);
                            I(j, i) = V(i) && V(j);
                        end
                    end
                end
            end
       
            
            % The voltage at both ends of the branch is observable, and the current of the branch is observable.
            for i = 1:N
                for j = 1:N
                    if j ~= i && V(i) == 0
                        V(i) = V(i) || (I(i, j) && V(j));
                    end
                end
            end
            
            % If the branch current and one end voltage are known, then the voltage at the other end of the branch is observable.
            for i = 1:N
                if ZIB_array(i) == 1
                    for j = 1:N
                        if i ~= j
                            for k = 1:N
                                A(k) = link_array(i, k) && I(i, k);
                            end
                            if sum(A) == sum(link_array(i, :)) - 1 && I(i, j) == 0
                                    I(i, j) = ZIB_array(i) && link_array(i, j);
                                    I(j, i) = ZIB_array(i) && link_array(i, j);
                            end
                        end
                    end
                end
            end
            
            % If ZIB is not installed with PMU, and only one incident branch current is unknown, then the branch current is observable.
            for i = 1:N
                if ZIB_array(i) == 1 && V(i) == 0
                    for j = 1:N
                        if i ~= j
                            for k = 1:N
                                B(k) = link_array(i, k) && V(k);
                            end
                            if (sum(B) == sum(link_array(i, :))) && (V(i) == 0)
                                V(i) = ZIB_array(i);
                            end
                         end
                    end
                end
            end
        end
 
      end
      function [V_Bus,I_Line_Result,G,L]=Index(obj,V, I,link_array)
         N  = length(obj.bus_mat);
        G = [];  
        L = [];  

        V_Bus = find( V ~= 0 );
        [I_Linerow, I_Linecol]= find( I ~= 0 );

        I_Line = [];
        I_Line_Result = [];
        for i =1:length(I_Linerow)
            P=[I_Linerow(i), I_Linecol(i)]==obj.branch_mat;
            index=find(sum(P,2) == 2);
            I_Line_Result=[I_Line_Result;obj.branch_mat(index,:)];
        end
       
        
        for i =1:length(obj.gen_mat)
            if V(obj.gen_mat(i))~=0 && sum(I(obj.gen_mat(i),:))==sum(link_array(obj.gen_mat(i),:))
                G=[G,obj.gen_mat(i)];
            end
        end
        for i =1:length(obj.load_mat)
            if V(obj.load_mat(i))~=0 && sum(I(obj.load_mat(i),:))==sum(link_array(obj.load_mat(i),:))
                L=[L,obj.load_mat(i)];
            end
        end
      end  
  end 
end