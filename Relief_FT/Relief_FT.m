function [ranked,weight] = Relief_FT(X,Y,K,varargin)
% Calculate feature importance for multi-dimension time series
% X is input features，and its shape is N*n*T，where N is the number of sample，n is feature dimension，T is the length of time 
% Y is label
if nargin<3
    error(message('stats:relieff:TooFewInputs'));
end

% 检查X数据格式是否正确
if ~isnumeric(X)
    error(message('stats:relieff:BadX'));
end

% Parse input arguments
validArgs = {'method' 'prior' 'updates'   'categoricalx' 'sigma'};
defaults  = {      ''      []     'all'            'off'      []};

% Get optional args
[method,prior,numUpdates,categoricalX,sigma] ...
    = internal.stats.parseArgs(validArgs,defaults,varargin{:});
[X,Y] = removeNaNs(X,Y);
    
% Group Y for classification. Get class counts and probabilities.
% Get groups and matrix of class counts
if isa(Y,'categorical')
    Y = removecats(Y);
end
[Y,grp] = grp2idx(Y);
[X,Y] = removeNaNs(X,Y);
Ngrp = numel(grp); 
N = size(X,1);
C = false(N,Ngrp);
C(sub2ind([N Ngrp],(1:N)',Y)) = true;

% Get class probs
if isempty(prior) || strcmpi(prior,'empirical')
    classProb = sum(C,1);
elseif strcmpi(prior,'uniform')
    classProb = ones(1,Ngrp);
elseif isstruct(prior)
    if ~isfield(prior,'group') || ~isfield(prior,'prob')
        error(message('stats:relieff:PriorWithMissingField'));
    end
    if iscell(prior.group)
        usrgrp = prior.group;
    else
        usrgrp = cellstr(prior.group);
    end
    [tf,pos] = ismember(grp,usrgrp);
    if any(~tf)
        error(message('stats:relieff:PriorWithClassNotFound', grp{ find( ~tf, 1 ) }));
    end
    classProb = prior.prob(pos);
elseif isnumeric(prior)
    if ~isfloat(prior) || length(prior)~=Ngrp || any(prior<0) || all(prior==0)
        error(message('stats:relieff:BadNumericPrior', Ngrp));
    end
    classProb = prior;
else
    error(message('stats:relieff:BadPrior'));
end
    
    % Normalize class probs
    classProb = classProb/sum(classProb);
    
    % If there are classes with zero probs, remove them
    zeroprob = classProb==0;
    if any(zeroprob)
        t = zeroprob(Y);
        if sum(t)==length(Y)
            error(message('stats:relieff:ZeroWeightPrior'));
        end
        Y(t) = [];
        X(t,:) = [];
        C(t,:) = [];
        C(:,zeroprob) = [];
        classProb(zeroprob) = [];
    end


% Do we have enough observations?
if length(Y)<2
    error(message('stats:relieff:NotEnoughObs'));
end

% Check the number of nearest neighbors
if ~isnumeric(K) || ~isscalar(K) || K<=0
    error(message('stats:relieff:BadK'));
end
K = ceil(K);

% 检查迭代数目
if (~ischar(numUpdates) || ~strcmpi(numUpdates,'all')) && ...
        (~isnumeric(numUpdates) || ~isscalar(numUpdates) || numUpdates<=0)
    error(message('stats:relieff:BadNumUpdates'));
end
if ischar(numUpdates)
    numUpdates = size(X,1);
else
    numUpdates = ceil(numUpdates);
end

% Check the type of X
if ~ischar(categoricalX) || ...
        (~strcmpi(categoricalX,'on') && ~strcmpi(categoricalX,'off'))
    error(message('stats:relieff:BadCategoricalX'));
end
categoricalX = strcmpi(categoricalX,'on');

% Check sigma
if ~isempty(sigma) && ...
        (~isnumeric(sigma) || ~isscalar(sigma) || sigma<=0)
    error(message('stats:relieff:BadSigma'));
end
if isempty(sigma)
        sigma = Inf;
end

% The # updates cannot be more than the # observations
numUpdates = min(numUpdates, size(X,1));

% Choose the distance function depending upon the categoricalX
if ~categoricalX
    distFcn = 'cityblock';
end

% Find max and min for every predictor
p = size(X,2); 
Xmax = max(X,[],[1 3]); 
Xmin = min(X,[],[1 3]);
Xdiff = Xmax-Xmin;

% Exclude single-valued attributes
isOneValue = Xdiff < eps(Xmax); 
if all(isOneValue)
    ranked = 1:p;
    weight = NaN(1,p);
    return;
end
X(:,isOneValue) = [];
Xdiff(isOneValue) = [];
rejected = find(isOneValue);
accepted = find(~isOneValue);

% 标准化
if ~categoricalX
    X = bsxfun(@rdivide,bsxfun(@minus,X,mean(X,[1,3])),Xdiff); 
end

% Get appropriate distance function in one dimension.
% thisx must be a row-vector for one observation.
% x can have more than one row.
if ~categoricalX
    dist1D = @(thisx,x) cityblock(thisx,x);
end

% Call ReliefF. By default all weights are set to NaN.
weight = NaN(1,p);
weight(accepted) = RelieffClass(X,C,classProb,numUpdates,K,distFcn,dist1D,sigma);

% Assign ranks to attributes
[~,sorted] = sort(weight(accepted),'descend');
ranked = accepted(sorted);
ranked(end+1:p) = rejected;



% -------------------------------------------------------------------------
function attrWeights = RelieffClass(scaledX,C,classProb,numUpdates,K,...
    distFcn,dist1D,sigma)
% ReliefF for classification

[numObs,numAttr,Time] = size(scaledX);
attrWeights = zeros(1,numAttr);
Nlev = size(C,2);

% Choose the random instances
rndIdx = randsample(numObs,numUpdates);
idxVec = (1:numObs)';

% Make searcher objects, one object per class. 
searchers = cell(Nlev,1);
for c=1:Nlev
    searchers{c} = createns(scaledX(C(:,c),:),'Distance',distFcn); % 这里要进行修改，构建kdtree
end

% Outer loop, for updating attribute weights iteratively
for i = 1:numUpdates
    thisObs = rndIdx(i);
    
    % Choose the correct random observation
    selectedX = scaledX(thisObs,:,:);

    % Find the class for this observation
    thisC = C(thisObs,:);
    
    % Find the k-nearest hits 
    sameClassIdx = idxVec(C(:,thisC));
    sameClassX=scaledX(sameClassIdx,:,:);
    
    % we may not always find numNeighbor Hits
    lenHits = min(length(sameClassIdx)-1,K);

    % find nearest hits
    % It is not guaranteed that the first hit is the same as thisObs. Since
    % they have the same class, it does not matter. If we add observation
    % weights in the future, we will need here something similar to what we
    % do in ReliefReg.
    
    Hits = [];
    if lenHits>0
        SameDistance=[];
        for i =1:length(sameClassIdx)
            distance=sqrt(sum(sum((selectedX-sameClassX(i,:,:)).^2)));
            SameDistance=[SameDistance,distance];
        end
        [B,IS]=sort(SameDistance);
        idxH = IS(1:K); % 这里要进行修改
        idxH(1) = [];
        Hits = sameClassIdx(idxH);
    end    
    
    % Process misses
    missClass = find(~thisC);
    Misses = [];
    
    if ~isempty(missClass) % Make sure there are misses!
        % Find the k-nearest misses Misses(C,:) for each class C ~= class(selectedX)
        % Misses will be of size (no. of classes -1)x(K)
        Misses = zeros(Nlev-1,min(numObs,K+1)); % last column has class index
        
        for mi = 1:length(missClass)
            
            % find all observations of this miss class
            missClassIdx = idxVec(C(:,missClass(mi)));
            missClassX=scaledX(missClassIdx,:,:);
            % we may not always find K misses
            lenMiss = min(length(missClassIdx),K);
            % find nearest misses
            MissDistance=[];
            for i =1:length(missClassIdx)
                distance=sqrt(sum(sum((selectedX-missClassX(i,:,:)).^2)));
                MissDistance=[MissDistance,distance];
            end
            [B,IM]=sort(MissDistance);           
            idxM =IM(1:K); % 这里要进行修改
            Misses(mi,1:lenMiss) = missClassIdx(idxM);
            
        end
        
        % Misses contains obs indices for miss classes, sorted by dist.
        Misses(:,end) = missClass;
    end
            
    %***************** ATTRIBUTE UPDATE *****************************
    % Inner loop to update weights for each attribute:
    
    for j = 1:numAttr
        dH = diffH(j,scaledX,thisObs,Hits,dist1D,sigma)/numUpdates;
        dM = diffM(j,scaledX,thisObs,Misses,dist1D,sigma,classProb)/numUpdates;
        attrWeights(j) = attrWeights(j) - dH + dM;
    end
    %****************************************************************
end





%Helper functions RelieffClass

%--------------------------------------------------------------------------
% DIFFH (for RelieffClass): Function to calculate difference measure
% for an attribute between the selected instance and its hits

function distMeas = diffH(a,X,thisObs,Hits,dist1D,sigma)

% If no hits, return zero by default
if isempty(Hits)
    distMeas = 0;
    return;
end

% Get distance weights
distWts = exp(-((1:length(Hits))/sigma).^2)';
distWts = distWts/sum(distWts);

% Calculate weighted sum of distances
distMeas = sum(dist1D(X(thisObs,a,:),X(Hits,a,:)).*distWts);


%--------------------------------------------------------------------------
% DIFFM (for RelieffClass) : Function to calculate difference measure
% for an attribute between the selected instance and its misses
function distMeas = diffM(a,X,thisObs,Misses,dist1D,sigma,classProb)

distMeas = 0;

% If no misses, return zero
if isempty(Misses)
    return;
end

% Loop over misses
for mi = 1:size(Misses,1)
    
    ismiss = Misses(mi,1:end-1)~=0;
    
    if any(ismiss)
        cls = Misses(mi,end);
        nmiss = sum(ismiss);
        
        distWts = exp(-((1:nmiss)/sigma).^2)';
        distWts = distWts/sum(distWts);
        
        distMeas = distMeas + ...
            sum(dist1D(X(thisObs,a,:),X(Misses(mi,ismiss),a,:)).*distWts(1:nmiss)) ...
            *classProb(cls);
    end
end

% Normalize class probabilities.
% This is equivalent to P(C)/(1-P(class(R))) in ReliefF paper.
totProb = sum(classProb(Misses(:,end)));
distMeas = distMeas/totProb;


function [X,Y] = removeNaNs(X,Y)
% Remove observations with missing data
NaNidx = bsxfun(@or,isnan(Y),any(isnan(X),2));
X(NaNidx,:) = [];
Y(NaNidx,:) = [];


function d = cityblock(thisX,X)
d = sqrt(sum((thisX-X).^2,[2,3]));
