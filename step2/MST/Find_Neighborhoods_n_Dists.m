% Function to compute MST-based neighborhoods of certain radius, and 
% find the normalized distances ratio to the neighbors.
% Inputs:   - datum     :   a HxWxC tensor
%           - Radius    :   the radius of the neighborhood
% Outputs:  - NeighMask :   A sparse matrix with the neighborhood mask
%                           of each node given in binary row vectors
%           - NeighRatio:   The ratio of sum of distances to the neighbors,
%                           normalized by the overall sum of each node's
%                           distances

function [NeighMask,NeighRatio]=Find_Neighborhoods_n_Dists(datum,Radius)
    datum=reshape(datum,[],size(datum,3)); % reshape datum to HWxC (row vectors)
    d=squareform(pdist(datum)); % Compute pairwise Euclidean Distances
    NNodes=length(d); % Nnumber of nodes
    [y1,y2]=MexKruskMST3(double(d)); % compute MST
    linearInd = sub2ind([NNodes NNodes], y1(1,:), y1(2,:)); %Convert indices  
    MST=zeros(NNodes,NNodes); % construct neighboring graph of MST
    MST(linearInd)=1;

    MST=MST+MST'-diag(diag(MST)); % remove self-connections
    GD = dijkstra( sparse(MST) , 1:length(MST) ); % Compute pairwise geodesic distances over the MST
    GD=int32(GD);
    RNeighs=GD<=Radius&(GD>0); % Define neigbors as those up to Radius distance
    %NeighMask=sparse(RNeighs); % Construct neighborhood mask | % sparse logical
    NeighMask=RNeighs; % Construct neighborhood mask | % logical


%             NeighRatio=sum(RNeighs.*d,2)./sum(d,2);
    NeighRatio=sum(RNeighs.*(d.^2),2)./sum((d.^2),2); % Compute ratios of distances
