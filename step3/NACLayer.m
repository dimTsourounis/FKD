classdef NACLayer < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 
   % calculates the Euclidean distance matrix between input vectors and
   % returns the Neighborhood Affinity Contrast (NAC)
   

    methods
        function layer = NACLayer(numInputs,name)
            % layer = NACLayer(name) creates a NAC Layer
            % for 3-D input volume HxWxC and specifies the layer
            % name.
			
            % Set number of inputs.
            layer.NumInputs = numInputs;
            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Neighborhood Affinity Contrast';

        end
        
        function NAC = predict(layer, varargin)
            % calculates Euclidean distance matrix and then the NAC
            
            dataInput = varargin;
            FeatureMaps = dataInput{1}; % H x W x D x B
            NeighMask = dataInput{2}; % N x N x 1 x B, N = H*W
            
            H = size(FeatureMaps,1);
            W = size(FeatureMaps,2);
            C = size(FeatureMaps,3);
            B = size(FeatureMaps,4);
            %
            L = size(NeighMask,1);
            %L = size(NeighMask,2);
            %B = size(NeighMask,4);
            
            % Euclidean distance matrix
            x=reshape(FeatureMaps,H*W,C,1,[]);
            G = zeros(L,L,1,B,'like',NeighMask);
            for j = 1:B
                G(:,:,1,j) = x(:,:,1,j)*x(:,:,1,j)';
            end
            %G = pagemtimes(x,pagetranspose(x)); % G = x*x';
            
            D = zeros(L,L,1,B,'like',NeighMask);
            for i = 1:B
                tmpG = G(:,:,1,i);
                %diagG= tmpG(sub2ind(size(tmpG),1:size(tmpG,1),1:size(tmpG,2)));
                diagG= tmpG(sub2ind([L L],1:L,1:L));
                dG=diagG'*ones(1,length(diagG)); % dG is a matrix with replicas of G's diagonal as columns
                D(:,:,1,i)=dG+dG'-2*tmpG; % final square Euclidean distance matrix
            end
            Dglobal=sum(D,2); % incorporate(i.e. sum) the distances with other vectors
            %%%%Dneigh1=sum(D.*NeighMask,2); % utilize the neighborhood mask incorporate(i.e. sum) the neighbors of each vector
            %NeighMask_squeeze = squeeze(NeighMask); % remove the dimension of length 1 for the next element-by-element matrix multiplication 
            %Dneigh=sum(D.*NeighMask_squeeze,2); % utilize the neighborhood mask incorporate(i.e. sum) the neighbors of each vector
            Dneigh=sum(D.*NeighMask,2); % utilize the neighborhood mask incorporate(i.e. sum) the neighbors of each vector
            % Neighborhood Affinity Contrast (NAC)
            %NAC_vol=Dneigh./Dglobal; % has one dimension of length 1 
            %NAC = squeeze(NAC_vol);
            NAC=Dneigh./Dglobal; 
        
        end


    end
end