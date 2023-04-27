classdef FeatureExtractionLayer < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 
   % calculates the Euclidean distance matrix between input vectors and
   % returns the Neighborhood Affinity Contrast (NAC)
   

    methods
        function layer = FeatureExtractionLayer(name)
            % layer = FeatureExtractionLayer(name) creates a Feature Extraction Layer
            % for extracted the feature vector from a -previous- FC layer and specifies the layer
            % name.
			            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Feature Extractor';

        end
        
        function Feature = predict(layer, dataInput)
            % extract the feature vector
            % just pass the input vector to output
            % forward input data through the layer
            Feature = dataInput;
%             F = size(dataInput,1);
%             B = size(dataInput,4);
%             
%             Feature = zeros(F,1,1,B,'like',dataInput);
%             for i = 1 : B
%             Feature(:,1,1,i) = dataInput;
%             end
            
        
        end


    end
end