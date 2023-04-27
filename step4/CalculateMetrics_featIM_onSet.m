function [loss, lossClassification,lossNAC1,lossNAC2,lossNAC3,lossFeature, AccClass] = CalculateMetrics_featIM_onSet(dlnet, path_data, NumOfElementsPerSubset,minibatch_size)
% This custom function calculate the metrics in the validation set of Training Data.
% This function is utilized to plot some metrics during training.

% loss, lossClassification,lossNAC1,lossNAC2,lossNAC3, lossFeature, AccClass -> metrics
% (total loss, CE loss, mse , mse, mse , mse, Accuracy)

% NumOfElementsPerSubset, minibatch_size -> for efficient memory usage
 

%% functions paths 
% addpath('step4');

%% data paths

data_list = dir(path_data);
data_list(1:2) = [];

%% Training hyper-parameters

% NumOfElementsPerSubset = ; 
% minibatch_size = ;

%%
iteration = 0; % for every mini-batch update

NumOfSubsets = floor(length(data_list)/NumOfElementsPerSubset); % number of subsets
%%%% Loop over subsets (1 set => #subsets) %%%
    for s = 1:NumOfSubsets

        % subset creation
        %
        % pre-allocate variables
        images = zeros(150,220,1,NumOfElementsPerSubset); 
        GT_labels = categorical(zeros(NumOfElementsPerSubset,1),(1:310));
        Masks1 = false(442,442,1,NumOfElementsPerSubset);
        GT_NACs1 = single(zeros(442,NumOfElementsPerSubset));
        Masks2 = false(96,96,1,NumOfElementsPerSubset);
        GT_NACs2 = single(zeros(96,NumOfElementsPerSubset));
        Masks3 = false(96,96,1,NumOfElementsPerSubset);
        GT_NACs3 = single(zeros(96,NumOfElementsPerSubset));
        GT_Features = single(zeros(2048,NumOfElementsPerSubset));
        for k = 1:NumOfElementsPerSubset % subset creation
            
            % select and load datum for creating a subset
            datum_idx = (s-1)*NumOfElementsPerSubset + k; % enumerate through shuffled data
            %
            j = datum_idx;
            load(fullfile(data_list(j).folder,data_list(j).name)); % load=> data => 9 structured Fields
            
            % store images (150x220) => 150 x 220 x 1 x NumOfElementsPerSubset
            images(:,:,1,k) = data.image; %images(:,:,2,k) = data.image; images(:,:,3,k) = data.image;
            % store GT_labels (class labels) => NumOfElementsPerSubset x 1
            GT_labels(k,1) = data.GT_label;
            % store Masks (MST) => H_Mask x W_Mask x 1 x NumOfElementsPerSubset
            Masks1(:,:,1,k) = data.Mask1;
            % store GT_NACs (MST) => dim_NAC x NumOfElementsPerSubset , (dim_NAC = H_Mask = W_Mask) 
            GT_NACs1(:,k) = data.GT_NAC1;
            % store Masks (MST) => H_Mask x W_Mask x 1 x NumOfElementsPerSubset
            Masks2(:,:,1,k) = data.Mask2;
            % store GT_NACs (MST) => dim_NAC x NumOfElementsPerSubset , (dim_NAC = H_Mask = W_Mask) 
            GT_NACs2(:,k) = data.GT_NAC2;
            % store Masks (MST) => H_Mask x W_Mask x 1 x NumOfElementsPerSubset
            Masks3(:,:,1,k) = data.Mask3;
            % store GT_NACs (MST) => dim_NAC x NumOfElementsPerSubset , (dim_NAC = H_Mask = W_Mask) 
            GT_NACs3(:,k) = data.GT_NAC3;
            % store GT_Features (2048) => dim_Feature x NumOfElementsPerSubset , (dim_Feature = 2048) 
            GT_Features(:,k) = data.GT_Feature;
            
        end %subset end

   
        % images: M x N x 1 x B
        dsImgTrain = arrayDatastore(images,'IterationDimension',4); %4-th dim is the number of images
        % GT_labels: B x 1
        dsLabelsTrain = arrayDatastore(GT_labels); %1-st dim is the number of images
        % Masks: L x L x 1 x B
        dsMasks1Train = arrayDatastore(Masks1,'IterationDimension',4); %4-th dim is the number of images
        % GT_NAC: L x B
        dsNACs1Train = arrayDatastore(GT_NACs1,'IterationDimension',2);%2-nd dim (column-oriented data) is the number of images);
        % Masks: L x L x 1 x B
        dsMasks2Train = arrayDatastore(Masks2,'IterationDimension',4);
        % GT_NAC: L x B
        dsNACs2Train = arrayDatastore(GT_NACs2,'IterationDimension',2);
        % Masks: L x L x 1 x B
        dsMasks3Train = arrayDatastore(Masks3,'IterationDimension',4);
        % GT_NAC: L x B
        dsNACs3Train = arrayDatastore(GT_NACs3,'IterationDimension',2);
        % GT_Feature: F x B
        dsFeaturesTrain = arrayDatastore(GT_Features,'IterationDimension',2);
        %%% readall(dsMasks1Train)
        
        % combine data
        % |images |class labels | Masks1 | NACs1 | Masks2 | NACs2 |Masks | NACs3 |
        dsTrain = combine(dsImgTrain,dsLabelsTrain,dsMasks1Train,dsNACs1Train,dsMasks2Train,dsNACs2Train,dsMasks3Train,dsNACs3Train,dsFeaturesTrain);
        
        % minibatch creation
        mbq = minibatchqueue(dsTrain,...
            'MiniBatchSize',minibatch_size,...
            'MiniBatchFcn', @preprocessMSTfeatIMnet,...
            'MiniBatchFormat',{'SSCB','','SSCB','','SSCB','','SSCB','',''});
            
        %[dlX,dlY,dlMasks1,dlNACs1,dlMasks2,dlNACs2,dlMasks3,dlNACs3] = next(mbq);
        

        
%%% Loop over mini-batches %%%
        i = 0;
        while hasdata(mbq)
            
            iteration = iteration + 1;
            i = i + 1;
            
            [dlX,dlY,dlMasks1,dlNACs1,dlMasks2,dlNACs2,dlMasks3,dlNACs3,dlFeatures] = next(mbq);
            % dlX: input images -> H x W x C x B
            % dlY: GT class labels -> B x 1
            % dlMask: Masks (using MST) -> L x L x 1 x B
            % dlNACs: GT NACs (usingn MST) -> L x B
            % dlFeatures: GT Features -> F x B
            
            % forward pass of dlnet
            [pred_softmax_label,pred_NAC1,pred_NAC2,pred_NAC3,pred_Feature] = predict(dlnet,dlX,dlMasks1,dlMasks2,dlMasks3,'Outputs',["softmax" "NAC1" "NAC2" "NAC3" "Feature"]);

    %%% calculate the losses
            
            % classification loss (Cross Entropy - CE)
            % CE
            lossClassification = crossentropy(pred_softmax_label,dlY);

            % Neighborhood Affinity Contrast loss (NAC based on MST)
            % NAC1
            pred_NAC1_squ = squeeze(pred_NAC1);
            lossNAC1 = mse(pred_NAC1_squ,dlNACs1);
            %lossNAC1 = l1loss(pred_NAC1_squ,dlNACs1);
            %lossNAC1 = l2loss(pred_NAC1_squ,dlNACs1)

            % Neighborhood Affinity Contrast loss (NAC based on MST)
            % NAC2
            pred_NAC2_squ = squeeze(pred_NAC2);
            lossNAC2 = mse(pred_NAC2_squ,dlNACs2);
            %lossNAC2 = l1loss(pred_NAC2_squ,dlNACs2);
            %lossNAC2 = l2loss(pred_NAC2_squ,dlNACs2);

            % Neighborhood Affinity Contrast loss (NAC based on MST)
            % NAC3
            pred_NAC3_squ = squeeze(pred_NAC3);
            lossNAC3 = mse(pred_NAC3_squ,dlNACs3);
            %lossNAC3 = l1loss(pred_NAC3_squ,dlNACs3);
            %lossNAC3 = l2loss(pred_NAC3_squ,dlNACs3);
            
            % Feature Imitation loss (extreme case of distillation loss)
            % Feature
            pred_Feature_squ = squeeze(pred_Feature);
            %lossFeature = mse(pred_Feature_squ,dlFeatures);
            %lossFeature = l1loss(pred_Feature_squ,dlFeatures);
            %lossFeature = l2loss(pred_Feature_squ,dlFeatures);
            pred_Feature_norm = pred_Feature_squ./(sqrt(sum(pred_Feature_squ.*pred_Feature_squ,1)) +10^(-6)); % l2norm
            % pred_Feature_norm = pred_Feature_squ./(max(pred_Feature_squ,[],1) +10^(-6)); % range [0,1]
            %pred_Feature_norm = (pred_Feature_squ - mean(pred_Feature_squ,2))./(std(pred_Feature_squ,[],2) +10^(-6)); % stndrz along batch
            GT_Feature_norm = dlFeatures./(sqrt(sum(dlFeatures.*dlFeatures,1)) +10^(-6));
            % GT_Feature_norm = dlFeatures./(max(dlFeatures,[],1) +10^(-6)); % range [0,1]
            %GT_Feature_norm = (GT_Feature - mean(GT_Feature,2))./(std(GT_Feature,[],2) +10^(-6)); % stndrz along batch
            lossFeature = mse(pred_Feature_norm,GT_Feature_norm);
            % lossFeature = crossentropy(pred_Feature_norm,GT_Feature_norm); 
            % lossFeature = crossentropy(softmax(pred_Feature_norm),softmax(dlarray(GT_Feature_norm,'CB'))); % λ^2, λ=2 -> + 4*lossFeature
            % temperature = 1;
            % pred_Feature_sofmax_with_t = exp(pred_Feature_norm/temperature)./sum(exp(pred_Feature_norm/temperature),1);
            % GT_Feature_sofmax_with_t = exp(GT_Feature_norm/temperature)./sum(exp(GT_Feature_norm/temperature),1);
            % lossFeature = crossentropy(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);
            %
            % temperature = 10;
            % pred_Feature_sofmax_with_t = exp(pred_Feature_squ/temperature)./sum(exp(pred_Feature_squ/temperature),1);
            % GT_Feature_sofmax_with_t = exp(dlFeatures/temperature)./sum(exp(dlFeatures/temperature),1);
            %lossFeature = crossentropy(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);
            %lossFeature = mse(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);
            % lossFeature = l1loss(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);

            % total loss (as combination of the above losses)
            loss = lossClassification + lossNAC1 + lossNAC2 + lossNAC3 + lossFeature;
            
    %%% calculate the accuracy
           
            % Class Accuracy
            % determine predicted classes
            pred_class_label = onehotdecode(pred_softmax_label,(1:310),1); % [~,pred_class_label]=max(pred_softmax_label,[],1);
            GT_class_label = onehotdecode(dlY,(1:310),1); % [~,GT_class_label]=max(dlY,[],1);
            % compare predicted and GT classes
            Correct_class = pred_class_label == GT_class_label;
            Acc_class_batch(i) = mean(Correct_class);

        end %minibatches end == one subset end
        Acc_class_OverBatches = mean(Acc_class_batch); % Acc_class_OverBatches == Acc_class_OverSubset
        
    end %subsets end == one epoch end
    AccClass = mean(Acc_class_OverBatches); % AccClass == Acc_class_OverSubsets == Acc_class_OverEpoch == Acc_class_OverValData
    



end
















