%% functions paths 
addpath(genpath('step4'));

%% data paths

data_train_list = dir(fullfile('training_data','Training_Data','*.mat'));
% Validation Data
path_data_Val = fullfile('training_data','Validation_Data');

% Trained results
checkpoints_name = "checkpoints_trained_student";
checkpointPath = fullfile(checkpoints_name);
mkdir(checkpointPath)

%% Visualize the training progress in a plot
plots = "training-progress";

% Initialize the training progress plot.
if plots == "training-progress"
    Fig_trainProg = figure;
    subplot(2,3,1);lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss');
    subplot(2,3,2);lineAccClassTrain = animatedline('Color',[0.85 0.325 0.098]);ylim([0 1]);xlabel("Iteration");ylabel("Accuracy");title('Accuracy class');
    subplot(2,3,3);lineLossCETrain = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss Classification');
    subplot(2,3,4);lineLossNAC1Train = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss NAC1');
    %subplot(2,3,5);lineLossNAC2Train = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss NAC2');
    subplot(2,3,5);lineLossFeatureTrain = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss FeatureIm');
    subplot(2,3,6);lineLossNAC3Train = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss NAC3');
    
    %grid on
end

%% Visualize the training progress in a plot using the Validation data
plotsVal = "training-progress-validation";

% Initialize the training progress plot.
if plotsVal == "training-progress-validation"
    Fig_valProg = figure;
    subplot(2,3,1);lineLossVal = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss - Val');
    subplot(2,3,2);lineAccClassVal = animatedline('Color',[0.85 0.325 0.098]);ylim([0 1]);xlabel("Iteration");ylabel("Accuracy");title('Accuracy class - Val');
    subplot(2,3,3);lineLossCEVal = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss Classification - Val');
    subplot(2,3,4);lineLossNAC1Val = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss NAC1 - Val');
    %subplot(2,3,5);lineLossNAC2Val = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss NAC2 - Val');
    subplot(2,3,5);lineLossFeatureVal = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss FeatureIm - Val');
    subplot(2,3,6);lineLossNAC3Val = animatedline('Color',[0.85 0.325 0.098]);ylim([0 inf]);xlabel("Iteration");ylabel("Loss");title('Loss NAC3 - Val');
end

%% Training hyper-parameters

D = gpuDevice(1);

% % Initialize parameters for Adam.
% trailingAvg = [];
% trailingAvgSq = [];
% learnRate = 0.01;
% gradDecay = 0.90;
% sqGradDecay = 0.95;

% Initialize parameters for SGDM optimizer
vel = [];
learnRate = 0.01;
momentum = 0.9;

NumEpochs = 60;
NumOfElementsPerSubset = 2000; % number of elements per subset
minibatch_size = 64;

% for efficient memory usage: 
% all the available data are partitioned into discrete Subsets in each epoch
% each Subset has #NumOfElementsPerSubset training data 
% the training data of each Subset create minibatches using #minibatch_size samples for each minibatch
% thus, the overall training process has: Loop over epochs -> Loop over Subsets -> Loop over minibatches 

%%
iteration = 0; % for every mini-batch update

%%% Loop over epochs (1 epoch => 1 set) %%%
for epoch = 1 : NumEpochs 

    % shuffle data
    rand_indices = randperm(length(data_train_list)); % perform random permutations of data => shuffle
    % create subsets using the shuffled data
    %NumOfElementsPerSubset = 10; % number of elements per subset
    NumOfSubsets = floor(length(rand_indices)/NumOfElementsPerSubset); % number of subsets
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
            j = rand_indices(datum_idx); % shuffled_datum_idx
            load(fullfile(data_train_list(j).folder,data_train_list(j).name)); % load=> data => 9 structured Fields
            
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
        % |images |class labels | Masks1 | NACs1 | Masks2 | NACs2 |Masks | NACs3 | Features
        dsTrain = combine(dsImgTrain,dsLabelsTrain,dsMasks1Train,dsNACs1Train,dsMasks2Train,dsNACs2Train,dsMasks3Train,dsNACs3Train,dsFeaturesTrain);
        
        % minibatch creation
        mbq = minibatchqueue(dsTrain,...
            'MiniBatchSize',minibatch_size,...
            'MiniBatchFcn', @preprocessMSTfeatIMnet,...
            'MiniBatchFormat',{'SSCB','','SSCB','','SSCB','','SSCB','', ''});
            
        %[dlX,dlY,dlMasks1,dlNACs1,dlMasks2,dlNACs2,dlMasks3,dlNACs3,dlFeatures] = next(mbq);
        
        % shuffle mini-batch data
        shuffle(mbq);
        
        
%%% display the training progress using the Validation data at iteration = 0 (before 1st update) - initial conditions
        if plotsVal == "training-progress-validation" && iteration == 0 
            [lossVal, lossClassificationVal,lossNAC1Val,lossNAC2Val,lossNAC3Val,lossFeatureVal, AccClassVal] = CalculateMetrics_featIM_onSet(dlnet, ...
                path_data_Val, NumOfElementsPerSubset,minibatch_size);
   
            addpoints(lineLossVal,iteration,double(gather(extractdata(lossVal))));
            addpoints(lineAccClassVal,iteration,double(AccClassVal));
            addpoints(lineLossCEVal,iteration,double(gather(extractdata(lossClassificationVal))));
            addpoints(lineLossNAC1Val,iteration,double(gather(extractdata(lossNAC1Val))));
            %addpoints(lineLossNAC2Val,iteration,double(gather(extractdata(lossNAC2Val))));
            addpoints(lineLossFeatureVal,iteration,double(gather(extractdata(lossFeatureVal))));
            addpoints(lineLossNAC3Val,iteration,double(gather(extractdata(lossNAC3Val))));
                                
            drawnow
        end
        
%%% Loop over mini-batches %%%
        while hasdata(mbq)
            
            iteration = iteration + 1;
            
            [dlX,dlY,dlMasks1,dlNACs1,dlMasks2,dlNACs2,dlMasks3,dlNACs3,dlFeatures] = next(mbq);
            % dlX: input images -> H x W x C x B
            % dlY: GT class labels -> B x 1
            % dlMask: Masks (using MST) -> L x L x 1 x B
            % dlNACs: GT NACs (usingn MST) -> L x B
            % dlFeatures: GT Features -> F x B
            
            % Evaluate the model gradients, state, and loss using dlfeval and the modelGradients function.
            [gradients,state,loss, lossClassification,lossNAC1,lossNAC2,lossNAC3,lossFeature, Acc_class] = dlfeval(@modelGradients_CE_NACs_featIM, ...
                dlnet,dlX,dlY,dlMasks1,dlNACs1,dlMasks2,dlNACs2,dlMasks3,dlNACs3,dlFeatures);
            dlnet.State = state;
            
            % Update the network parameters using the Adam optimizer.
%             [dlnet,trailingAvg,trailingAvgSq] = adamupdate(dlnet,gradients, ...
%             trailingAvg,trailingAvgSq,iteration);
            %
%             [dlnet,trailingAvg,trailingAvgSq] = adamupdate(dlnet,gradients, ...
%             trailingAvg,trailingAvgSq,iteration, learnRate,gradDecay,sqGradDecay);
        
            % Update the network parameters using the SGDM optimizer.
            [dlnet,vel] = sgdmupdate(dlnet,gradients,vel, learnRate,momentum);
            
        
            % Display the training progress.
            if plots == "training-progress"
                addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
                addpoints(lineAccClassTrain,iteration,double(Acc_class))
                addpoints(lineLossCETrain,iteration,double(gather(extractdata(lossClassification))))
                addpoints(lineLossNAC1Train,iteration,double(gather(extractdata(lossNAC1))))
                %addpoints(lineLossNAC2Train,iteration,double(gather(extractdata(lossNAC2))))
                addpoints(lineLossFeatureTrain,iteration,double(gather(extractdata(lossFeature))))
                addpoints(lineLossNAC3Train,iteration,double(gather(extractdata(lossNAC3))))
                                
                %title("Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end
            
        end %minibatches end == one subset end
        
        
    end %subsets end == one epoch end

    % During training, at the end of an epoch, display the training progress using the Validation data
    if plotsVal == "training-progress-validation"
        [lossVal, lossClassificationVal,lossNAC1Val,lossNAC2Val,lossNAC3Val,lossFeatureVal, AccClassVal] = CalculateMetrics_featIM_onSet(dlnet, ...
            path_data_Val, NumOfElementsPerSubset,minibatch_size);
   
        addpoints(lineLossVal,iteration,double(gather(extractdata(lossVal))));
        addpoints(lineAccClassVal,iteration,double(AccClassVal));
        addpoints(lineLossCEVal,iteration,double(gather(extractdata(lossClassificationVal))));
        addpoints(lineLossNAC1Val,iteration,double(gather(extractdata(lossNAC1Val))));
        %addpoints(lineLossNAC2Val,iteration,double(gather(extractdata(lossNAC2Val))));
        addpoints(lineLossFeatureVal,iteration,double(gather(extractdata(lossFeatureVal))));
        addpoints(lineLossNAC3Val,iteration,double(gather(extractdata(lossNAC3Val))));
                                
        drawnow
    end
    
    savefig(Fig_trainProg,fullfile(checkpointPath,'TrainProgress.fig'));
    savefig(Fig_valProg,fullfile(checkpointPath,'ValProgress.fig'));
    
    
    % During training, at the end of an epoch, save the network in a MAT file.
    if ~isempty(checkpointPath)
    D = datestr(now,'yyyy_mm_dd__HH_MM_SS');
    filename = "dlnet_checkpoint__iter" + iteration + "__" + D + ".mat";
    save(fullfile(checkpointPath,filename),"dlnet")
    end
    
    if epoch == 20 || epoch == 40 %|| epoch == 60 
    %if epoch == 40 || epoch == 60     
    learnRate = learnRate * 0.1; % drop learning rate
    end
    
end %epochs end
            




















