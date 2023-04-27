clear;
%% net
% design the Student model following the ResNet-18 architecture

%
% %% define the backbone of a ResNet architecture
% lgraph0 = resnet18('Weights','none');
% % % net = resnet18('Weights','imagenet');
% % % lgraph0 = layerGraph(net);
% % analyzeNetwork(lgraph0);
% % figure, plot(lgraph0)
% 
% %% modify the ResNet layers with the layers in larray (in order to fit with our task)
% 
% % replace input (image)
% ImgIn = imageInputLayer([150 220 1],'Normalization','none','Name','ImgIn');
% lgraph = replaceLayer(lgraph0,'data',ImgIn);
% 
% 
% % modify architecture to capture the proper spatial dimensions ([17 x 26] & [8 x 12])
% deepNetworkDesigner % for modify lgraph and export lgraph_1 or lgraph_2
% %
% % conv1: Padding = 0 0 0 0
% %conv1 = convolution2dLayer(7,64,'Stride',2,'Padding',0,'PaddingValue',0,'WeightsInitializer','glorot','BiasLearnRateFactor',0,'Name','conv1');
% % pool1: Padding = 0 0 0 0 
% %pool1 = maxPooling2dLayer(3,'Stride',2,'Name','pool1');
% % res3a_branch1 -> bn3a_branch1 >> pool3a_branch1: 2,'Stride',1,'Padding',[0 0 0 0] 
% %pool3a_branch1 = maxPooling2dLayer(2,'Stride',1,'Padding',[0 0 0 0],'Name','pool3a_branch1');
% % res3a_branch2a: Padding = 0 0 0 0
% % res4a_branch1 -> bn4a_branch1 >> pool4a_branch1: 2,'Stride',1,'Padding',[0 0 0 0] 
% % res4a_branch2a: Padding = 0 0 0 0
% 
% % % replace output (class propabilities) & insert intermediate fully connected layer for feature extraction 
% % % remove last fc layer, softmax layer, and CE classification layer 
% % clear lgraph;lgraph = lgraph_2; % from deepNetworkDesigner
% % lgraph = removeLayers(lgraph,'ClassificationLayer_predictions'); % remove the CE classification layer that computes the cross-entropy loss
% % lgraph = removeLayers(lgraph,'prob'); % remove the softmax layer 
% % lgraph = removeLayers(lgraph,'fc1000'); % remove the fully connected layer corresponding to the 1000 classes
% % % define last layers: fc -> bn -> relu -> fc -> softmax(classes)
% % feature_dim = 2048;
% % NumClassses = 310;
% % finalLayer = [
% %     % fc (for feature extraction)
% %     fullyConnectedLayer(feature_dim,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','fc_feature')
% %     batchNormalizationLayer('Name','bn_feature')
% %     reluLayer('Name','relu_feature')
% %     % fc (for classes)
% %     fullyConnectedLayer(NumClassses,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','fcout')
% %     softmaxLayer('Name','softmax')];
% % % add layers 
% % lgraph = addLayers(lgraph,finalLayer);
% % % connect layers
% % lgraph = connectLayers(lgraph,'pool5','fc_feature');
% % % analyzeNetwork(lgraph);
% 
% 
% %% save modified model
% lgraphNet = lgraph;
% filename = 'ResNetGraphScratch6.mat';
% save(fullfile('step3',filename),"lgraphNet");
% %
% load('ResNetGraphScratch6.mat');
% analyzeNetwork(lgraphNet);

addpath(genpath('step3'));
load('ResNetGraphScratch6.mat');
% analyzeNetwork(lgraphNet);

%% NAC layers

% addpath(genpath('step3'));


% NAC1
% define layer 
layerInMask1 = imageInputLayer([17*26 17*26],'Normalization','none','Name','InMask1');
% put layer in the field
lgraphNet = addLayers(lgraphNet,layerInMask1);
% define layer
layerNAC1 = NACLayer(2, 'NAC1');
% put layer in the field
lgraphNet = addLayers(lgraphNet,layerNAC1);

% NAC2
% define layer 
layerInMask2 = imageInputLayer([8*12 8*12],'Normalization','none','Name','InMask2');
% put layer in the field
lgraphNet = addLayers(lgraphNet,layerInMask2);
% define layer
layerNAC2 = NACLayer(2, 'NAC2');
% put layer in the field
lgraphNet = addLayers(lgraphNet,layerNAC2);

% NAC3
% define layer 
layerInMask3 = imageInputLayer([8*12 8*12],'Normalization','none','Name','InMask3');
% put layer in the field
lgraphNet = addLayers(lgraphNet,layerInMask3);
% define layer
layerNAC3 = NACLayer(2, 'NAC3');
% put layer in the field
lgraphNet = addLayers(lgraphNet,layerNAC3);

%% connect NAC layers
% connect NAClayer with the net

% NAC1
lgraphNet = connectLayers(lgraphNet,'res3b_relu','NAC1/in1');
lgraphNet = connectLayers(lgraphNet,'InMask1','NAC1/in2');

% NAC2
lgraphNet = connectLayers(lgraphNet,'res4a_relu','NAC2/in1');
lgraphNet = connectLayers(lgraphNet,'InMask2','NAC2/in2');

% NAC3
lgraphNet = connectLayers(lgraphNet,'res4b_relu','NAC3/in1');
lgraphNet = connectLayers(lgraphNet,'InMask3','NAC3/in2');

%% Feature extraction layer
% Feature imitation - extreme distillation loss

% Feature
% define layer
layerFeature = FeatureExtractionLayer('Feature');
% put layer in the field
lgraphNet = addLayers(lgraphNet,layerFeature);

%% connect Feature Extraction layer
% connect layerFeature with the net

% Feature
lgraphNet = connectLayers(lgraphNet,'relu_feature','Feature');


%% print network topology
% analyzeNetwork(lgraphNet);
% figure, plot(lgraphNet)

%% dlnetwork object 

%  A dlnetwork object allows you to train a network specified as a layer graph using automatic differentiation.
dlnet = dlnetwork(lgraphNet);

% % calculate learnable parameters of a network model
% layers=dlnet.Learnables.Value;
% num_layers = size(layers,1);
% num_para=0;
% for i=1:num_layers
%    num_para=num_para+prod(size(layers{i}));
% end
% num_para

clearvars -except dlnet lgraphNet

