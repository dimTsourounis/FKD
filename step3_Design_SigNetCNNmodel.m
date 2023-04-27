clear;
%% net
% design the Student model following the SigNet architecture

% NumClassses = 310;
% 
% layers = [
%     % input
%     imageInputLayer([150 220 1],'Normalization','none','Name','ImgIn')
%     % layer1
%     convolution2dLayer(11,96,'Stride',4,'Padding',0,'PaddingValue',0,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     reluLayer('Name','relu1')
%     maxPooling2dLayer(3,'Stride',2,'Name','maxp1')
%     % layer2
%     convolution2dLayer(5,256,'Stride',1,'Padding',2,'PaddingValue',0,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     reluLayer('Name','relu2')
%     maxPooling2dLayer(3,'Stride',2,'Name','maxp2')
%     % layer3
%     convolution2dLayer(3,384,'Stride',1,'Padding',1,'PaddingValue',0,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','conv3')
%     batchNormalizationLayer('Name','bn3')
%     reluLayer('Name','relu3')
%     % layer4
%     convolution2dLayer(3,384,'Stride',1,'Padding',1,'PaddingValue',0,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','conv4')
%     batchNormalizationLayer('Name','bn4')
%     reluLayer('Name','relu4')
%     % layer5
%     convolution2dLayer(3,256,'Stride',1,'Padding',1,'PaddingValue',0,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','conv5')
%     batchNormalizationLayer('Name','bn5')
%     reluLayer('Name','relu5')
%     maxPooling2dLayer(3,'Stride',2,'Name','maxp5')
%     % fc6
%     fullyConnectedLayer(2048,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','fc6')
%     batchNormalizationLayer('Name','bn6')
%     reluLayer('Name','relu6')
%     % fc7
%     fullyConnectedLayer(2048,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','fc7')
%     batchNormalizationLayer('Name','bn7')
%     reluLayer('Name','relu7')
%     % fc out
%     fullyConnectedLayer(NumClassses,'WeightsInitializer','he','BiasLearnRateFactor',0,'Name','fcout')
%     softmaxLayer('Name','softmax')];
%     
% lgraphNet = layerGraph(layers);
% % analyzeNetwork(lgraphNet);
% % % dlNet = dlnetwork(lgraphNet);
% filename = 'SigNetGraphScratch.mat';
% save(fullfile('step3',filename),"lgraphNet");

addpath(genpath('step3'));
load('SigNetGraphScratch.mat');
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
lgraphNet = connectLayers(lgraphNet,'relu2','NAC1/in1');
lgraphNet = connectLayers(lgraphNet,'InMask1','NAC1/in2');

% NAC2
lgraphNet = connectLayers(lgraphNet,'relu3','NAC2/in1');
lgraphNet = connectLayers(lgraphNet,'InMask2','NAC2/in2');

% NAC3
lgraphNet = connectLayers(lgraphNet,'relu5','NAC3/in1');
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
lgraphNet = connectLayers(lgraphNet,'relu7','Feature');


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
