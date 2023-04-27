% This example shows how to extract features for a new signature,
% using a mat-file CNN.
% Also, this script includes how to transform the trained Student model from mat-file to ONNX-file

%%
clear;

%% Load model in mat-file (checkpoint)

% Define the path for the trained Student model (mat-file)
model_name = 'ResNet18_CL_KD_GEOM_BC';

path_mynet_mat = fullfile('checkpoints_trained_student',model_name,sprintf("dlnet_checkpoint__%s.mat", model_name));
load(path_mynet_mat); %dlnet

% % % transform dlnetwork to LayerGraph
%
lgraphNet = layerGraph(dlnet);
% % analyzeNetwork(lgraphNet);
% % figure, plot(lgraphNet)

% % % remove non-useful branches for feature extraction
%
% dlnet (out from layer 7 (fc7>>bn7>>relu7>>) => 1 x 1 x 2048)
newlgraphNet = removeLayers(lgraphNet,'InMask1');   
newlgraphNet = removeLayers(newlgraphNet,'NAC1');
newlgraphNet = removeLayers(newlgraphNet,'InMask2');
newlgraphNet = removeLayers(newlgraphNet,'NAC2');
newlgraphNet = removeLayers(newlgraphNet,'InMask3');
newlgraphNet = removeLayers(newlgraphNet,'NAC3');
newlgraphNet = removeLayers(newlgraphNet,'fcout');
newlgraphNet = removeLayers(newlgraphNet,'softmax');
newlgraphNet = removeLayers(newlgraphNet,'Feature');
%
% analyzeNetwork(newlgraphNet);
% figure, plot(newlgraphNet)

% % %  transform LayerGraph to dlnetwork 
dlnet_feature_extractor = dlnetwork(newlgraphNet); % CNN as feature extractor

% % % save the new dlnetwork, this dlnetwork could be used as feature extractor 
% path_mynetFeature_mat = fullfile('models','matModels',sprintf('feature_extractor_%s.mat',model_name));
% mkdir(fullfile('models','matModels'));
% save(path_mynetFeature_mat,"dlnet_feature_extractor")


%% Utilize CNN (mat-file model) for feature extraction

% clear;

% % % load the signature
path_img_processed = fullfile('sigver_WD','data','some_signature_processed.png');
im = im2double(imread(path_img_processed));
dlX = dlarray(im,"SSCB");

% % % load mat-file model
model_name = 'ResNet18_CL_KD_GEOM_BC';
path_mynetFeature_mat = fullfile('models','matModels',sprintf('feature_extractor_%s.mat',model_name));
load(path_mynetFeature_mat); %dlnet_feature_extractor

% % % extract features
f = predict(dlnet_feature_extractor, dlX); % forward pass | f: feature_dim x 1 (dlarray)
f = extractdata(f); % f: feature_dim x 1 (single)


%% Export model in onnx-file

% % % save model in onnx-file
path_mynet_onnx = fullfile('models','onnxModels', sprintf('feature_extractor_%s.onnx',model_name));
mkdir(fullfile('models','onnxModels'));

opv = 13;
exportONNXNetwork(dlnet_feature_extractor,path_mynet_onnx, 'OpsetVersion', opv) % opv 6 - 13


%% Utilize CNN (onnx-file model) for feature extraction

% clear;

% % % load the signature
path_img_processed = fullfile('sigver_WD','data','some_signature_processed.png');
im = im2double(imread(path_img_processed));

% % % load onnx-file model
model_name = 'ResNet18_CL_KD_GEOM_BC';
path_mynet_onnx = fullfile('models','onnxModels', sprintf('feature_extractor_%s.onnx',model_name));
%
% % 1st approach as LayerGraph:
% lgraph = importONNXLayers(path_mynet_onnx);
% analyzeNetwork(lgraph);
% 
% % 2nd approach as DAGNetwork:
net = importONNXNetwork(path_mynet_onnx);

% % % extract features
%
% % 1st approach:
% params_network = importONNXFunction(path_mynet_onnx,'FeatureExtractor');
% f = FeatureExtractor(im,params_network); % forward pass | f: feature_dim x 1 (double)
%
% % 2nd approach:
f = predict(net,im)'; % forward pass | f: feature_dim x 1 (single)





