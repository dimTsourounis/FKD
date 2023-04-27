clear

%% network

addpath(genpath('step2'));

% modelfile = fullfile('step2','base_model.onnx');
modelfile = fullfile('models','SigNet_teacher','base_model_feature_extractor_SigNet.onnx');
% % net = importONNXNetwork(modelfile,'OutputLayerType','classification');
lgraph = importONNXLayers(modelfile,'ImportWeights',true);
% analyzeNetwork(lgraph)

% dlnet1 (out from conv2: node 7 => 17 x 26 x 256)
newlgraph1 = removeLayers(lgraph,'node_8');   %
newlgraph1 = removeLayers(newlgraph1,'node_9');
newlgraph1 = removeLayers(newlgraph1,'node_10');
newlgraph1 = removeLayers(newlgraph1,'node_11');
newlgraph1 = removeLayers(newlgraph1,'node_12');
newlgraph1 = removeLayers(newlgraph1,'node_13');
newlgraph1 = removeLayers(newlgraph1,'node_14');
newlgraph1 = removeLayers(newlgraph1,'node_15');
newlgraph1 = removeLayers(newlgraph1,'node_16');
newlgraph1 = removeLayers(newlgraph1,'node_17');
newlgraph1 = removeLayers(newlgraph1,'node_18');
newlgraph1 = removeLayers(newlgraph1,'node_19');
newlgraph1 = removeLayers(newlgraph1,'node_20');
newlgraph1 = removeLayers(newlgraph1,'node_21');
newlgraph1 = removeLayers(newlgraph1,'node_22');
newlgraph1 = removeLayers(newlgraph1,'node_23');
newlgraph1 = removeLayers(newlgraph1,'node_24');
newlgraph1 = removeLayers(newlgraph1,'node_25');
newlgraph1 = removeLayers(newlgraph1,'node_26');
newlgraph1 = removeLayers(newlgraph1,'node_27');
newlgraph1 = removeLayers(newlgraph1,'node_28');
newlgraph1 = removeLayers(newlgraph1,'node_29');
newlgraph1 = removeLayers(newlgraph1,'node_30');
newlgraph1 = removeLayers(newlgraph1,'node_31');
newlgraph1 = removeLayers(newlgraph1,'node_32');
newlgraph1 = removeLayers(newlgraph1,'node_33');
newlgraph1 = removeLayers(newlgraph1,'node_34');
newlgraph1 = removeLayers(newlgraph1,'node_35');
% analyzeNetwork(newlgraph1)
%
% extract feature from node 7 (conv2)
dlnet1 = dlnetwork(newlgraph1);

% dlnet2 (out from conv3: node 11 => 8 x 12 x 384)
newlgraph2 = removeLayers(lgraph,'node_12');
newlgraph2 = removeLayers(newlgraph2,'node_13');
newlgraph2 = removeLayers(newlgraph2,'node_14');
newlgraph2 = removeLayers(newlgraph2,'node_15');
newlgraph2 = removeLayers(newlgraph2,'node_16');
newlgraph2 = removeLayers(newlgraph2,'node_17');
newlgraph2 = removeLayers(newlgraph2,'node_18');
newlgraph2 = removeLayers(newlgraph2,'node_19');
newlgraph2 = removeLayers(newlgraph2,'node_20');
newlgraph2 = removeLayers(newlgraph2,'node_21');
newlgraph2 = removeLayers(newlgraph2,'node_22');
newlgraph2 = removeLayers(newlgraph2,'node_23');
newlgraph2 = removeLayers(newlgraph2,'node_24');
newlgraph2 = removeLayers(newlgraph2,'node_25');
newlgraph2 = removeLayers(newlgraph2,'node_26');
newlgraph2 = removeLayers(newlgraph2,'node_27');
newlgraph2 = removeLayers(newlgraph2,'node_28');
newlgraph2 = removeLayers(newlgraph2,'node_29');
newlgraph2 = removeLayers(newlgraph2,'node_30');
newlgraph2 = removeLayers(newlgraph2,'node_31');
newlgraph2 = removeLayers(newlgraph2,'node_32');
newlgraph2 = removeLayers(newlgraph2,'node_33');
newlgraph2 = removeLayers(newlgraph2,'node_34');
newlgraph2 = removeLayers(newlgraph2,'node_35');
% analyzeNetwork(newlgraph2)
%
% extract feature from node 11 (conv3)
dlnet2 = dlnetwork(newlgraph2);

% dlnet3 (out from conv3: node 17 => 8 x 12 x 256)
newlgraph3 = removeLayers(lgraph,'node_18');
newlgraph3 = removeLayers(newlgraph3,'node_19');
newlgraph3 = removeLayers(newlgraph3,'node_20');
newlgraph3 = removeLayers(newlgraph3,'node_21');
newlgraph3 = removeLayers(newlgraph3,'node_22');
newlgraph3 = removeLayers(newlgraph3,'node_23');
newlgraph3 = removeLayers(newlgraph3,'node_24');
newlgraph3 = removeLayers(newlgraph3,'node_25');
newlgraph3 = removeLayers(newlgraph3,'node_26');
newlgraph3 = removeLayers(newlgraph3,'node_27');
newlgraph3 = removeLayers(newlgraph3,'node_28');
newlgraph3 = removeLayers(newlgraph3,'node_29');
newlgraph3 = removeLayers(newlgraph3,'node_30');
newlgraph3 = removeLayers(newlgraph3,'node_31');
newlgraph3 = removeLayers(newlgraph3,'node_32');
newlgraph3 = removeLayers(newlgraph3,'node_33');
newlgraph3 = removeLayers(newlgraph3,'node_34');
newlgraph3 = removeLayers(newlgraph3,'node_35');
% analyzeNetwork(newlgraph3)
%
% extract feature from node 17 (conv5)
dlnet3 = dlnetwork(newlgraph3);

% dlnet4 (out from fc7: node 35 => 1 x 1 x 2048)
params_network4 = importONNXFunction(modelfile,'FeatureExtractor');

%% MST
% calculate MST
% addpath('MST');
Radius = 5;

%% image
% save data considering one img into one mat-file
% generate supevisory signals using the Teacher model to constructe both the training and the validation sets in two runs

generate_supervisory_signals = 'Train';
% generate_supervisory_signals = 'Val';

if strcmp(generate_supervisory_signals, 'Train')
    %% training data
    path_data_train = fullfile('training_images','Training_Text_images_dataset');
    path_dataIn = path_data_train;
    save_path = fullfile('training_data','Training_Data');
elseif strcmp(generate_supervisory_signals,'Val')
    %% validation data
    path_data_val = fullfile('training_images','Validation_Text_images_dataset');
    path_dataIn = path_data_val;
    save_path = fullfile('training_data','Validation_Data');
end

%%
dataIn = dir(fullfile(path_dataIn,'*.tif'));

for i = 1: length(dataIn)

% read img -and center crop-
img = im2double(imread(fullfile(dataIn(i).folder, dataIn(i).name))); % 170 x 242
targetSize = [150 220]; % input image should be 150 x 220
img_center = centerCropWindow2d(size(img),targetSize); % center crop
I = imcrop(img,img_center); % 150 x 220 pixels 
data.image = I;
dlX = dlarray(I,"SSCB");

% set class GT label
img_name = dataIn(i).name;
class_char = img_name(8:10); % class is defined from image name
class_label = categorical(str2double(class_char),(1:310));
data.GT_label = class_label;

% extract Mask & GT NAC (using MST)
f1 = predict(dlnet1, dlX); % forward pass
FeatureMaps1 = extractdata(f1); % extract FeatureMaps Volume
[neighMask1,neighRatio1] = Find_Neighborhoods_n_Dists(FeatureMaps1,Radius); 
data.Mask1 = neighMask1;
data.GT_NAC1 = neighRatio1;

% extract Mask & GT NAC (using MST)
f2 = predict(dlnet2, dlX); % forward pass
FeatureMaps2 = extractdata(f2); % extract FeatureMaps Volume
[neighMask2,neighRatio2] = Find_Neighborhoods_n_Dists(FeatureMaps2,Radius); 
data.Mask2 = neighMask2;
data.GT_NAC2 = neighRatio2;

% extract Mask & GT NAC (using MST)
f3 = predict(dlnet3, dlX); % forward pass
FeatureMaps3 = extractdata(f3); % extract FeatureMaps Volume
[neighMask3,neighRatio3] = Find_Neighborhoods_n_Dists(FeatureMaps3,Radius); 
data.Mask3 = neighMask3;
data.GT_NAC3 = neighRatio3;

% extract GT Features 2048-dim
f4 = FeatureExtractor(I,params_network4); % forward pass -> % extract Feature f4: 2048 x 1
data.GT_Feature = f4; 

% save data as one structure variable in one mat-file for each one image
dataOut_name = [img_name(1:end-4) '_allInputData' '.mat']; % remove ".tif" and replace with ".mat"
save(fullfile(save_path,dataOut_name), 'data');
clear dataOut_name data ;

% if i == 10000 || i == 20000 || i == 30000 || i == 40000 || i == 50000 || i == 60000 || i == 70000 || i == 80000 || i == 90000 || i == 100000 || i == 110000 || i == 120000
%   pack;
% end

end