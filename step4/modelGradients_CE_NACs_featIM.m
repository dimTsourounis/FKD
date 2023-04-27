function [gradients,state,loss, lossClassification,lossNAC1,lossNAC2,lossNAC3,lossFeature, Acc_class] = modelGradients_CE_NACs_featIM(dlnet,dlX,GT_label,dlMasks1,GT_NAC1,dlMasks2,GT_NAC2,dlMasks3,GT_NAC3,GT_Feature)
% This custom function uses dlgradient to compute derivatives using automatic differentiation for custom training loop.
% This custom function is utilized to create the FKD losses.

% To compute network outputs for training, use the forward function. To compute network outputs for inference, use the predict function.
[pred_softmax_label,pred_NAC1,pred_NAC2,pred_NAC3,pred_Feature, state] = forward(dlnet,dlX,dlMasks1,dlMasks2,dlMasks3,'Outputs',["softmax" "NAC1" "NAC2" "NAC3" "Feature"]);

% classification loss (Cross Entropy - CE)
% CE
lossClassification = crossentropy(pred_softmax_label,GT_label);



% Neighborhood Affinity Contrast loss (NAC based on MST)
% NAC1
pred_NAC1_squ = squeeze(pred_NAC1);
lossNAC1 = mse(pred_NAC1_squ,GT_NAC1);
% lossNAC1 = l1loss(pred_NAC1_squ,GT_NAC1);
% lossNAC1 = l2loss(pred_NAC1_squ,GT_NAC1);
% pred_NAC1_sofmax_with_t = exp(pred_NAC1_squ/temperature)./sum(exp(pred_NAC1_squ/temperature),1);
% GT_NAC1_sofmax_with_t = exp(GT_NAC1/temperature)./sum(exp(GT_NAC1/temperature),1);
% lossNAC1 = crossentropy(pred_NAC1_sofmax_with_t,GT_NAC1_sofmax_with_t);

% Neighborhood Affinity Contrast loss (NAC based on MST)
% NAC2
pred_NAC2_squ = squeeze(pred_NAC2);
lossNAC2 = mse(pred_NAC2_squ,GT_NAC2);
% lossNAC2 = l1loss(pred_NAC2_squ,GT_NAC2);
% lossNAC2 = l2loss(pred_NAC2_squ,GT_NAC2);
% pred_NAC2_sofmax_with_t = exp(pred_NAC2_squ/temperature)./sum(exp(pred_NAC2_squ/temperature),1);
% GT_NAC2_sofmax_with_t = exp(GT_NAC2/temperature)./sum(exp(GT_NAC2/temperature),1);
% lossNAC2 = crossentropy(pred_NAC2_sofmax_with_t,GT_NAC2_sofmax_with_t);

% Neighborhood Affinity Contrast loss (NAC based on MST)
% NAC3
pred_NAC3_squ = squeeze(pred_NAC3);
lossNAC3 = mse(pred_NAC3_squ,GT_NAC3);
% lossNAC3 = l1loss(pred_NAC3_squ,GT_NAC3);
% lossNAC3 = l2loss(pred_NAC3_squ,GT_NAC3);
% pred_NAC3_sofmax_with_t = exp(pred_NAC3_squ/temperature)./sum(exp(pred_NAC3_squ/temperature),1);
% GT_NAC3_sofmax_with_t = exp(GT_NAC3/temperature)./sum(exp(GT_NAC3/temperature),1);
% lossNAC3 = crossentropy(pred_NAC3_sofmax_with_t,GT_NAC3_sofmax_with_t);

% Feature Imitation loss (extreme case of distillation loss)
% Feature
pred_Feature_squ = squeeze(pred_Feature);
% lossFeature = mse(pred_Feature_squ,GT_Feature); % batch normalization
% lossFeature = l1loss(pred_Feature_squ, GT_Feature);
% lossFeature = l2loss(pred_Feature_squ,GT_Feature);
% pred_Feature_norm = pred_Feature_squ./(sqrt(sum(pred_Feature_squ.*pred_Feature_squ,1)) +10^(-6)); % l2norm
% pred_Feature_norm = pred_Feature_squ./(max(pred_Feature_squ,[],1) +10^(-6)); % range [0,1]
% pred_Feature_norm = pred_Feature_squ./(std(pred_Feature_squ,[],1) +10^(-6)); % unit variance UV
pred_Feature_norm = (pred_Feature_squ - mean(pred_Feature_squ,2))./(std(pred_Feature_squ,[],2) +10^(-6)); % stndrz along batch
% GT_Feature_norm = GT_Feature./(sqrt(sum(GT_Feature.*GT_Feature,1)) +10^(-6)); % l2norm
% GT_Feature_norm = GT_Feature./(max(GT_Feature,[],1) +10^(-6)); % range [0,1]
% GT_Feature_norm = GT_Feature./(std(GT_Feature,[],1) +10^(-6)); % unit variance UV
GT_Feature_norm = (GT_Feature - mean(GT_Feature,2))./(std(GT_Feature,[],2) +10^(-6)); % stndrz along batch
% lossFeature = mse(pred_Feature_norm,GT_Feature_norm);
% lossFeature = crossentropy(pred_Feature_norm,GT_Feature_norm); 
% lossFeature = crossentropy(softmax(pred_Feature_norm),softmax(dlarray(GT_Feature_norm,'CB'))); % λ^2, λ=2 -> + 4*lossFeature
% temperature = 10;
% pred_Feature_sofmax_with_t = exp(pred_Feature_norm/temperature)./sum(exp(pred_Feature_norm/temperature),1);
% GT_Feature_sofmax_with_t = exp(GT_Feature_norm/temperature)./sum(exp(GT_Feature_norm/temperature),1);
% lossFeature = crossentropy(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);


% loss_feat = dlarray(zeros(1,size(pred_Feature_norm,2))); % cosine loss 
% for z=1:size(pred_Feature_norm,2) 
%     loss_feat_tmp = sum(pred_Feature_norm(:,z).*GT_Feature_norm(:,z))/...
%     ( sqrt(sum(pred_Feature_norm(:,z).*pred_Feature_norm(:,z))) * sqrt(sum(GT_Feature_norm(:,z).*GT_Feature_norm(:,z))) );
%     loss_feat(z) = loss_feat_tmp;
% end
% lossFeature = mean(loss_feat);
% temperature = 10;
% pred_Feature_sofmax_with_t = exp(pred_Feature_squ/temperature)./sum(exp(pred_Feature_squ/temperature),1);
% pred_Feature_sofmax_with_t = exp(pred_Feature_norm/temperature)./sum(exp(pred_Feature_norm/temperature),1);
% GT_Feature_sofmax_with_t = exp(GT_Feature/temperature)./sum(exp(GT_Feature/temperature),1);
% GT_Feature_sofmax_with_t = exp(GT_Feature_norm/temperature)./sum(exp(GT_Feature_norm/temperature),1);
% % lossFeature1 = crossentropy(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t); % a study approach where all the globalFeature losses are combined together
% lossFeature = crossentropy(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);
% lossFeature = mse(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);
% lossFeature = l1loss(pred_Feature_sofmax_with_t,GT_Feature_sofmax_with_t);

%
% barlow
feature_dim = size(GT_Feature,1); % size(pred_Feature_squ,1)
minibatch_size = size(GT_Feature,2); % size(pred_Feature_squ,2)
pred_Features_matrix = stripdims(pred_Feature_norm);
GT_Features_matrix = stripdims(GT_Feature_norm);
% c: correlation matrix (rows correspond to student - columns correspond to teather)
% i.e. c(i,j) is the correlation between i-th element of student's vectors with the j-th element of teacher's vectors
% toy example: student_vector: 2048x1, teacher_vector: 1x2048 -> 
% c(i,j) is the the correlation between i-th element of student's vector with the j-th element of teacher's vector
c = (pred_Features_matrix * GT_Features_matrix' ) / minibatch_size ; % dlarray
%
% % |ordinary Barlow :  BARLOW TWINS's objective function measures the cross-correlation
% c_onDiagElements = zeros(1,feature_dim);
% c_offDiagElements = c; % create a replica of c
% for j = 1: feature_dim
%     c_onDiagElements(1,j) = c(j,j); % select diagonal elements
%     c_offDiagElements(j,j) = 0; % set to zero the diagonal elements
% end
% % % c_onDiagElements = diag(c);
% % % c_offDiagElements = c(~eye(feature_dim,feature_dim));
% lamda = 0.0001; % % lamda = 5*0.0001;
% lossFeature = sum((1-c_onDiagElements).^2) + lamda*sum(sum(c_offDiagElements.^2));
% % lossFeature2 = sum((1-c_onDiagElements).^2) + lamda*sum(sum(c_offDiagElements.^2)); % a study approach where all the globalFeature losses are combined together
%
% |modified Barlow : BARLOW COLLEGUES -> BARLOW TWINS's objective function measures the cross-correlation with a relaxation sense
% maximun value per row of c : focus on different vectors' components
% search for the maximun value per row and map from subscripts (indexing by position) to linear indices for matrix c
[~ , max_positions_c ] = max(c,[],2); % find the maximun value per row (i.e. max correlation of i-th element of student's vector with all the element of teacher's vector)
max_positions_c_ind = sub2ind([feature_dim,feature_dim], (1:feature_dim)', max_positions_c); % linear indices (for matrix c)
c_max_elements = c(max_positions_c_ind); % access the maximun values
% other elements except maximun value per row of c
% set the maximun value per row (as defined above) to zero (this allows to sum the elements of new matrix c with zeros at the maximum values) 
c_other_elements = c; % create a replica of c
c_other_elements(max_positions_c_ind) = 0; % set max values to zero
% define objective loss function 
lamda = 0.0001; % % lamda = 5*0.00001;
lossFeature = sum((1-c_max_elements).^2) + lamda*sum(sum(c_other_elements.^2));
% % lossFeature3 = sum((1-c_max_elements).^2) + lamda*sum(sum(c_other_elements.^2)); % a study approach where all the globalFeature losses are combined together

% total loss (as combination of the above losses)
loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.0001*lossFeature;

% loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0*lossFeature;
% loss = 1*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3 + 0*lossFeature;
% loss = 1*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3 + 0.0001*lossFeature;
% loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.0001*lossFeature;
% loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.00001*lossFeature;
% loss = 1*lossClassification + 100*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0*lossFeature;
% t = temperature; 
% loss = 1*lossClassification + 0.1*lossNAC1 + 0*lossNAC2 + 1*lossNAC3 + (temperature*temperature)*lossFeature;
% loss = 1*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3 + 0*10*10*lossFeature;
% loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + (0.0001)*1*1*lossFeature;
% loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.001*(temperature*temperature)*lossFeature;
% loss = 1*lossClassification + 0*1*1*lossFeature;
% loss = 1*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3;
% loss = 1*lossClassification + 0.1*lossNAC1 + 0*lossNAC2 + 1*lossNAC3;
% loss = 1*lossClassification + 1*lossNAC1 + 0*lossNAC2 + 10*lossNAC3 + (0)*1*1*lossFeature;
% loss = 1*lossClassification + 0.1*lossNAC1 + 0*lossNAC2 + 1*lossNAC3 + (0)*1*1*lossFeature;
% loss = 1*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3 + 0.001*(temperature*temperature)*lossFeature;
% loss = 0*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0*lossFeature;
% loss = 0*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.0001*lossFeature;
% loss = 0*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3 + 0.0001*lossFeature;
% loss = 0*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3 + 1*lossFeature;
% % lossFeature = lossFeature1 + lossFeature2 + lossFeature3; % a study approach where all the globalFeature losses are combined together
% % loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.001*(temperature*temperature)*lossFeature1 + 0.0001*lossFeature2 + 0.0001*lossFeature3;
% % lossFeature = lossFeature1 + lossFeature3;
% % loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.001*(temperature*temperature)*lossFeature1 + 0.0001*lossFeature3;
% % lossFeature = lossFeature2 + lossFeature3; 
% % loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.0001*lossFeature2 + 0.0001*lossFeature3;


% % loss = 1*lossClassification + 0*lossNAC1 + 0*lossNAC2 + 0*lossNAC3 + 0*lossFeature;
% % if epoch == 5
% % loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0*lossFeature;    
% % elseif epoch == 19
% % % loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0*lossFeature;
% % loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.000001*lossFeature;
% % % elseif epoch == 39
% % % loss = 1*lossClassification + 10*lossNAC1 + 0*lossNAC2 + 100*lossNAC3 + 0.000001*lossFeature;
% % end


% calculate the total gradients
gradients = dlgradient(loss,dlnet.Learnables);



% Class Accuracy
% determine predicted classes
pred_class_label = onehotdecode(pred_softmax_label,(1:310),1); % [~,pred_class_label]=max(pred_softmax_label,[],1);
GT_class_label = onehotdecode(GT_label,(1:310),1); % [~,GT_class_label]=max(GT_label,[],1);
% compare predicted and GT classes
Correct_class = pred_class_label == GT_class_label;
Acc_class = mean(Correct_class);



end