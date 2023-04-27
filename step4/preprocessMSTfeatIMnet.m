function [images,labels,masks1,NACs1,masks2,NACs2,masks3,NACs3,Features] = preprocessMSTfeatIMnet(ImagesCell,LabelsCell,Masks1Cell,NACs1Cell,Masks2Cell,NACs2Cell,Masks3Cell,NACs3Cell,FeaturesCell)
% This custom function concatenates the mini-batch variables into arrays.

    % Extract image data from cell and concatenate
    images = cat(4,ImagesCell{:});
    
    % Extract label data from cell and concatenate
    labels = cat(2,LabelsCell{:});
    % One-hot encode labels
    labels = onehotencode(labels,1);
    
    % Extract mask data from cell and concatenate
    masks1 = cat(4,Masks1Cell{:});
    
    % Extract label data from cell and concatenate
    NACs1 = cat(2,NACs1Cell{:});
    
    % Extract mask data from cell and concatenate
    masks2 = cat(4,Masks2Cell{:});
   
    % Extract label data from cell and concatenate
    NACs2 = cat(2,NACs2Cell{:});
    
    % Extract mask data from cell and concatenate
    masks3 = cat(4,Masks3Cell{:});
    
    % Extract label data from cell and concatenate
    NACs3 = cat(2,NACs3Cell{:});
    
    % Extract label data from cell and concatenate
    Features = cat(2,FeaturesCell{:});

        
% |images |class labels | Masks1 | NACs1 | Masks2 | NACs2 |Masks | NACs3 | Features
 
% images: H x W x 1 x B => 4-th dim is the number of images
% GT_labels: B x 1 => 1-st dim (row-oriented data) is the number of images
% Masks: N x N x 1 x B => 4-th dim is the number of images
% GT_NAC: N x B => 2-nd dim (column-oriented data) is the number of images
% GT_Feature: F x B => 2-nd dim (column-oriented data) is the number of images

    
end