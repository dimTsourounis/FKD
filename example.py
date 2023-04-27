""" This example shows how to extract features for a new signature,
    using ONNX CNN.

"""
import torch
import onnx
import onnxruntime
import numpy as np

from skimage.io import imread
from skimage import img_as_ubyte

#from sigver_WD.sigver.preprocessing.normalize import preprocess_signature

canvas_size = (952, 1360)  # Maximum signature size

# If GPU is available, use it:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

# Load and pre-process the signature
#original = img_as_ubyte(imread('/sigver_WD/data/some_signature.png', as_gray=True))
#processed = preprocess_signature(original, canvas_size)
processed = img_as_ubyte(imread('/sigver_WD/data/some_signature_processed.png', as_gray=True))
'''img = img_as_ubyte(imread('/sigver_WD/data/text_img_example.tif', as_gray=True))
img_shape = img.shape
shape = (150, 220)
start_y = (img_shape[0] - shape[0]) // 2
start_x = (img_shape[1] - shape[1]) // 2
cropped = img[start_y: start_y + shape[0], start_x:start_x + shape[1]]
processed = cropped
'''
# Note: the image needs to be a pytorch tensor with pixels in the range [0, 1]
input = torch.from_numpy(processed).view(1, 1, 150, 220)
input = input.float().div(255).to(device)

# Load the model (onnx)
model_path = '/models/onnxModels/feature_extractor_ResNet18_CL_KD_GEOM_BC.onnx'
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# Extract features
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None, ort_inputs)
f = np.array(ort_outs) # numpy 1x1x2048
feat = np.squeeze(f,1) # numpy 1x2048
features = torch.from_numpy(feat) # tensor 1x2048

features = features.cpu()[0]
print('Feature vector size:', len(features))
print('Feature vector:', features)


