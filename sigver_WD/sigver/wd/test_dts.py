import torch
from sigver.featurelearning.data import extract_features
#import sigver.featurelearning.models as models
import argparse
from sigver.datasets.util import load_dataset, get_subset
import sigver.wd.training as training
import numpy as np
import pickle
import onnx
import onnxruntime

"""
#Hafemann code - SigNet (.pth) evaluation
def main(args):
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    assert len(
        set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'

    state_dict, class_weights, forg_weights = torch.load(args.model_path,
                                                                 map_location=lambda storage, loc: storage)
    
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    base_model = models.available_models[args.model]().to(device).eval()
    base_model.load_state_dict(state_dict)  

    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)
"""

#"""
def main(args):
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    assert len(
        set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'

    #state_dict, class_weights, forg_weights = torch.load(args.model_path,
    #                                                            map_location=lambda storage, loc: storage)
    
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    #base_model = models.available_models[args.model]().to(device).eval()
    #base_model.load_state_dict(state_dict)
    #base_model = torch.load(args.model_path)
    #base_model.to(device)
    #base_model.eval()
    #
    #model_name = 'dlnet_feature13.onnx'
    model_name = args.model_path
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model_name, providers=EP_list)

    def process_fn(batch):
        input = batch[0].to(device)
        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
        ort_outs = ort_session.run(None, ort_inputs)
        f = np.array(ort_outs) # numpy 1x1x2048
        feat = np.squeeze(f,1) # numpy 1x2048
        feats = torch.from_numpy(feat) # tensor 1x2048
        #return base_model(input)
        return feats
  

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#"""
    
    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)

    features = extract_features(x, process_fn, args.batch_size, args.input_size)

    data = (features, y, yforg)

    exp_set = get_subset(data, exp_users)
    dev_set = get_subset(data, dev_users)

    rng = np.random.RandomState(1234)

    eer_u_list = []
    eer_list = []
    all_results = []
    """
    for _ in range(args.folds):
        classifiers, results = training.train_test_all_users(exp_set,
                                                             dev_set,
                                                             svm_type=args.svm_type,
                                                             C=args.svm_c,
                                                             gamma=args.svm_gamma,
                                                             num_gen_train=args.gen_for_train,
                                                             num_forg_from_exp=args.forg_from_exp,
                                                             num_forg_from_dev=args.forg_from_dev,
                                                             num_gen_test=args.gen_for_test,
                                                             rng=rng)
        this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)
    print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))
    """
    for _ in range(args.folds):
        classifiers, results = training.train_test_all_users_dts(exp_set,
                                                             dev_set,
                                                             svm_type=args.svm_type,
                                                             C=args.svm_c,
                                                             gamma=args.svm_gamma,
                                                             num_gen_train=args.gen_for_train,
                                                             num_forg_from_exp=args.forg_from_exp,
                                                             num_forg_from_dev=args.forg_from_dev,
                                                             num_gen_test=args.gen_for_test,
                                                             num_SF_test=args.SF_for_test,
                                                             rng=rng)
        this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)
    print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))



    if args.save_path is not None:
        print('Saving results to {}'.format(args.save_path))
        with open(args.save_path, 'wb') as f:
            pickle.dump(all_results, f)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-m', choices=models.available_models, required=True,
    #                    help='Model architecture', dest='model')
    #parser = argparse.ArgumentParser(description='wd classifiers')

    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--save-path')
    parser.add_argument('--input-size', nargs=2, default=(150, 220))

    parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 300))
    parser.add_argument('--dev-users', type=int, nargs=2, default=(300, 881))

    parser.add_argument('--gen-for-train', type=int, default=12)
    #parser.add_argument('--gen-for-train', type=int, default=10)
    parser.add_argument('--gen-for-test', type=int, default=10)
    parser.add_argument('--SF-for-test', type=int, default=10)
    #parser.add_argument('--forg-from_exp', type=int, default=0)
    #parser.add_argument('--forg-from_dev', type=int, default=14)
    parser.add_argument('--forg-from_exp', type=int, default=2)
    parser.add_argument('--forg-from_dev', type=int, default=0)

    parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
    parser.add_argument('--svm-c', type=float, default=1)
    parser.add_argument('--svm-gamma', type=float, default=2**-11)

    parser.add_argument('--gpu-idx', type=int, default=0)
    #parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--folds', type=int, default=10)

    arguments = parser.parse_args()
    print(arguments)

    main(arguments)
