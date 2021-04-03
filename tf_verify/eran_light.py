"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import torch
import numpy as np
from eran import ERAN
from read_net_file import read_onnx_net
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import verify_network_with_milp
import argparse
from config import config
import PIL
from constraint_utils import get_constraints_for_dominant_label
import re
import itertools
#from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import spatial
from copy import deepcopy
from onnx_translator import ONNXTranslator
from optimizer import Optimizer
from analyzer import layers
from pprint import pprint
# if config.domain=='gpupoly' or config.domain=='refinegpupoly':
from refine_gpupoly import refine_gpupoly_results
from utils import parse_vnn_lib_prop, translate_output_constraints, translate_input_to_box

#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname

def generate_constraints(class_num, y):
    return [[(y,i,0)] for i in range(class_num) if i!=y]

def normalize(image, means, stds, is_nchw, input_shape):
    # normalization taken out of the network
    current_shape = image.shape
    means = means.reshape((1, -1, 1, 1)) if is_nchw else means.reshape((1, 1, 1, -1))
    stds = stds.reshape((1, -1, 1, 1)) if is_nchw else stds.reshape((1, 1, 1, -1))
    return ((image.reshape(input_shape)-means)/stds).reshape(current_shape)

def denormalize(image, means, stds, is_nchw, input_shape):
    # denormalization taken out of the network
    current_shape = image.shape
    means = means.reshape((1,-1,1,1)) if is_nchw else means.reshape((1,1,1,-1))
    stds = stds.reshape((1, -1, 1, 1)) if is_nchw else stds.reshape((1, 1, 1, -1))
    return (image.reshape(input_shape)*stds + means).reshape(current_shape)

def mnist_data_loader():
    x = []
    y = []

    mean = np.array([0])
    std = np.array([1])

    is_nchw = True
    input_shape = (1,28,28)
    with open('../data/mnist_test_full.csv','r') as csvfile:
        tests = csv.reader(csvfile, delimiter=',')
        for test in tests:
            x.append(np.array(test[1:]).reshape(input_shape).astype(np.float64)/255)
            y.append(int(test[0]))

    data_loader_test = zip(x,y)

    return data_loader_test, mean, std, is_nchw, input_shape


def cifar10_data_loader():
    x = []
    y = []

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    is_nchw = True
    input_shape = (3, 32, 32)
    with open('../data/cifar10_test_full.csv', 'r') as csvfile:
        tests = csv.reader(csvfile, delimiter=',')
        for test in tests:
            x.append(np.array(test[1:]).reshape(input_shape).as_type(float)/255)
            y.append(int(test[0]))

    data_loader_test = zip(x, y)

    return data_loader_test, mean, std, is_nchw, input_shape


class DAVE_dataset(torch.utils.data.Dataset):
    def __init__(self, idx, data_path, y, mean=None, std=None, limit_sample=None):
        super(DAVE_dataset).__init__()
        self.idx = idx[:limit_sample]
        self.data_path = data_path
        self.y = y[:limit_sample]
        self.mean = 0.0 if mean is None else mean
        self.std = 1.0 if std is None else std
        self.limit_sample = limit_sample

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        #         idx = idx if self.limit_sample is None else idx%self.limit_sample
        timestamp = self.idx[idx]
        y = self.y[idx]
        image_pil = PIL.Image.open(os.path.join(self.data_path, f"{timestamp:d}.jpg"))
        image_pil = image_pil.resize(size=(200, 66), resample=PIL.Image.BILINEAR, box=(0, 80, 640, 480), )
        image_np = np.asarray(image_pil, dtype=np.float32) / 255
        image_normalized = (image_np - self.mean) / self.std
        return image_np.transpose(2, 0, 1), y

def dave_data_loadeer():
    mean = np.array([0.27241945, 0.29117319, 0.34793583])
    std = np.array([0.23264934, 0.23466264, 0.26194483])
    timestamps = {}
    angle = {}
    input_shape = (3,66,200)
    is_nchw = True

    img_dir = "/home/mark/Projects/DAVE_SD/data/Ch2_002_out/"
    test_HMB = [4]
    for HMB_set in range(1, 7):
        timestamps_i = []
        angle_i = []
        with open(os.path.join(img_dir, f"HMB_{HMB_set:d}", "interpolated.csv"), "r") as f:
            lines = csv.reader(f, delimiter=",")
            for i, line in enumerate(lines):
                if line[4] != "center_camera":
                    continue
                if i == 0:
                    continue
                timestamps_i.append(int(line[1]))
                angle_i.append(float(line[6]) / np.pi)
        timestamps[HMB_set] = timestamps_i
        angle[HMB_set] = angle_i

    timestamps_test = []
    angle_test = []
    for HMB_set in test_HMB:
        timestamps_test += timestamps[HMB_set]
        angle_test.append(np.array(angle[HMB_set]))
    angle_test = np.concatenate(angle_test)

    dave_ds = DAVE_dataset(timestamps_test, "/home/mark/Projects/DAVE_SD/data/all", angle_test, mean=None, std=None, limit_sample=None)
    sampler = torch.utils.data.SequentialSampler(dave_ds)
    dave_dl = torch.utils.data.DataLoader(dave_ds, batch_size=1, num_workers=0, pin_memory=True, drop_last=False, sampler=sampler)
    return dave_dl, mean, std, is_nchw, input_shape


def get_data_loader():
    # return mnist_data_loader()
    # return cifar10_data_loader()
    return dave_data_loadeer()
#    return data_loader_test, mean, std, is_nchw, input_shape


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

def get_args():
    parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
    parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
    parser.add_argument('--epsilon_y', type=float, default=config.epsilon, help='maximum delta for regression tasks')
    parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly')
    parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
    parser.add_argument("--regression", type=str2bool, default=config.regression, help="Whether to verify a regression or classification task")

    parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
    parser.add_argument('--timeout_final_lp', type=float, default=config.timeout_final_lp,  help='timeout for the final LP solver')
    parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
    parser.add_argument('--timeout_final_milp', type=float, default=config.timeout_final_lp,  help='timeout for the final MILP solver')
    parser.add_argument('--timeout_complete', type=float, default=None,  help='Cumulative timeout for the complete verifier, superseeds timeout_final_milp if set')

    parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
    parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
    parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')

    # PRIMA parameters
    parser.add_argument('--k', type=int, default=config.k, help='refine group size')
    parser.add_argument('--s', type=int, default=config.s, help='refine group sparsity parameter')
    parser.add_argument("--approx_k", type=str2bool, default=config.approx_k, help="Use approximate fast k neuron constraints")
    parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')

    # Refinement parameters
    parser.add_argument('--refine_neurons', action='store_true', default=config.refine_neurons, help='whether to refine intermediate neurons')
    parser.add_argument('--n_milp_refine', type=int, default=config.n_milp_refine, help='Number of milp refined layers')
    parser.add_argument('--max_milp_neurons', type=int, default=config.max_milp_neurons,  help='Maximum number of neurons to use for partial MILP encoding.')
    parser.add_argument('--partial_milp', type=int, default=config.partial_milp,  help='Number of layers to encode using MILP')

    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config.json = vars(args)
    pprint(config.json)

    assert config.netname, 'a network has to be provided for analysis.'

    filename, file_extension = os.path.splitext(config.netname)

    is_onnx = file_extension == ".onnx"
    assert is_onnx, "file extension not supported in ERAN light"

    assert config.domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly', 'gpupoly',
                      'refinegpupoly'], "domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly"

    return config

def evaluate_net(x, domain, network=None, eran=None):
    if "gpu" in domain:
        net_out = network.eval(np.array(x))[:, 0]
        label = np.argmax(net_out)
    else:
        label, _, net_out, _, _, _ = eran.analyze_box(x, x, 'deepzono', config.timeout_lp,
                                                          config.timeout_milp,
                                                          config.use_default_heuristic,
                                                          approx_k=config.approx_k, label=0 if config.regression else -1)
        net_out = net_out[-1]
    return label, net_out


def run():
    config = get_args()

    netname = config.netname

    epsilon = config.epsilon
    epsilon_y = config.epsilon_y

    domain = config.domain

    model, is_conv = read_onnx_net(netname)
    nn = layers()

    if domain == 'gpupoly' or domain == 'refinegpupoly':
        translator = ONNXTranslator(model, True)
        operations, resources = translator.translate()
        optimizer = Optimizer(operations, resources)
        network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn)
    else:
        eran = ERAN(model, is_onnx=True)

    os.sched_setaffinity(0,cpu_affinity)

    correctly_classified_images = 0
    verified_images = 0
    unsafe_images = 0
    cum_time = 0

    test_data_loader, mean, std, is_nchw, input_shape = get_data_loader()

    for i, (x,y) in enumerate(test_data_loader):
        if config.from_test and i < config.from_test:
            continue
        if config.num_tests is not None and i >= config.from_test + config.num_tests:
            break

        image = np.float64(x).reshape(input_shape)
        image_normalized = normalize(image, mean, std, is_nchw, input_shape).reshape(-1)

        start = time.time()

        label, net_out = evaluate_net(image_normalized, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)

        if config.regression:
            is_correctly_classified = (abs(y-net_out[0]) < epsilon_y)
            label = float(y)
        else:
            is_correctly_classified = (label == y)
            label = int(y)

        ## start certification
        if is_correctly_classified:
            # Only attemp certificaiton for correctly classified samples
            perturbed_label = -1
            correctly_classified_images += 1
            failed_constraints = None
            is_verified = False

            specLB = np.clip(image - epsilon,0,1)
            specUB = np.clip(image + epsilon,0,1)
            normalize(specLB, mean, std, is_nchw, input_shape)
            normalize(specUB, mean, std, is_nchw, input_shape)
            specLB = specLB.reshape(-1)
            specUB = specUB.reshape(-1)

            prop = -1

            if domain == 'gpupoly' or domain =='refinegpupoly':
                if config.regression:
                    res = network.evalAffineExpr(np.array([1]), back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)
                    is_verified = (label-res[0][0] < config.epsilon_y) and (res[0][1]-label < config.epsilon_y)
                else:
                    is_verified = network.test(specLB, specUB, label)
                #print("res ", res)
                if not is_verified and domain == 'refinegpupoly':
                    nn.specLB = specLB
                    nn.specUB = specUB

                    perturbed_label, _, nlb, nub, failed_constraints, x = refine_gpupoly_results(nn, network, num_gpu_layers,
                                                            relu_layers, label, adv_labels=prop,
                                                            K=config.k, s=config.s,complete=config.complete,
                                                            timeout_final_lp=config.timeout_final_lp,
                                                            timeout_final_milp=config.timeout_final_milp,
                                                            timeout_lp=config.timeout_lp,
                                                            timeout_milp=config.timeout_milp,
                                                            use_milp=config.use_milp,
                                                            partial_milp=config.partial_milp,
                                                            max_milp_neurons=config.max_milp_neurons,
                                                            approx=config.approx_k)
                    if config.regression:
                        is_verified = is_verified or (abs(label - nlb[-1][0]) < epsilon_y and abs(nub[-1][0]-label)<epsilon_y)
                    else:
                        is_verified = is_verified or (perturbed_label == label)
            else:
                if domain.endswith("poly"):
                    # First do a cheap pass without PRIMA
                    perturbed_label, nn, nlb, nub, failed_constraints, x = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=prop, K=0, s=0,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=False,
                                                                                      complete=False,
                                                                                      terminate_on_failure=(not config.complete or domain != "refinepoly"),
                                                                                      partial_milp=0,
                                                                                      max_milp_neurons=0,
                                                                                      approx_k=0)
                    if config.debug:
                        print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_constraints)
                    if config.regression:
                        is_verified = is_verified or (abs(label - nlb[-1][0]) < epsilon_y and abs(nub[-1][0]-label)<epsilon_y)
                    else:
                        is_verified = is_verified or (perturbed_label == label)
                if (not is_verified) and (not domain.endswith("poly") or "refine" in domain):
                    # Do a second more precise run, for refinepoly
                    perturbed_label, nn, nlb, nub, failed_constraints, x = eran.analyze_box(specLB, specUB, domain,
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=prop,
                                                                                      K=config.k, s=config.s,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=config.use_milp,
                                                                                      complete=config.complete,
                                                                                      terminate_on_failure=not config.complete,
                                                                                      partial_milp=config.partial_milp,
                                                                                      max_milp_neurons=config.max_milp_neurons,
                                                                                      approx_k=config.approx_k)

                    if config.regression:
                        is_verified = is_verified or (abs(label - nlb[-1][0]) < epsilon_y and abs(nub[-1][0]-label)<epsilon_y)
                    else:
                        is_verified = is_verified or (perturbed_label == label)
            if is_verified:
                print("img", i, "Verified", label)
                verified_images += 1
            else:
                adex_found = False
                if x is not None:
                    for x_i in x:
                        cex_label, cex_out = evaluate_net(x_i, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
                        if config.regression:
                            adex_found = abs(y - cex_out[0]) > epsilon_y
                        else:
                            adex_found = cex_label != label
                        if adex_found:
                            if config.regression:
                                print(f"img {i} Failed, with adversarial example with output {cex_out[0]}. Correct output is {label}")
                            else:
                                print(f"img {i} Failed, with adversarial example with label {cex_label}. Correct label is {label}")
                            denormalize(np.array(x_i), mean, std, is_nchw, input_shape)
                            unsafe_images += 1
                            break
                if not adex_found:
                    if config.complete and (failed_constraints is not None):
                        is_verified, x, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub,
                                                                                   failed_constraints)
                        if is_verified:
                            print("img", i, "Verified", label, "using MILP against", failed_constraints)
                            verified_images += 1
                        else:
                            cex_label, cex_out = evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
                            if config.regression:
                                adex_found = abs(y - cex_out[0]) > epsilon_y
                            else:
                                adex_found = cex_label != label
                            if adex_found:
                                if config.regression:
                                    print(f"img {i} Failed, with adversarial example with output {cex_out[0]}. Correct output is {label}")
                                else:
                                    print(f"img {i} Failed, with adversarial example with label {cex_label}. Correct label is {label}")
                                denormalize(np.array(x), mean, std, is_nchw, input_shape)
                                unsafe_images += 1
                                break
                            else:
                                print(f"img {i} Failed, without a adversarial example")
                    else:
                        print(f"img {i} Failed")
        else:
            print(f"img {i} not considered, incorrectly classified")

        end = time.time()
        cum_time += end - start  # only count samples where we did try to certify

        print(f"progress: {1 + i - config.from_test}/{config.num_tests}, "
              f"correct:  {correctly_classified_images}/{1 + i - config.from_test}, "
              f"verified: {verified_images}/{correctly_classified_images}, "
              f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
              f"time: {end - start:.3f}; {0 if correctly_classified_images==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")



        print(f'certified images: {verified_images}; analysis precision ',verified_images + unsafe_images,'/ ', correctly_classified_images)

if __name__ == '__main__':
    run()