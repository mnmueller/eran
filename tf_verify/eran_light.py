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
from ai_milp import
import argparse
from config import config
from constraint_utils import get_constraints_for_dominant_label
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import spatial
from copy import deepcopy
from tensorflow_translator import
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

# def parse_input_box(text):
#     intervals_list = []
#     for line in text.split('\n'):
#         if line!="":
#             interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
#             intervals = []
#             for interval in interval_strings:
#                 interval = interval.replace('[', '')
#                 interval = interval.replace(']', '')
#                 [lb,ub] = interval.split(",")
#                 intervals.append((np.double(lb), np.double(ub)))
#             intervals_list.append(intervals)
#
#     # return every combination
#     boxes = itertools.product(*intervals_list)
#     return list(boxes)


# def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
#     print('==================================================================')
#     for i in range(n_rows):
#         print('  ', end='')
#         for j in range(n_cols):
#             print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
#         print('  |  ', end='')
#         for j in range(n_cols):
#             print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
#         print('  |  ')
#     print('==================================================================')


def normalize(image, means, stds, is_nchw):
    # normalization taken out of the network
    means = means.reshape((1,-1,1,1)) if is_nchw else means.reshape((1,1,1,-1))
    stds = stds.reshape((1, -1, 1, 1)) if is_nchw else stds.reshape((1, 1, 1, -1))
    return (image-means)/stds

def denormalize(image, means, stds, is_nchw):
    # denormalization taken out of the network
    means = means.reshape((1,-1,1,1)) if is_nchw else means.reshape((1,1,1,-1))
    stds = stds.reshape((1, -1, 1, 1)) if is_nchw else stds.reshape((1, 1, 1, -1))
    return image*stds+means

def model_predict(base, input):
    if is_onnx:
        pred = base.run(input)
    else:
        pred = base.run(base.graph.get_operation_by_name(model.op.name).outputs[0], {base.graph.get_operations()[0].name + ':0': input})
    return pred


def get_data_loader():
    return data_loader_test, mean, std, is_nchw


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
parser.add_argument('--epsilon_y', type=float, default=config.epsilon, help='maximum delta for regression tasks')
parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly')
parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')

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

netname = config.netname
filename, file_extension = os.path.splitext(netname)

is_onnx = file_extension == ".onnx"
assert is_onnx, "file extension not supported in ERAN light"

epsilon = config.epsilon
epsilon_y = config.epsilon_y

domain = config.domain
assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly', 'gpupoly', 'refinegpupoly'], "domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly"

complete = config.complete==True

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

model, is_conv = read_onnx_net(netname)
if domain == 'gpupoly' or domain == 'refinegpupoly':
    translator = ONNXTranslator(model, True)
    operations, resources = translator.translate()
    optimizer = Optimizer(operations, resources)
    nn = layers()
    network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn)
else:
    eran = ERAN(model, is_onnx=is_onnx)

is_trained_with_pytorch = True

os.sched_setaffinity(0,cpu_affinity)

correctly_classified_images = 0
verified_images = 0
unsafe_images = 0
cum_time = 0

test_data_loader, mean, std, is_nchw = get_data_loader()


target = []

for i, (x,y) in enumerate(test_data_loader):
    if config.from_test and i < config.from_test:
        continue
    if config.num_tests is not None and i >= config.from_test + config.num_tests:
        break

    image= np.float64(x)
    specLB = np.copy(image)
    specUB = np.copy(image)

    normalize(specLB, mean, std, is_nchw)
    normalize(specUB, mean, std, is_nchw)

    # specUB = specUB.reshape(-1)
    # specLB = specLB.reshape(-1)
    is_correctly_classified = False
    start = time.time()
    if domain == 'gpupoly' or domain == 'refinegpupoly':
        is_correctly_classified = network.test(specLB, specUB, y, True)
    else:
        label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
        print("concrete ", nlb[-1])
        if config.regression:
            if abs(y-nlb[-1])<epsilon_y:
                is_correctly_classified = True
        else:
            if label == int(y):
                is_correctly_classified = True

    if is_correctly_classified == True:
        perturbed_label = None
        correctly_classified_images += 1
        specLB = np.clip(image - epsilon,0,1)
        specUB = np.clip(image + epsilon,0,1)
        normalize(specLB, mean, std, is_nchw)
        normalize(specUB, mean, std, is_nchw)

        prop = -1

        if domain == 'gpupoly' or domain =='refinegpupoly':
            num_outputs = len(nn.weights[-1])

            if config.regression:
                res = network.evalAffineExpr(np.array([1]), back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)
                is_verified = (y-res[0][0] < config.epsilon_y) and (res[0][1]-y < config.epsilon_y)
            else:
                is_verified = network.test(specLB, specUB, y)
            #print("res ", res)
            if is_verified:
                print("img", i, "Verified", y)
                verified_images+=1
            elif domain == 'refinegpupoly':
                nn.specLB = specLB
                nn.specUB = specUB
                nn.predecessors = []
                constraints = []
                for pred in range(0, nn.numlayer+1):
                    predecessor = np.zeros(1, dtype=np.int)
                    predecessor[0] = int(pred-1)
                    nn.predecessors.append(predecessor)


                if config.regression:
                    constraints.append([(0,-1,y-epsilon_y)])
                    constraints.append([(-1, 0, y + epsilon_y)])
                else:
                    num_outputs = len(nn.weights[-1])
                    # Matrix that computes the difference with the expected layer.
                    diffMatrix = np.delete(-np.eye(num_outputs), int(y), 0)
                    diffMatrix[:, label] = 1
                    diffMatrix = diffMatrix.astype(np.float64)

                    # gets the values from GPUPoly.
                    res = network.evalAffineExpr(diffMatrix, back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)

                    var = 0
                    # constraints += get_constraints_for_dominant_label(int(y), range(num_outputs))
                    for labels in range(num_outputs):
                        if labels != int(y):
                            if res[var][0] < 0:
                                constraints.append([(int(y),int(labels),0)])
                            var = var+1

                is_verified, adex = refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, int(y),
                                                        K=config.k, s=config.s,
                                                        complete=config.complete,
                                                        timeout_final_lp=config.timeout_final_lp,
                                                        timeout_final_milp=config.timeout_final_milp,
                                                        timeout_lp=config.timeout_lp,
                                                        timeout_milp=config.timeout_milp,
                                                        use_milp=config.use_milp,
                                                        partial_milp=config.partial_milp,
                                                        max_milp_neurons=config.max_milp_neurons,
                                                        approx=config.approx_k, constraints=constraints)
                if is_verified:
                    print("img", i, "Verified", y)
                    verified_images += 1
                else:
                    if adex != None:
                        adv_image = np.array(adex)

                        if config.regression:
                            adex_found = abs(y-(network.eval(adv_image)))>epsilon_y
                        else:
                            adex_found = np.argmax((network.eval(adv_image))[:, 0]) != int(y)
                        if adex_found:
                            denormalize(adex, mean, std, is_nchw)
                            # print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", cex_label, "correct label ", int(test[0]))
                            print("img", i, "Verified unsafe against label ", res, "correct label ", y)
                            unsafe_images += 1

                        else:
                            print("img", i, "Failed")
                    else:
                        print("img", i, "Failed")
            else:
                print("img", i, "Failed")
        else:
            if domain.endswith("poly"):
                perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                  config.timeout_lp,
                                                                                  config.timeout_milp,
                                                                                  config.use_default_heuristic,
                                                                                  label=label, prop=prop, K=0, s=0,
                                                                                  timeout_final_lp=config.timeout_final_lp,
                                                                                  timeout_final_milp=config.timeout_final_milp,
                                                                                  use_milp=False,
                                                                                  complete=False,
                                                                                  terminate_on_failure = True,
                                                                                  partial_milp=0,
                                                                                  max_milp_neurons=0,
                                                                                  approx_k=0)
                print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
            if not domain.endswith("poly") or not (perturbed_label==label):
                perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, domain,
                                                                                  config.timeout_lp,
                                                                                  config.timeout_milp,
                                                                                  config.use_default_heuristic,
                                                                                  label=label, prop=prop,
                                                                                  K=config.k, s=config.s,
                                                                                  timeout_final_lp=config.timeout_final_lp,
                                                                                  timeout_final_milp=config.timeout_final_milp,
                                                                                  use_milp=config.use_milp,
                                                                                  complete=config.complete,
                                                                                  terminate_on_failure=not config.complete and domain == "refinepoly",
                                                                                  partial_milp=config.partial_milp,
                                                                                  max_milp_neurons=config.max_milp_neurons,
                                                                                  approx_k=config.approx_k)
                print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_labels)
            if (perturbed_label==label):
                print("img", i, "Verified", label)
                verified_images += 1
            else:
                if complete==True and failed_labels is not None:
                    failed_labels = list(set(failed_labels))
                    constraints = get_constraints_for_dominant_label(label, failed_labels)
                    verified_flag, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                    if(verified_flag==True):
                        print("img", i, "Verified as Safe using MILP", label)
                        verified_images += 1
                    else:
                        if adv_image != None:
                            cex_label,_,_,_,_,_ = eran.analyze_box(adv_image[0], adv_image[0], 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic, approx_k=config.approx_k)
                            if(cex_label!=label):
                                denormalize(adv_image[0], means, stds, dataset)
                                # print("img", i, "Verified unsafe with adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                                print("img", i, "Verified unsafe against label ", cex_label, "correct label ", label)
                                unsafe_images+=1
                            else:
                                print("img", i, "Failed with MILP, without a adeversarial example")
                        else:
                            print("img", i, "Failed with MILP")
                else:

                    if x != None:
                        cex_label,_,_,_,_,_ = eran.analyze_box(x,x,'deepzono',config.timeout_lp, config.timeout_milp, config.use_default_heuristic, approx_k=config.approx_k)
                        print("cex label ", cex_label, "label ", label)
                        if(cex_label!=label):
                            denormalize(x,means, stds, dataset)
                            # print("img", i, "Verified unsafe with adversarial image ", x, "cex label ", cex_label, "correct label ", label)
                            print("img", i, "Verified unsafe against label ", cex_label, "correct label ", label)
                            unsafe_images += 1
                        else:
                            print("img", i, "Failed, without a adversarial example")
                    else:
                        print("img", i, "Failed")

        end = time.time()
        cum_time += end - start # only count samples where we did try to certify
    else:
        print("img",i,"not considered, incorrectly classified")
        end = time.time()

    print(f"progress: {1 + i - config.from_test}/{config.num_tests}, "
          f"correct:  {correctly_classified_images}/{1 + i - config.from_test}, "
          f"verified: {verified_images}/{correctly_classified_images}, "
          f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
          f"time: {end - start:.3f}; {0 if cum_time==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")



print('analysis precision ',verified_images,'/ ', correctly_classified_images)
