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

import matplotlib.pyplot as plt

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
import onnx
import onnx_tf
import foolbox as fb

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
        return image_normalized.transpose(2, 0, 1), y

def dave_data_loader():
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

def dave_plot(image_normalized, image_id, mean, std, is_nchw, input_shape, y, y_nat, y_lb, y_ub, y_adv = None, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(denormalize(image_normalized, mean, std, is_nchw, input_shape).reshape(input_shape).transpose(1, 2, 0))
    ax.plot([100, 100 - np.tan(y * np.pi) * 66], [66, 0], linewidth=5, color="g")
    ax.plot([100, 100 - np.tan(y_nat * np.pi) * 66], [66, 0], linewidth=5, color="b")
    if y_adv is not None:
        ax.plot([100, 100 - np.tan(y_adv * np.pi) * 66], [66, 0], linewidth=5, color="r")
    cert_shape_x = np.array([100, 100 - np.sin(y_lb * np.pi) * 200, 100 - np.sin(y_nat * np.pi) * 200 ,100 - np.sin(y_ub * np.pi) * 200])
    cert_shape_y = np.array([66, 66 - np.cos(y_lb * np.pi)*200, 66 - np.cos(y_nat * np.pi)*200, 66 - np.cos(y_ub * np.pi)*200])
    ax.fill(cert_shape_x, cert_shape_y, closed=True, fill=True, alpha=0.3, color="b")
    ax.set_axis_off()
    ax.set_xlim([200,0])
    ax.set_ylim([66,0])
    fig.savefig(f"./DAVE_plots/Dave_img{image_id}_eps{np.round(config.epsilon*255):.0f}_{config.domain}.eps", bbox_inches=0)
    fig.savefig(f"./DAVE_plots/Dave_img{image_id}_eps{np.round(config.epsilon*255):.0f}_{config.domain}.png", bbox_inches=0)
    if show:
        plt.show()


def get_data_loader():
    # return mnist_data_loader()
    # return cifar10_data_loader()
    return dave_data_loader()
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
    parser.add_argument('--data_dilation', type=int, default=config.data_dilation, help='Use only every nth image')
    parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')

    # PRIMA parameters
    parser.add_argument('--k', type=int, default=config.k, help='refine group size')
    parser.add_argument('--s', type=int, default=config.s, help='refine group sparsity parameter')
    parser.add_argument("--approx_k", type=str2bool, default=config.approx_k, help="Use approximate fast k neuron constraints")
    parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')
    parser.add_argument("--max_gpu_batch", type=int, default=config.max_gpu_batch, help="maximum number of queries on GPU")

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


def onnx_2_tf(model_path, mean=None, std=None, bounds=(0,1)):
    onnx_model = onnx.load(model_path)  # load onnx model
    tf_rep = onnx_tf.backend.prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph("./temp_tf_model")  # export the model
    graph = tf.compat.v1.Graph()
    session = tf.compat.v1.Session(graph=graph)
    tf.compat.v1.disable_eager_execution()

    if mean is not None and std is not None:
        preprocessing = dict(mean=mean, std=mean, axis=-3)
    else:
        preprocessing = None

    with session.as_default():
        tf.saved_model.loader.load(session, export_dir="./temp_tf_model", tags=['serve'])
        images = graph.get_tensor_by_name("serving_default_input:0")
        logits = graph.get_tensor_by_name("PartitionedCall:0")
        fb_model = fb.models.TensorFlowModel(images, logits, bounds=bounds, preprocessing=preprocessing)
    return fb_model, session


def evaluate_net(x, domain, network=None, eran=None):
    if "gpu" in domain:
        net_out = network.eval(np.array(x))[:, 0]
        label = np.argmax(net_out)
    else:
        label, _, net_out, _, _, _, _ = eran.analyze_box(x, x, 'deepzono', config.timeout_lp,
                                                          config.timeout_milp,
                                                          config.use_default_heuristic,
                                                          approx_k=config.approx_k, label=0 if config.regression else -1)
        net_out = np.array(net_out[-1])
    return label, net_out


def evaluate_model_bounds(obj_lb, obj_ub, model_bounds):
    keys = model_bounds.keys()
    for key in keys:
        if key[0] == 0 and key[1] == -1:
            if model_bounds[key] is not None:
                obj_lb = np.maximum(obj_lb, model_bounds[key] + key[2])
        elif key[0] == -1 and key[1] == 0:
            if model_bounds[key] is not None:
                obj_ub = np.minimum(obj_ub, key[2] - model_bounds[key])
        else:
            assert False, f"Unexpected constraint encountered f{key}"
    return obj_lb, obj_ub


def get_regression_evaluation_function(epsilon_y, omitted_layers):
    ### Define evaluation of the output of a regression network, potentially applying a final non-linearity
    if epsilon_y == 0:
        if omitted_layers is None:
            def reg_eval(lb, ub, target):
                return np.maximum(np.square(lb[0] - target), np.square(ub[0] - target)), lb[0], ub[0]
        elif omitted_layers == ["Tanh"]:
            def reg_eval(lb, ub, target):
                return np.maximum(np.square(np.tanh(lb[0]) - target), np.square(np.tanh(ub[0]) - target)), np.tanh(
                    lb[0]), np.tanh(ub[0])
        elif omitted_layers == ["Sigmoid"]:
            def reg_eval(lb, ub, target):
                return np.maximum(np.square(1 / (1 + np.exp(-lb[0])) - target),
                                  np.square(1 / (1 + np.exp(-ub[0])) - target)), 1 / (1 + np.exp(-lb[0])), 1 / (
                                   1 + np.exp(-ub[0]))
        else:
            assert False
    else:
        if omitted_layers is None:
            def reg_eval(lb, ub, target):
                return (lb[0] > target - epsilon_y).__and__(ub[0] < target + epsilon_y), lb[0], ub[0]
        elif omitted_layers == ["Tanh"]:
            def reg_eval(lb, ub, target):
                return (np.tanh(lb[0]) > target - epsilon_y).__and__(np.tanh(ub[0]) < target + epsilon_y), np.tanh(
                    lb[0]), np.tanh(ub[0])
        elif omitted_layers == ["Sigmoid"]:
            def reg_eval(lb, ub, target):
                return (1 / (1 + np.exp(-lb[0])) > target - epsilon_y).__and__(
                    1 / (1 + np.exp(-ub[0])) < target + epsilon_y), 1 / (1 + np.exp(-lb[0])), 1 / (1 + np.exp(-ub[0]))
        else:
            assert False
    return reg_eval


def run_analysis():
    config = get_args()

    netname = config.netname
    domain = config.domain

    epsilon = config.epsilon
    epsilon_y = config.epsilon_y ### if epsilon_y is set to 0 bounds on output are computed, otherwise verification is attempted

    ### Prepare analysis, translating the model
    model, is_conv = read_onnx_net(netname)
    nn = layers()
    if domain == 'gpupoly' or domain == 'refinegpupoly':
        translator = ONNXTranslator(model, True)
        operations, resources = translator.translate()
        optimizer = Optimizer(operations, resources)
        network, relu_layers, num_gpu_layers, omitted_layers = optimizer.get_gpupoly(nn)
    else:
        eran = ERAN(model, is_onnx=True)
        omitted_layers = None

    os.sched_setaffinity(0, cpu_affinity)

    ### Initialize counters
    correctly_classified_images = 0
    verified_images = 0
    unsafe_images = 0
    mse_nat_total = []
    mse_cert_total = []
    cum_time = 0

    test_data_loader, mean, std, is_nchw, input_shape = get_data_loader()

    if config.regression:
        reg_eval = get_regression_evaluation_function(epsilon_y, omitted_layers)

    for i, (x, y) in enumerate(test_data_loader):
        if config.from_test and i < config.from_test:
            # skip samples up to config.from_test
            continue
        if i % config.data_dilation != 0:
            # evaluate only every config.data_dilation sample
            continue
        if config.num_tests is not None and i >= (config.from_test + config.num_tests*config.data_dilation):
            # stop analysis once config.num_tests samples have been evaluated
            break

        image = np.float64(x).reshape(input_shape)
        image_normalized = normalize(image, mean, std, is_nchw, input_shape).reshape(-1)
        y = np.float(y) if config.regression else np.int(y)

        obj_lb, obj_ub, mse_cert, y_adv = None, None, None, None
        start = time.time()

        # Get prediction for unperturbed sample
        label, net_out = evaluate_net(image_normalized, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)

        if config.regression:
            if epsilon_y == 0:
                mse_nat, y_nat, _ = reg_eval(net_out, net_out, y)
                is_correctly_classified = True
                label = float(y_nat) ### When computing bounds on the network prediction use natural prediction as label
            else:
                is_correctly_classified, y_nat, _ = reg_eval(net_out, net_out, y)
                label = float(y)
        else:
            is_correctly_classified = (label == y)
            label = int(y)

        ## start certification
        if is_correctly_classified:
            # Only attemp certificaiton for correctly classified samples
            correctly_classified_images += 1
            failed_constraints = None
            is_verified = False

            specLB = np.clip(image - epsilon, 0, 1)
            specUB = np.clip(image + epsilon, 0, 1)
            specLB = normalize(specLB, mean, std, is_nchw, input_shape)
            specUB = normalize(specUB, mean, std, is_nchw, input_shape)
            specLB = specLB.reshape(-1)
            specUB = specUB.reshape(-1)

            prop = -1 ### Evaluate against all adversarial classes

            if domain == 'gpupoly' or domain =='refinegpupoly':
                if config.regression:
                    res = network.evalAffineExpr_withProp(specLB, specUB, np.array([[1]], dtype=network._last_dtype), back_substitute=1)
                    if epsilon_y == 0:
                        obj_lb, obj_ub = res[0][0:1], res[0][1:2]
                        mse_cert, y_lb, y_ub = reg_eval(obj_lb, obj_ub, y)
                        is_verified = False
                    else:
                        is_verified = reg_eval(res[0][0:1], res[0][1:2], y)
                else:
                    is_verified = network.test(specLB, specUB, label)

                if not is_verified and domain == 'refinegpupoly':
                    nn.specLB = specLB
                    nn.specUB = specUB

                    perturbed_label, _, nlb, nub, failed_constraints, x, model_bounds = refine_gpupoly_results(nn, network, config,
                                                            relu_layers, label, adv_labels=prop,
                                                            K=config.k, s=config.s,complete=config.complete,
                                                            timeout_final_lp=config.timeout_final_lp,
                                                            timeout_final_milp=config.timeout_final_milp,
                                                            timeout_lp=config.timeout_lp,
                                                            timeout_milp=config.timeout_milp,
                                                            use_milp=config.use_milp,
                                                            partial_milp=config.partial_milp,
                                                            max_milp_neurons=config.max_milp_neurons,
                                                            approx=config.approx_k,
                                                            max_eqn_per_call=config.max_gpu_batch,
                                                            terminate_on_failure=(not config.complete) and not (config.regression and epsilon_y == 0))
                    if config.regression:
                        if epsilon_y == 0:
                            obj_lb, obj_ub = evaluate_model_bounds(obj_lb, obj_ub, model_bounds)
                            mse_cert_temp, y_lb, y_ub = reg_eval(obj_lb, obj_ub, y)
                            mse_cert = np.minimum(mse_cert, mse_cert_temp)
                            is_verified = False
                        else:
                            is_verified = is_verified or (failed_constraints is None)
                    else:
                        is_verified = is_verified or (perturbed_label == label)
            else:
                if domain.endswith("poly"):
                    # First do a cheap pass without multi-neuron constraints
                    perturbed_label, nn, nlb, nub, failed_constraints, x, _ = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=prop, K=0, s=0,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=False,
                                                                                      complete=False,
                                                                                      terminate_on_failure=(not config.complete or domain != "refinepoly") and not (config.regression and epsilon_y == 0),
                                                                                      partial_milp=0,
                                                                                      max_milp_neurons=0,
                                                                                      approx_k=0)
                    if config.debug:
                        print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_constraints)
                    if config.regression:
                        if epsilon_y == 0:
                            obj_lb, obj_ub = nlb[-1], nub[-1]
                            mse_cert, y_lb, y_ub = reg_eval(obj_lb, obj_ub, y)
                            is_verified = False
                        else:
                            is_verified, y_lb, y_ub = is_verified or reg_eval(nlb[-1], nub[-1], y)[0]
                    else:
                        is_verified = is_verified or (perturbed_label == label)
                if (not is_verified) and (not domain.endswith("poly") or "refine" in domain):
                    perturbed_label, nn, nlb, nub, failed_constraints, x, model_bounds = eran.analyze_box(specLB, specUB, domain,
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=prop,
                                                                                      K=config.k, s=config.s,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=config.use_milp,
                                                                                      complete=config.complete,
                                                                                      terminate_on_failure=(not config.complete) and not (config.regression and epsilon_y == 0),
                                                                                      partial_milp=config.partial_milp,
                                                                                      max_milp_neurons=config.max_milp_neurons,
                                                                                      approx_k=config.approx_k)

                    if config.regression:
                        if epsilon_y == 0:
                            obj_lb, obj_ub = nlb[-1], nub[-1]
                            if model_bounds is not None:
                                obj_lb, obj_ub = evaluate_model_bounds(obj_lb, obj_ub, model_bounds)
                            mse_cert_temp, y_lb, y_ub = reg_eval(obj_lb, obj_ub, y)
                            mse_cert = mse_cert_temp if mse_cert is None else np.minimum(mse_cert, mse_cert_temp)
                            is_verified = False
                        else:
                            is_verified = is_verified or reg_eval(nlb[-1], nub[-1], y)[0]
                    else:
                        is_verified = is_verified or (perturbed_label == label)
            if is_verified:
                print("img", i, "Verified", label)
                verified_images += 1
            elif config.regression and epsilon_y == 0:
                print(f"img {i}: y: {y:.4f}; y_nat: {y_nat:.4f}; y_lb/ub: {y_lb:.4f}/{y_ub:.4f} natural MSE: {mse_nat:.4e} and certified MSE: {mse_cert:.4e}")
            else:
                adex_found = False
                if x is not None:
                    for x_i in x:
                        # Check if returned solution is an adversarial example
                        cex_label, cex_out = evaluate_net(x_i, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
                        if config.regression:
                            adex_found, y_adv, _ = reg_eval(cex_out, cex_out, y)
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
                        # Run complete verification on uncertified constraints
                        is_verified, x, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, failed_constraints)
                        if is_verified:
                            print("img", i, "Verified", label, "using MILP against", failed_constraints)
                            verified_images += 1
                        else:
                            # Check if returned feasible solution is an adversarial example
                            cex_label, cex_out = evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
                            if config.regression:
                                adex_found, y_adv, _ = reg_eval(cex_out, cex_out, y)
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
            end = time.time()
            cum_time += end - start  # only count samples where we did try to certify
        else:
            end = time.time()
            print(f"img {i} not considered, incorrectly classified")



        dave_plot(image_normalized, i, mean, std, is_nchw, input_shape, y, y_nat, y_lb, y_ub, y_adv=None, show=False)

        if config.regression and epsilon_y == 0:
            mse_nat_total.append(mse_nat)
            mse_cert_total.append(mse_cert)
            print(f"progress: {1 + i/config.data_dilation - config.from_test:.0f}/{config.num_tests}, "
                  f"nat MSE:  {np.median(mse_nat_total):.4e}/{np.mean(mse_nat_total):.4e}, "
                  f"cert MSE: {np.median(mse_cert_total):.4e}/{np.mean(mse_cert_total):.4e}, "
                  f"time: {end - start:.3f}; {0 if correctly_classified_images==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")

        else:
            print(f"progress: {1 + i/config.data_dilation - config.from_test:d}/{config.num_tests}, "
                  f"correct:  {correctly_classified_images}/{1 + i/config.data_dilation - config.from_test}, "
                  f"verified: {verified_images}/{correctly_classified_images}, "
                  f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
                  f"time: {end - start:.3f}; {0 if correctly_classified_images==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")



    print(f'certified images: {verified_images}; analysis precision ',verified_images + unsafe_images,'/ ', correctly_classified_images)

if __name__ == '__main__':
    run_analysis()