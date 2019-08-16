import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import math
from argparse import ArgumentParser
from torch.distributions import normal
from torch import autograd
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import csv

from Person_val import Person_val
from utils.re_ranking import re_ranking_npz
from utils.propfunc import ccpp, softmax
# from submission_example.eval import eval

def getIndices(result):
    
    indices = []
    for i in range(result.shape[0]):
        s = result[i]
        idx = np.argsort(s)[::-1]

        indices.append(idx)
    return indices


def run_ccpp(ct_affmat, tt_affmat, gpu_id=-1):
    n_cast, n_instance = ct_affmat.shape
    n_sample = n_cast + n_instance
    W = np.zeros((n_sample, n_sample))
    W[:n_cast, n_cast:] = ct_affmat
    W[n_cast:, :n_cast] = ct_affmat.T
    W[n_cast:, n_cast:] = tt_affmat
    Y0 = np.zeros((n_sample, n_cast))
    for i in range(n_cast):
        Y0[i, i] = 1
    if gpu_id < 0:
        result = ccpp(W, Y0, init_fratio=0.3, steps=2, temperature=0.03)
       # result = ccpp(W, Y0
    else:
        result = gpu_ccpp(W, Y0, gpu_id=gpu_id)
    return result

def get_face_aff_ct(fname, n_cast, is_all = False):
    original_aff = np.load(fname)
    n_candidate = original_aff.shape[0] - n_cast
    face_aff = original_aff.copy()
    
    if is_all == False:
        _, ct_dist = re_ranking_npz( (1-original_aff), n_cast, n_candidate, k1=55, k2=8, lambda_value=0.2)
        original_aff = 1-ct_dist
    return original_aff, face_aff

def get_body_aff_tt(fname, n_cast):
    original_aff = np.load(fname)
    n_candidate = original_aff.shape[0] - n_cast
    
    body_aff = original_aff.copy()

    return body_aff[n_cast:, n_cast:], body_aff

def force_1_label(result):
    
    n_cast = result.shape[0]
    n_candidate = result.shape[1]
    
    result = result.copy()
    np.set_printoptions(precision=3, suppress=True)
    
    result_max = np.max(result, axis=1)
    
    
    result_max = np.expand_dims(result_max, axis=0)
    result_max = np.repeat(result_max, n_candidate,  axis=0).T
    result = np.divide(result, result_max)
    result_active = softmax(result.T, T=0.3).T
    
    result_final = np.multiply(result, result_active)
   # error
    return result_final

def make_no_connected_graph(no_connected_dict, tt_affmat, cand_names):
    
    make_cand_name_table = []
    for i, cand_name in enumerate(cand_names):
        cand_name = cand_name[0]
        make_cand_name_table.append(cand_name)
    
    for idx, key in enumerate(make_cand_name_table):
        if len(no_connected_dict[key]) > 0:
            for no_connected_key in no_connected_dict[key]:
                no_connected_idx = make_cand_name_table.index(no_connected_key)
                tt_affmat[idx, no_connected_idx] = -1.

    return tt_affmat

def test(fname_out, npz_face_list, npz_body_list):
    
    print("There are ", len(npz_face_list), "answer to ensemble!")
    
    for i in range( len(npz_face_list)):
        npz_face_list[i] = npz_face_list[i] + "/"
        
    for i in range(len(npz_body_list)):
        npz_body_list[i] = npz_body_list[i] + "/"

    is_all = False
    with open(fname_out, 'w') as txtfile:
        for batch_idx, (cast_names, candidate_names) in enumerate(validset_loader):
         
            print(batch_idx)
           
            n_cast = len(cast_names)
            n_candidate = len(candidate_names)
            print(n_cast, n_candidate)
            
            ct_affmat = []
            face_affmat = []
            for i in range(0, len(npz_face_list) ):
                
                fname = npz_face_list[i] + str(batch_idx) +  '/similarity.npy'
                original_ct_affmat, original_face_affmat = get_face_aff_ct(fname, n_cast, is_all)
                ct_affmat.append(original_ct_affmat)
                face_affmat.append(original_face_affmat)

            ct_affmat = np.array(ct_affmat)
            ct_affmat = ct_affmat.mean(axis=0)
            
            face_affmat = np.array(face_affmat)
            face_affmat = face_affmat.mean(axis=0)
             
                
            if is_all == True:
                _, ct_dist = re_ranking_npz( (1-ct_affmat), n_cast, n_candidate, k1=55, k2=8, lambda_value=0.2)
                ct_affmat = 1-ct_dist
            
            tt_afftmat = []
            body_affmat = []
            for i in range(0, len(npz_body_list) ):
                
                fname = npz_body_list[i] + str(batch_idx) +  '/similarity.npy'
                original_tt_affmat, original_body_affmat = get_body_aff_tt(fname, n_cast)
                tt_afftmat.append(original_tt_affmat)
                body_affmat.append(original_body_affmat)
            
            tt_afftmat = np.array(tt_afftmat)
            tt_afftmat = tt_afftmat.mean(axis=0)
            body_affmat = np.array(body_affmat)
            body_affmat = body_affmat.mean(axis=0)
            
            tt_afftmat = make_no_connected_graph(no_connected_graph, tt_afftmat, candidate_names)
            
            result = run_ccpp(ct_affmat, tt_afftmat)
            result = force_1_label(result)
            result = run_ccpp(result, tt_afftmat)
            result = force_1_label(result)
            indices = getIndices(result)
            
            str_ans = ""
            # for all cast
            cast_names = np.array(cast_names)

            for i in range( len(indices) ):
                
                str_ans = ""
                candidates = np.array(candidate_names.copy())
                fname_rank = []
                cand_sort_idx = indices[i].copy()
                
                candidates = candidates[cand_sort_idx]
                
                for j in range(len(candidates)):
                    base = os.path.basename(candidates[j][0])
                    no_ext = os.path.splitext(base)[0]
                    fname_rank.append(no_ext)
                    if j+1 == len(candidates):
                        str_ans = str_ans + no_ext
                    else:
                        str_ans = str_ans + no_ext +","
                cast_base = os.path.basename(cast_names[i][0])
                cast_no_ext = os.path.splitext(cast_base)[0]
                txtfile.write(cast_no_ext + ' ' + str_ans + '\n')

    print('Write answer to ', fname_out, '. Done!!')
    
    # if 'val' in fname_out:
    #     mAp = eval(fname_out, './submission_example/val_label.json')
    #     return mAp

parser = ArgumentParser()
parser.add_argument("--test_dir", help="test directory")
parser.add_argument("--output_name", help="output_name")
args = parser.parse_args()

output_name = args.output_name
test_dir = args.test_dir

no_connected_graph_file = './test_dlcv_format_no_connected_graph.npy'
no_connected_graph = np.load(no_connected_graph_file).item()

print(no_connected_graph_file)
validset = Person_val(data_dir=test_dir)
validset_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=1)

npz_face_list = [
    './ensemble_face_output/test_re_crop_face_fusion_model_irse_120/',
    './ensemble_face_output/test_re_crop_face_fusion_model_irse_63/',
    './ensemble_face_output/test_re_crop_face_fusion_model_irse_asia/',
    './ensemble_face_output/test_re_crop_face_fusion_resnet50_ft_dag/',
    './ensemble_face_output/test_re_crop_face_fusion_resnet50_scratch_dag/',
    './ensemble_face_output/test_re_crop_face_fusion_senet50_ft_dag/',
    './ensemble_face_output/test_re_crop_face_fusion_senet50_scratch_dag/',
]
npz_body_list = [
    './ensemble_output/test_se_resnext101_best/',
    './ensemble_output/test_se_resnext50_best/',
    './ensemble_output/test_se_resnext50_cand_to_cand_best/',
    './ensemble_output/test_se_resnext50_ranking_loss_best/',
    './ensemble_output/test_se_resnext50_ranking_loss_clustering/',
    './ensemble_output/test_se_resnext50_allset_cand_to_cand_best/',
    './ensemble_output/test_se_resnext50_ranking_loss_triplet_allset_best/',
    './ensemble_output/test_se_resnext50face244_best/',
]

test(output_name, npz_face_list, npz_body_list)






