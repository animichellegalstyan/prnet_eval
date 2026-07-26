# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-06-13 09:47:44
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-11-04 15:56:30

import os
import sys

# Adjust the root dir to be priority import root
SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(sys.path)

import anndata as ad
import argparse 
from datetime import datetime
import pandas as pd
import scanpy as sc
import numpy as np
import torch 
from trainer.PRnetTrainer import PRnetTrainer


print("Is CUDA available?", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

def parse_args():
    parse = argparse.ArgumentParser(description='perturbation-conditioned generative model') 

    parse.add_argument('--input_data', required=True, type=str, help='Path to input h5ad dataset') 
    parse.add_argument('--split_key', default='canonical_smiles_split_0', type=str, help='split key of data') 
    parse.add_argument('--smoke_test', action='store_true', help='Run a fast pipeline test with minimal data') # to test workflow without training
    parse.add_argument('--delete_fp_duplicates', default=False, help='Set to true if wanting to delete false duplicate fingerprints.')

    args = parse.parse_args()  
    return args

if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()

    epochs = 2 if args_train.smoke_test else 500
    save_directory = './checkpoint_smoke_test/' if args_train.smoke_test else './checkpoint/'

    print("Split Key: ", args_train.split_key)
    config_kwargs = {
        'batch_size' : 4000, # default was 512
        'comb_num' : 1,
        'n_epochs' : epochs,   # default was 500
        'split_key' : args_train.split_key,
        'x_dimension' : 978,
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
        #'drug_dimension': 1031,
        'drug_dimension': 1024,
        'dr_rate' : 0.05,
        'lr' : 1e-3,  
        'weight_decay' : 1e-8,
        'scheduler_factor' : 0.5,
        'scheduler_patience' : 10,
        'n_genes' : 20,
        'loss' : ['GUSS'], 
        'obs_key' : 'cov_drug_name'
    }  

    print(os.getcwd())

    adata = sc.read(args_train.input_data)

    import scipy.sparse as sp

    if sp.issparse(adata.X):
        print("Converting sparse matrix to dense array for speed...")
        adata.X = adata.X.toarray()

    adata.X = adata.X.astype('float32')

    print(f"Shape before empty row filtering: {adata.shape}")
    sc.pp.filter_cells(adata, min_counts=0.00001)
    print(f"Shape after filtering out empty rows: {adata.shape}")

    adata.X = np.clip(adata.X, a_min=0, a_max=None)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    print("HAS NANS IN ADATA.X:", np.isnan(adata.X).any())
    print("MIN VALUES IN ADATA.X AFTER:", adata.X.min())

    # current_split_key = f"{args_train.split_key}_split_{split}"

    # current_save_dir = os.path.join(save_directory, f"{args_train.split_key}")
    # os.makedirs(current_save_dir, exist_ok=True)


    print(f" STARTING TRAINING FOR FOLD {args_train.split_key}")

    # Ensure current_dir ends with os.sep so string concatenation in PRnetTrainer doesn't mangle folder names
    current_dir = os.path.abspath(os.getcwd()) + os.sep
    Trainer = PRnetTrainer(
                            adata,
                            batch_size=config_kwargs['batch_size'],
                            comb_num=config_kwargs['comb_num'],
                            split_key=args_train.split_key,
                            model_save_dir=current_dir,
                            x_dimension=config_kwargs['x_dimension'],
                            hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
                            z_dimension=config_kwargs['z_dimension'],
                            adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
                            comb_dimension=config_kwargs['comb_dimension'],
                            drug_dimension=config_kwargs['drug_dimension'],
                            dr_rate=config_kwargs['dr_rate'],
                            n_genes=config_kwargs['n_genes'],
                            loss = config_kwargs['loss'],
                            obs_key = config_kwargs['obs_key']
                                )

    Trainer.train(
        n_epochs = config_kwargs['n_epochs'],
        lr = config_kwargs['lr'], 
        weight_decay= config_kwargs['weight_decay'], 
        scheduler_factor=config_kwargs['scheduler_factor'],
        scheduler_patience=config_kwargs['scheduler_patience'])
        
    print(f"FINISHED TRAINING FOR FOLD {args_train.split_key}\n")

    # Duration stats
    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    print(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
