# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-05-10 09:04:03
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-03-21 21:48:29
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import random

import torch 
from torch.utils.data import Dataset
from ._utils import Condition_encoder, Drug_SMILES_encode, rank_genes_groups_by_cov, Drug_dose_encoder

class DrugDoseAnnDataset(Dataset):
    '''
    Dataset for loading tensors from AnnData objects.
    ''' 
    def __init__(self,
                 adata,
                 dtype='train',
                 obs_key='cov_drug',
                 comb_num=1
                 ):
        self.dtype = dtype
        self.obs_key = obs_key        
        
        
        self.dense_adata = adata
        print(self.dense_adata)

        if sparse.issparse(adata.X):
            self.dense_adata  = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True), uns=adata.uns.copy(deep=True))
        
                    
        self.drug_adata = self.dense_adata[self.dense_adata.obs['control']==0] 
         
     
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)

 
        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()


        # Encode condition strings to integer
        self.drug_type_list = self.drug_adata.obs['canonical_smiles'].to_list()
        self.dose_list = self.drug_adata.obs['pert_dose'].to_list()
        self.obs_list = self.drug_adata.obs[obs_key].to_list()
        self.encode_drug_doses = Drug_dose_encoder(self.drug_adata.obs['fingerprint_smiles'].to_list(), self.dose_list, comb_num=comb_num)

        # Convert to numpy array first to locate bad rows
        encoded_np = np.array(self.encode_drug_doses)
        bad_rows = np.isnan(encoded_np).any(axis=1)

        if bad_rows.any():
            bad_indices = np.where(bad_rows)[0]
            print(f"Found {len(bad_indices)} rows with NaN SMILES/dose encodings!")
            for idx in bad_indices[:5]: # Print first 5 offending items
                print(f"  Row {idx} | SMILES: {self.drug_adata.obs['canonical_smiles'].iloc[idx]} | Dose: {self.dose_list[idx]}")

        self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)



    def __len__(self):
        return len(self.drug_adata)

    def __getitem__(self, index):
        outputs = dict()
        outputs['x'] = self.data[index, :]

        
        # Create a high-speed lookup map if it doesn't exist yet
        if not hasattr(self, '_index_lookup_map'):
            self._index_lookup_map = {name: i for i, name in enumerate(self.dense_adata_index)}

        # Change this line from .index() to the high-speed dictionary lookup
        control_index = self._index_lookup_map[self.paired_control_index[index]] 
        

        #control_index = self.dense_adata_index.index(self.paired_control_index[index]) 

        outputs['control'] = self.dense_data[control_index,:]
        outputs['drug_dose'] = self.encode_drug_doses[index, :]
        outputs['label'] = outputs['drug_dose']

        obs_info = self.obs_list[index]
        outputs['cov_drug'] = obs_info
        
        return {'features':(outputs['control'], outputs['x']), 'label':outputs['label'], 'cov_drug': outputs['cov_drug']}


