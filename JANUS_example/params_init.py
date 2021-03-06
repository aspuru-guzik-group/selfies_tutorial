#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:46:02 2021

@author: akshat
"""


def generate_params(): 
    '''
    Parameters for initiating JANUS. The parameters here are picked based on prior 
    experience by the authors of the paper. 
    '''
    
    params_ = {}
    
    # Number of iterations that JANUS runs for: 
    params_['generations']        = 2
    
    # The number of molecules for which fitness calculations are done, within each generation
    params_['generation_size']    = 5000
    
    # Location of file containing SMILES that will be user for the initial population. 
    # NOTE: number of smiles must be greater than generation size. 
    params_['start_population']   = './DATA/sample_start_smiles.txt'
    
    # Number of molecules that are exchanged between the exploration and exploitation 
    # componenets of JANUS. 
    params_['num_exchanges']      = 5
    
    # An option to generate fragments and use then when performing mutations. 
    # Fragmenets are generated using the SMILES provided for the starting population. 
    # The list of generated fragments is stored in './DATA/fragments_selfies.txt'
    params_['use_fragments']      = True # Set to true
    
    # An option to use a classifier for sampling. If set to true, the trailed model 
    # is saved at the end of every generation in './RESULTS/'. 
    params_['use_NN_classifier']  = False # Set this to true! 
    
    return params_




from rdkit import Chem
from rdkit.Chem import Descriptors

def calc_prop(smi): 
    '''
    Given a SMILES string (smi), a user needs to provide code for calculating a 
    property value of interest. This function is used throughout JANUS for obtaining the 
    fitness values. As a dummy value, the function return 1.0 for a give SMILES. 
    
    NOTE: 
        If the objective is to minimize the property value, please add a minus sign to 
        the property value. 

    Parameters
    ----------
    smi : str
        Valid SMILE string generated by JANUS.

    Returns
    -------
    float
        Property value of SMILES string 'smi'.

    '''
    
    # if rdcmd.CalcNumBridgeheadAtoms(mol) == 0 and rdcmd.CalcNumSpiroAtoms(mol) == 0 and aromaticity_degree(mol) >= 0.5 and conjugation_degree(mol) >= 0.7 and (5 <= maximum_ring_size(mol) <=7) and (5 <= minimum_ring_size(mol) <=7) and substructure_violations(mol)==False and mol_hydrogen.GetNumAtoms()<=70: 

    
    return Descriptors.MolLogP(Chem.MolFromSmiles(smi))
