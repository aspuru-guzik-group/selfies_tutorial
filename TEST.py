import time
from mutate_mols import get_mutated_mols, get_fp_scores
from rdkit import Chem 
from rdkit.Chem import Draw 
    

smi     = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'   # Celecoxib
fp_type = 'ECFP4'

num_random_samples = 1000     
num_mutation_ls    = [1, 2, 3, 4, 5]


canon_smi_ls = get_mutated_mols(smi, num_random_samples, num_mutation_ls, fp_type)


start_time = time.time()
canon_smi_ls_scores = get_fp_scores(canon_smi_ls, target_smi=smi, fp_type=fp_type)
print('Fingerprint calculation time: ', time.time()-start_time)

# Molecules with fingerprint similarity > 0.8
indices_thresh_8 = [i for i,x in enumerate(canon_smi_ls_scores) if x > 0.8]
mols_8 = [Chem.MolFromSmiles(canon_smi_ls[idx]) for idx in indices_thresh_8]

img = Draw.MolsToGridImage(mols_8[:8],molsPerRow=4,subImgSize=(200,200))    
img



######### SYBA FILTRATION: 
import numpy as np 
from syba.syba import SybaClassifier

syba = SybaClassifier()
syba.fitDefaultScore()

syba_scores = []
for item in canon_smi_ls: 
    syba_scores.append(syba.predict(smi=item))
        
A = np.argsort(syba_scores)
smi_arranged = [canon_smi_ls[i] for i in A]
smi_arranged = smi_arranged[-8:]

mols_ = [Chem.MolFromSmiles(x) for x in smi_arranged]

img=Draw.MolsToGridImage(mols_,molsPerRow=4, subImgSize=(200,200))    
img