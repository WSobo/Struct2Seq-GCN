import torch
import numpy as np

def encode_ligand_elements(element_ids):
    """
    Converts LigandMPNN element integer IDs into a [M, 6] one-hot tensor.
    LigandMPNN maps elements to integers based on periodic table order:
    H=1, He=2, Li=3, C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, I=53...
    
    We map these to 6 biological bins:
    0: Carbon (6)
    1: Nitrogen (7)
    2: Oxygen (8)
    3: Sulfur (16)
    4: Phosphorus (15)
    5: Other / Halogens (Everything else)
    """
    M = element_ids.shape[0]
    one_hot = torch.zeros(M, 6, dtype=torch.float32)
    
    # 0: Carbon
    mask_C = (element_ids == 6)
    one_hot[mask_C, 0] = 1.0
    
    # 1: Nitrogen
    mask_N = (element_ids == 7)
    one_hot[mask_N, 1] = 1.0
    
    # 2: Oxygen
    mask_O = (element_ids == 8)
    one_hot[mask_O, 2] = 1.0
    
    # 3: Sulfur
    mask_S = (element_ids == 16)
    one_hot[mask_S, 3] = 1.0
    
    # 4: Phosphorus
    mask_P = (element_ids == 15)
    one_hot[mask_P, 4] = 1.0
    
    # 5: Other (Everything not 6, 7, 8, 15, 16)
    mask_other = ~(mask_C | mask_N | mask_O | mask_S | mask_P)
    one_hot[mask_other, 5] = 1.0
    
    return one_hot

# Test it
test_ids = torch.tensor([6, 8, 7, 16, 15, 9, 35, 6]) # C, O, N, S, P, F, Br, C
encoded = encode_ligand_elements(test_ids)
print("Input IDs:", test_ids.tolist())
print("Encoded [M, 6]:\n", encoded)
