def map_indices_to_aligned(original, aligned, indices):
    aligned_indices = []
    original_pos = 0
    for i, char in enumerate(aligned):
        if char != '-':
            original_pos += 1
        if original_pos in indices:
            aligned_indices.append(i+1)
            indices.remove(original_pos)
            if not indices:
                break
    return aligned_indices

def map_aligned_to_original(aligned, indices):
    original_indices = []
    aligned_indices_set = set(indices)  # Use a set for O(1) lookup times
    original_pos = 0
    for i, char in enumerate(aligned):
        if char != '-':
            original_pos += 1
        if (i + 1) in aligned_indices_set:
            if char != '-':
                original_indices.append(original_pos)
            else:
                original_indices.append(None)  # Append None for gap positions
    return original_indices

# Original sequence
original_seq = "QAQITGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVPAIAFTMYLSMLLGYGLTMVPFGGEQNPIYWARYADWLFTTPLLLLDLALLVDADQGTILALVGADGIMIGTGLVGALTKVYSYRFVWWAISTAAMLYILYVLFFGFTSKAESMRPEVASTFKVLRNVTVVLWSAYPVVWLIGSEGAGIVPLNIETLLFMVLDVSAKVGFGLILLRSRAIFGEAEAPEPSAGDGAAATSD"

# Aligned sequence
aligned_seq = "------------------------------------------------------------"
aligned_seq += "-----------------------------------------------------------Q"
aligned_seq += "--AQITGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITT-LVPAIAFTMYL"
aligned_seq += "SMLLGYGLTMVPF-----G-------------GEQNP---IYWARYADWLFTTPLLLLDL"
aligned_seq += "ALL------V------------D-----ADQGTILA-LVGA----DGIMIGTGLVGAL--"
aligned_seq += "TKV---------YSYRFVWWAIST-AAMLY-------ILY-VLFFGF-T-----------"
aligned_seq += "-------S----------KA--ESMRPE-------------------VASTFKVLRNVTV"
aligned_seq += "VLWSAYPVVWLIGSE-G-A-GI---------V--PLNIETLLFMVLDVSAKVGFGLI-LL"
aligned_seq += "R---SRAI-FGEAEA-----PEPSAG----DGAAATSD----------------------"
aligned_seq += "---------------"

# Indices to map
indices = [20, 49, 53, 83, 85, 86, 89, 90, 93, 118, 119, 122, 138, 141, 142, 145, 182, 185, 186, 189, 208, 212, 215, 216]

# Map indices
aligned_indices = map_indices_to_aligned(original_seq, aligned_seq, indices)
print(aligned_indices)


aligned_seq_354 = (
    "------------------------------------------------------------"
    "-----------------MLMT---VFSSAPELAL--LGSTFAQVDP---SN-LSVSDSLT"
    "--YGQFNLVYNAFS-FAIAAMFASALFFFSAQALVGQRYRLALLVSAI--VVSIAGYHYF"
    "RIFNSWDAAYVL---EN-GVYS------L----TSEK--FNDAYRYVDWLLTVPLLLVET"
    "VAV------LTLPA--------K-----EARPLLIK-LTVA----SVLMIATGYPGEI--"
    "SD---------DITTRIIWGTVST-IPFAY-------ILY-VLWVEL-S-----------"
    "-------------------RS-LVRQPAA------------------VQTLVRNMRWLLL"
    "LSWGVYPIAYLLPML-GVS-G----------TS-AAVGVQVGYTIADVLAKPVFGLL-VF"
    "A-IALVKT-KADQES-----SEPHAA------IGAAANKSGGSLIS--------------"
    "---------------"
)

# Indices to extract (1-based indexing)

# Convert indices to 0-based indexing
indices_to_extract_zero_based = [i - 1 for i in aligned_indices]

# Extract residues at the specified indices
extracted_residues = [aligned_seq_354[i] for i in indices_to_extract_zero_based]

# Convert to a string
extracted_residues_str = ''.join(extracted_residues)

print(extracted_residues_str)

new_ind = map_aligned_to_original(aligned_seq_354,aligned_indices)

prot_loc = [f'{r}{l}' for r,l in zip(extracted_residues,new_ind)]
print(prot_loc)

