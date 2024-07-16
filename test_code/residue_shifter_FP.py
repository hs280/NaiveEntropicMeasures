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
original_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# Aligned sequence
aligned_seq = "------------------------------------------------------------\
------M-SKG--EELFTGVVP-ILVELDGDVNGHKF--SVSGEG-EGDATYGKL-TLK-\
FICT--TG----KLPVPWPTLVTTFSY---GVQCFS----RYPDHMKQHDFFKSAMPEG-\
YVQ---------ERTIFFKDDGNYKTRAEV--KF-----EG-DTL-VNR-IELK--GIDF\
KED--G----NILGHK-LEYN----YNSHNVYIMADKQ-KNGIKVNFKIRH--NIED--G\
SVQLADHYQQNTPIGDGPV---LLP-DNHYLST-QSALSKDPNE--KRDHMVLLEF-VTA\
------AGI-T--HGMDELYK------------------------"


# Indices to map
indices = [65,66,67,96,143,145,148,203,205,222]

# Map indices
aligned_indices = map_indices_to_aligned(original_seq, aligned_seq, indices)
print(aligned_indices)


aligned_seq_354 = "------------------------------------------------------------\
---M-S--LSK---HGITQEMP-TKYHMKGSVNGHEF--EIEGVG-TGHPYEGTH-MAE-\
LVIIKPAGK---PLPFSFDILSTVIQY---GNRCFT----KYPADLP--DYFKQAYPGG-\
MSY---------ERSFVYQDGGIATASWNV--GLE-----G-NCF-IHK-STYL--GVNF\
PAD--G----PVMTKKTIGWD----KAFEKMTG--F---NEVLRGDVTEFL--MLEG--G\
-GYHSCQFHSTYK-PEKPV-E--LP-PNHVIEH-HIVRTDLGKTA-KGFMVKLVQH-AAA\
--HV--NPL-K--VQ------------------------------"

# Indices to extract (1-based indexing)

# Convert indices to 0-based indexing
indices_to_extract_zero_based = [i - 1 for i in aligned_indices]

# Extract residues at the specified indices
extracted_residues = [aligned_seq_354[i] for i in indices_to_extract_zero_based]

# Convert to a string
extracted_residues_str = ''.join(extracted_residues)

print(extracted_residues_str)

print(len(extracted_residues_str))
print(len(aligned_indices))
print(len(indices))


print(map_aligned_to_original(aligned_seq_354,aligned_indices))

new_ind = map_aligned_to_original(aligned_seq_354,aligned_indices)

prot_loc = [f'{r}{l}' for r,l in zip(extracted_residues,new_ind)]
print(prot_loc)

