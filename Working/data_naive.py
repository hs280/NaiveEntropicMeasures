import sys
import os
import numpy as np 
from pathlib import Path  # Module for working with filesystem paths
import pickle as pkl
script_path = os.path.abspath(__file__)
curent_path = os.path.dirname(script_path)
directory_folder = str(Path(curent_path).parents[0])
sys.path.insert(0, directory_folder)
import Bin as Bin
import csv

def save_dict_to_csv(data_dict, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(['Key', 'Value'])
        
        # Write the data
        for key, value in data_dict.items():
            writer.writerow([key, value])

def process_data_based(seq_data_path,target_data_path,rank_data,save_path,file_names):
    aligned_residues_df = Bin.fasta_to_dataframe(seq_data_path)
    target_df = Bin.read_dat_file(target_data_path)
    os.makedirs(save_path,exist_ok=True)
    rank_data_path = f'{save_path}/rank_data.pkl'
    with open(rank_data_path,'wb') as f:
        pkl.dump(rank_data,f)
    num_samples=np.inf

    max_seq_length = len(aligned_residues_df.values[0])

    Bin.search_sequence_lengths(save_path, 
                            aligned_residues_df, 
                            target_df, 
                            rank_data_path, 
                            num_samples, 
                            split_fraction=0.2, 
                            max_seq_length=max_seq_length, 
                            num_runs=5,
                            file_names=file_names,
                            alpha=0)
    
    Bin.sanitize_directory(save_path)

    resultr = Bin.calculate_sum(save_path)

    save_dict_to_csv(resultr, f'{save_path}/merit_values.csv')

    return resultr

# data paths alk mono_oxygenase
amo_focus = 'MKFGLFFLNFMNSKRSSDQVIEEMLDTAHYVDQLKFDTLAVYENHFSNNGVVGAPLTVAGFLLGMTKNAKVASLNHVITTHHPVRVAEEACLLDQMSEGRFAFGFSDCEKSADMRFFNRPTDSQFQLFSECHKIINDAFTTGYCHPNNDFYSFPKISVNPHAFTEGGPAQFVNATSKEVVEWAAKLGLPLVFRWDDSNAQRKEYAGLYHEVAQAHGVDVSQVRHKLTLLVNQNVDGEAARAEARVYLEEFVRESYSNTDFEQKMGELLSENAIGTYEESTQAARVAIECCGAADLLMSFESMEDKAQQRAVIDVVNANIVKYHS'

amo_focus_aligned = '------------------------------------------------------------\
---------------------------------------------MK-------------\
--F-----G-----LFFLNFMNSK-R------SSDQVIEEMLDTAHYVDQ--LK-FDTLA\
VYENHFS-NN------GVVGAPLTVAGFLLGMTKNAKVASLNHVITT-HHPVRVAEEACL\
LDQMS-----E-----GRFAFGFSD-CEKS---ADM-R-FFN-------------RP-TD\
---S-------------QF-------------Q-LFSE-CHKIINDAF----T-TG----\
-Y-------------CHP---NND--FYSFPKIS------VNPH---A-FTEGGP-----\
---------------------------------------------AQ-FVNATS-----K\
EV--VEWAAKLGLPLVFRWDDSN--AQRKE-YA---GLYHEVAQ----------------\
------------------------------------AHGVD-----VS--QVR--HKLTL\
LVN-QNVD--GEAARAEARVYLEE-FVRESYS----------------------------\
------------------------NTDF------EQ--KMGELLSENAIGTYEESTQAAR\
VAIECCG--AADLLMSFESMEDKAQQRAVIDVVNA-N---------I----V--------\
----K--YHS--------------------------------------------------\
------------------------------------------------------------\
-------------------------------'

amo_seq_data = './Data/AlkMonoxygenase/alkanal_monoxygenase_aligned.dat'

amo_target_data = './Data/AlkMonoxygenase/target_phopt.dat'

save_path_amo = './Results/AlkMonoxygenase/databased_naive'

## GR
gr_focus = 'MLMTVFSSAPELALLGSTFAQVDPSNLSVSDSLTYGQFNLVYNAFSFAIAAMFASALFFFSAQALVGQRYRLALLVSAIVVSIAGYHYFRIFNSWDAAYVLENGVYSLTSEKFNDAYRYVEWLLTVPLLLVETVAVLTLPAKEARPLLIKLTVASVLMIATLYPGWISDDITTRIIWGTVSTIPFAYILYVLWVELSRSLVRQPAAVQTLVRNMRWLLLLSWGVYPIAYLLPMLGVSGTSAAVGVQVGYTIADVLSKPVFGLLVFAIALVKTKADQESSEPHAAIGAAANKSGGSLI'
gr_focus_aligned = '------------------------------------------------------------\
-----------------MLMT---VFSSAPELAL--LGSTFAQVDP---SN-LSVSDSLT\
--YGQFNLVYNAFS-FAIAAMFASALFFFSAQALVGQRYRLALLVSAI--VVSIAGYHYF\
RIFNSWDAAYVL---EN-GVYS------L----TSEK--FNDAYRYVDWLLTVPLLLVET\
VAV------LTLPA--------K-----EARPLLIK-LTVA----SVLMIATGYPGEI--\
SD---------DITTRIIWGTVST-IPFAY-------ILY-VLWVEL-S-----------\
-------------------RS-LVRQPAA------------------VQTLVRNMRWLLL\
LSWGVYPIAYLLPML-GVS-G----------TS-AAVGVQVGYTIADVLAKPVFGLL-VF\
A-IALVKT-KADQES-----SEPHAA------IGAAANKSGGSLIS--------------\
---------------'

gr_seq_data = './Data/BacRhod/residues_aligned_reordered.dat'

gr_target_data = './Data/BacRhod/wavelengths.dat'

save_path_gr = './Results/BacRhod/databased_naive'

## FP
FP_focus = 'MSLSKHGITQEMPTKYHMKGSVNGHEFEIEGVGTGHPYEGTHMAELVIIKPAGKPLPFSFDILSTVIQYGNRCFTKYPADLPDYFKQAYPGGMSYERSFVYQDGGIATASWNVGLEGNCFIHKSTYLGVNFPADGPVMTKKTIGWDKAFEKMTGFNEVLRGDVTEFLMLEGGGYHSCQFHSTYKPEKPVELPPNHVIEHHIVRTDLGKTAKGFMVKLVQHAAAHVNPLKVQ'
FP_focus_aligned = '------------------------------------------------------------\
---M-S--LSK---HGITQEMP-TKYHMKGSVNGHEF--EIEGVG-TGHPYEGTH-MAE-\
LVIIKPAGK---PLPFSFDILSTVIQY---GNRCFT----KYPADLP--DYFKQAYPGG-\
MSY---------ERSFVYQDGGIATASWNV--GLE-----G-NCF-IHK-STYL--GVNF\
PAD--G----PVMTKKTIGWD----KAFEKMTG--F---NEVLRGDVTEFL--MLEG--G\
-GYHSCQFHSTYK-PEKPV-E--LP-PNHVIEH-HIVRTDLGKTA-KGFMVKLVQH-AAA\
--HV--NPL-K--VQ------------------------------'

FP_seq_data = './Data/GFP/aligned_sequences.dat'

FP_fluor_target_data = './Data/GFP/fp_fluorescence_wavelengths.dat'

save_path_FP_fluor = './Results/GFP_fluor/databased_naive'

FP_emission_target_data = './Data/GFP/fp_emission_wavelengths.dat'

save_path_FP_emission = './Results/GFP_emission/databased_naive'

FP_QY_target_data = './Data/GFP/fp_quantum_yield.dat'

save_path_FP_QY = './Results/GFP_QY/databased_naive'

## load_data 
data_amo,_ = Bin.naive_loader(amo_focus_aligned,amo_focus,'./Results/AlkMonoxygenase')
data_gr,_ = Bin.naive_loader(gr_focus_aligned,gr_focus,'./Results/BacRhod')
data_FP_fluor,_ = Bin.naive_loader(FP_focus_aligned,FP_focus,'./Results/GFP_fluor')
data_FP_emission,_ = Bin.naive_loader(FP_focus_aligned,FP_focus,'./Results/GFP_emission')
data_FP_QY,_ = Bin.naive_loader(FP_focus_aligned,FP_focus,'./Results/GFP_QY')

Full_data = [data_amo,data_gr,data_FP_fluor,data_FP_emission,data_FP_QY]
legends = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']


import ray

# Initialize Ray
ray.init()


amo_result = process_data_based(amo_seq_data,amo_target_data,data_amo,save_path_amo,legends)#
print('done')
gr_result = process_data_based(gr_seq_data,gr_target_data,data_gr,save_path_gr,legends)
print('done')
FP_result = process_data_based(FP_seq_data,FP_fluor_target_data,data_FP_fluor,save_path_FP_fluor,legends)
print('done')
FP_result = process_data_based(FP_seq_data,FP_emission_target_data,data_FP_fluor,save_path_FP_emission,legends)
print('done')
FP_result = process_data_based(FP_seq_data,FP_QY_target_data,data_FP_fluor,save_path_FP_QY,legends)
print('done')

ray.shutdown()