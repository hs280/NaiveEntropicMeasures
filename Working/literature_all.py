import sys
import os
import numpy as np 
from pathlib import Path  # Module for working with filesystem paths
script_path = os.path.abspath(__file__)
curent_path = os.path.dirname(script_path)
directory_folder = str(Path(curent_path).parents[0])
sys.path.insert(0, directory_folder)
import Bin as Bin


# Rhodopsins
rhodopsins_indices = [52, 80, 84, 119, 121, 122, 125, 126, 129, 158, 159, 162, 178, 181, 182, 185, 222, 225, 226, 229, 249, 253, 256, 257]

# GFP and Variants
gfp_indices = [68, 69, 70, 97, 145, 147, 150, 199, 201, 219]

# Alkanal Monooxygenases
alkanol_monooxygenases_indices = [44, 45, 106, 113, 227]


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

hotspot_amo = './Data/AlkMonoxygenase/hotspot_wizard.csv'
plmc_amo = 'Data/AlkMonoxygenase/Sij.csv'
deep_amo = './Data/AlkMonoxygenase/key_tuning_residues_no_rep.csv'

amo_key_indices = list((np.asarray(alkanol_monooxygenases_indices)-1)) #0 inexing


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

hotspot_gr = 'Data/AlkMonoxygenase/hotspot_wizard.csv'
plmc_gr = './Data/BacRhod/Sij.csv'
deep_gr = './Data/BacRhod/key_tuning_residues_no_rep.csv'

gr_key_indices = list((np.asarray(rhodopsins_indices)-1))
## FP
FP_focus = 'MSLSKHGITQEMPTKYHMKGSVNGHEFEIEGVGTGHPYEGTHMAELVIIKPAGKPLPFSFDILSTVIQYGNRCFTKYPADLPDYFKQAYPGGMSYERSFVYQDGGIATASWNVGLEGNCFIHKSTYLGVNFPADGPVMTKKTIGWDKAFEKMTGFNEVLRGDVTEFLMLEGGGYHSCQFHSTYKPEKPVELPPNHVIEHHIVRTDLGKTAKGFMVKLVQHAAAHVNPLKVQ'
FP_focus_aligned = '------------------------------------------------------------\
---M-S--LSK---HGITQEMP-TKYHMKGSVNGHEF--EIEGVG-TGHPYEGTH-MAE-\
LVIIKPAGK---PLPFSFDILSTVIQY---GNRCFT----KYPADLP--DYFKQAYPGG-\
MSY---------ERSFVYQDGGIATASWNV--GLE-----G-NCF-IHK-STYL--GVNF\
PAD--G----PVMTKKTIGWD----KAFEKMTG--F---NEVLRGDVTEFL--MLEG--G\
-GYHSCQFHSTYK-PEKPV-E--LP-PNHVIEH-HIVRTDLGKTA-KGFMVKLVQH-AAA\
--HV--NPL-K--VQ------------------------------'

hotspot_FP = 'Data/AlkMonoxygenase/hotspot_wizard.csv'
plmc_FP = './Data/GFP/Sij.csv'
deep_FP = './Data/GFP/key_tuning_residues_no_rep.csv'

FP_key_indices = list((np.asarray(gfp_indices)-1))

## load_data 
_,data_amo_existing = Bin.main_loader(amo_focus_aligned,amo_focus,deep_amo,plmc_amo,hotspot_amo)
_,data_gr_existing = Bin.main_loader(gr_focus_aligned,gr_focus,deep_gr,plmc_gr,hotspot_gr)
_,data_FP_existing = Bin.main_loader(FP_focus_aligned,FP_focus,deep_FP,plmc_FP,hotspot_FP)


## load_data 
_,data_amo_naive = Bin.naive_loader(amo_focus_aligned,amo_focus,'./Results/AlkMonoxygenase')
_,data_gr_naive = Bin.naive_loader(gr_focus_aligned,gr_focus,'./Results/BacRhod')
_,data_FP_naive = Bin.naive_loader(FP_focus_aligned,FP_focus,'./Results/GFP_fluor')



Full_data_existing = [data_amo_existing,data_gr_existing,data_FP_existing]

Full_data_naive = [data_amo_naive,data_gr_naive,data_FP_naive]

labels = ['AMO', 'GR', 'FP']
legends_existing = ['Hotspot Wizard', 'EV+', 'EV-', 'Deep+', 'Deep-', 'AbsDeep']
legends_naive = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']
key_residues = [amo_key_indices,gr_key_indices, FP_key_indices]

Protein_Families = ['Alkane \n Monoxygenase',
               'Rhodopsins',
               'Fluorescent \n Proteins',
               'Fluorescent Proteins',
               'Fluorescent Proteins'
               ]

# Bin.plot_stacked_lines(Full_data_existing, labels, legends_existing, key_residues,output_filename='lit_existing_stacked.png',Protein_Families=Protein_Families)
# Bin.plot_stacked_lines(Full_data_naive, labels, legends_naive, key_residues,output_filename='lit_stacked_naive.png',Protein_Families=Protein_Families)

min_ratio_dict_existing = Bin.plot_minimal_bars(Full_data_existing, key_residues, legends_existing,output_filename='lit_existing_bar.png')
min_ratio_dict_naive = Bin.plot_minimal_bars(Full_data_naive, key_residues, legends_naive,output_filename='lit_existing_bar.png')

Bin.plot_bar_chart_with_error_bars(min_ratio_dict_existing, min_ratio_dict_naive,labels = ['dict1','dict2'],ylabel='Coherence',
                                   yticks=[0.001,1], ylim=(0.01, 1.1), 
                                   yticklabels=['1E-3','1'],rotation=90,
                                   figsize=(4/3*11.69, 11.69),
                                   figsize_cross_corr = (4/3*11.69,4/3*11.69),outfolder='figures',is_log=True)






