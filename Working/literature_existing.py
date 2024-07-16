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
rhodopsins_indices = [40, 81, 85, 130, 132, 133, 136, 137, 140, 168, 169, 173, 188, 192, 193, 196, 241, 244, 245, 249, 267, 271, 275, 276]

# GFP and Variants
gfp_indices = [65, 66, 148, 203, 222]

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
_,data_amo = Bin.main_loader(amo_focus_aligned,amo_focus,deep_amo,plmc_amo,hotspot_amo)
_,data_gr = Bin.main_loader(gr_focus_aligned,gr_focus,deep_gr,plmc_gr,hotspot_gr)
_,data_FP = Bin.main_loader(FP_focus_aligned,FP_focus,deep_FP,plmc_FP,hotspot_FP)

Full_data = [data_amo,data_gr,data_FP]


labels = ['AMO', 'GR', 'FP']
legends = ['HotspotWizard', 'EV+', 'EV-', 'Deep+', 'Deep-', 'Abs(Deep)']
key_residues = [amo_key_indices,gr_key_indices, FP_key_indices]

Bin.plot_stacked_lines(Full_data, labels, legends, key_residues,output_filename='lit_existing_stacked.png')
min_ratio_dict_existing = Bin.plot_minimal_bars(Full_data, key_residues, legends,output_filename='lit_existing_bar.png')






