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

FP_key_indices = list((np.asarray(gfp_indices)-1))

## load_data 
_,data_amo = Bin.naive_loader(amo_focus_aligned,amo_focus,'./Results/AlkMonoxygenase')
_,data_gr = Bin.naive_loader(gr_focus_aligned,gr_focus,'./Results/BacRhod')
_,data_FP = Bin.naive_loader(FP_focus_aligned,FP_focus,'./Results/GFP_fluor')

Full_data = [data_amo,data_gr,data_FP]


labels = ['Set 1', 'Set 2', 'Set 3']
legends = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']
key_residues = [amo_key_indices,gr_key_indices, FP_key_indices]

Bin.plot_stacked_lines(Full_data, labels, legends, key_residues,output_filename='lit_stacked_naive.png')
Bin.plot_optimal_bars(Full_data, key_residues, legends,output_filename='lit_bar_naive.png')




