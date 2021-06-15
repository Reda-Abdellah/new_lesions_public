import glob
from test_utils import *

img_path='dataset/training_preprocessed/for_testing/'
listaFLAIR1 = sorted(glob.glob(img_path+"*/*time01*"))
listaFLAIR2 = sorted(glob.glob(img_path+"*/*time02*"))
listaBRAIN = sorted(glob.glob(img_path+"*/*brain*"))
in_filepath = sorted(glob.glob("weights/2times_in_iqda_v2_LDICE_loss2/*.pt"))

#seg_to_folder_with_Weightlist_times(in_filepath,listaFLAIR1,listaFLAIR2,listaBRAIN)
listaGT = sorted(glob.glob(img_path+"*/*truth.nii*"))
listaPred = sorted(glob.glob("SEG/2times_in_iqda_v2_withDICE_PPV_LTPR_9_3_1/*SEG*"))
check_seg(listaGT, listaPred, listaBRAIN)

listaRater1 = sorted(glob.glob(img_path+"*/*expert1.nii*"))
listaRater2 = sorted(glob.glob(img_path+"*/*expert2.nii*"))
listaRater3 = sorted(glob.glob(img_path+"*/*expert3.nii*"))
listaRater4 = sorted(glob.glob(img_path+"*/*expert4.nii*"))
check_seg_many_raters([listaRater1,listaRater2,listaRater3,listaRater4], listaPred, listaBRAIN)
