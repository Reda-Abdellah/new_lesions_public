import glob, os, shutil
from test_utils import *
import configparser as ConfParser
import argparse
from Registration.registration import MASK_to_native

# Argument parsing
parser = argparse.ArgumentParser(
    description="""Blabla""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-t1', '--time1', type=str, required=True)
parser.add_argument('-t2', '--time2', type=str, required=True)
parser.add_argument('-o', '--output_name', type=str, required=True)

args = parser.parse_args()
preprocessed_flair1_name= '/data/patients_preprocessed/patient_X/flair_time01_on_middle_space.nii.gz'
preprocessed_flair2_name= '/data/patients_preprocessed/patient_X/flair_time02_on_middle_space.nii.gz'
brain_mask_name= '/data/patients_preprocessed/patient_X/brain_mask.nii.gz'
flair1_name= '/data/patients/patient_X/flair_time01_on_middle_space.nii.gz'
flair2_name= '/data/patients/patient_X/flair_time02_on_middle_space.nii.gz'
shutil.copyfile(args.time1, flair1_name)
shutil.copyfile(args.time2, flair2_name)

cmd= "python /anima/Anima-Scripts-Public/ms_lesion_segmentation/animaMSLongitudinalPreprocessing.py -i /data/patients/ -o /data/patients_preprocessed/"
os.system(cmd)

regestred_FLAIR1_name=preprocessed_flair1_name.replace('flair', 'mni_flair' )
regestred_FLAIR2_name= preprocessed_flair2_name.replace('flair', 'mni_flair' )
reg_mask_name= brain_mask_name.replace('brain', 'mni_brain' )
affine_intoMNI= times_to_mni(preprocessed_flair1_name, preprocessed_flair2_name, brain_mask_name,regestred_FLAIR1_name, regestred_FLAIR2_name, reg_mask_name, strategy='decoder_with_FMs')

mni_les=(args.output_name).replace('.nii','_mni.nii')
get_new_lesions_mni(mni_les,regestred_FLAIR1_name,regestred_FLAIR2_name,reg_mask_name)
MASK_to_native(affine_intoMNI, mni_les, flair1_name, args.output_name)
    