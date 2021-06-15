import glob, os, shutil
from test_utils import *
import configparser as ConfParser
import argparse

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
#print(cmd)
os.system(cmd)
get_new_lesions(args.output_name,preprocessed_flair1_name,preprocessed_flair2_name,brain_mask_name)
