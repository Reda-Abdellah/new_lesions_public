import os
import platform
import subprocess
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import logging
from shutil import copyfile
import nibabel as nii

refernce_MNI='Registration/MNI_miplab-flair_sym_with-skull.nii'
refernce_mask_MNI='Registration/MNI_miplab-flair_sym_mask.nii'

def read_transform(filename):  # Simplified version of read_transform
    f = open(filename)
    text = f.read().split()
    M = np.eye(4)
    index = 9
    for i in range(4):
        if index >= 21:  # We read until word 21
            break
        for j in range(3):
            M[j, i] = np.float64(text[index])
            index += 1
    return M

def QC(fname, filename= refernce_MNI):
    mask = nii.load(
        filename=refernce_mask_MNI)
    mask = mask.get_fdata()
    T1 = nii.load(filename= filename)
    T1 = T1.get_fdata()
    ima = nii.load(filename=fname)
    ima = ima.get_fdata()

    ind = np.where(np.isnan(ima))
    cc = np.corrcoef(T1[ind] * mask[ind],
                     ima[ind] * mask[ind])
    correlation1 = cc[0][1]
    cc = np.corrcoef(T1[ind], ima[ind])
    correlation2 = cc[0][1]
    return correlation1, correlation2

def run_cmd(cmd):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()
    print(output)

def MASK_to_native(affine_intoMNI, mask, reference, native_mask):
    bin_name = 'Registration/antsApplyTransforms'
    #interpolation='GenericLabel[Linear]'
    interpolation='MultiLabel[0.3,0]'
    if('.gz' in mask):
        cmd='gunzip '+mask
        run_cmd(cmd)
        cmd='gunzip '+reference
        run_cmd(cmd)
        mask=mask.replace('.gz','')
        reference=reference.replace('.gz','')
        native_mask=native_mask.replace('.gz','')
        cmd= bin_name+' -d 3 -i '+mask+' -r '+reference+' -o '+native_mask+' -n '+interpolation+' -t ['+affine_intoMNI+', 1]'
        run_cmd(cmd)
        cmd='gzip '+mask
        run_cmd(cmd)
        cmd='gzip '+reference
        run_cmd(cmd)
        cmd='gzip '+native_mask
        run_cmd(cmd)
    else:
        cmd= bin_name+' -d 3 -i '+mask+' -r '+reference+' -o '+native_mask+' -n '+interpolation+' -t ['+affine_intoMNI+', 1]'
        run_cmd(cmd)
    return True

def MASK_to_MNI(affine_intoMNI, native_mask,mni_mask):
    bin_name = 'Registration/antsApplyTransforms'
    interpolation='MultiLabel[0.3,0]'
    if('.gz' in native_mask):
        cmd='gunzip '+native_mask
        run_cmd(cmd)
        native_mask=native_mask.replace('.gz','')
        mni_mask=mni_mask.replace('.gz','')
        cmd= bin_name+' -d 3 -i '+native_mask+' -r '+refernce_MNI+' -o '+mni_mask+' -n '+interpolation+' -t '+affine_intoMNI
        run_cmd(cmd)
        cmd='gzip '+native_mask
        run_cmd(cmd)
        cmd='gzip '+mni_mask
        run_cmd(cmd)
    else:
        cmd= bin_name+' -d 3 -i '+native_mask+' -r '+refernce_MNI+' -o '+mni_mask+' -n '+interpolation+' -t '+affine_intoMNI
        run_cmd(cmd)
    return True

def FLAIR_to_MNI(affine_intoMNI, in_flair, registred_flair):
    bin_name = 'Registration/antsApplyTransforms'
    if('.gz' in in_flair):
        cmd='gunzip '+in_flair
        run_cmd(cmd)
        in_flair=in_flair.replace('.gz','')
        registred_flair=registred_flair.replace('.gz','')
        cmd= bin_name+' -d 3 -i '+in_flair+' -r '+refernce_MNI+' -o '+registred_flair+' -n BSpline -t '+affine_intoMNI
        run_cmd(cmd)
        cmd='gzip '+in_flair
        run_cmd(cmd)
        cmd='gzip '+registred_flair
        run_cmd(cmd)
        
    else:
        cmd= bin_name+' -d 3 -i '+in_flair+' -r '+refernce_MNI+' -o '+registred_flair+' -n BSpline -t '+affine_intoMNI
        run_cmd(cmd)
    return True

def ToMNI_ANTS_ref(filename):
    # Parsing filenames
    # Getting image namefile  ###########
    ZIP=False
    if('.gz' in filename):
        cmd='gunzip '+filename
        run_cmd(cmd)
        ZIP=True
        filename=filename.replace('.gz','')
    ima1_name = filename.rsplit('/', 1)[-1]
    ft = filename.replace(ima1_name, 'affine_'+ima1_name)
    ft2 = ft.replace('.nii', 'Affine.txt')
    bin_name1 = 'Registration/ANTS'  # Declaring both binary name files for os compatible
    args = (bin_name1, '3', '-m', 'MI['+refernce_MNI+',' +
            filename+',1,32]', '-i', '0', '-o', ft)
    subprocess.run(args)
    args = (bin_name1, '3', '-m', 'MI['+refernce_MNI+','+filename+',1,32]', '-i', '0' '-o',
            ft, '--mask-image', refernce_mask_MNI, '--initial-affine', ft2)
    subprocess.run(args)
    if(ZIP):
        cmd='gzip '+filename
        run_cmd(cmd)
    return  ft2
