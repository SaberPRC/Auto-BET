import os
import ants
import argparse

from tqdm import tqdm
from IPython import embed


def _ACPC_Correction(moving_img_path, fixed_img_path=None, type_of_transform='Rigid'):
    moving_img = ants.image_read(moving_img_path)
    fixed_img = ants.image_read(fixed_img_path)
    res = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=type_of_transform)
    
    return res['warpedmovout']
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Setting for Persudo Brain Mask Generation')
    parser.add_argument('--input', type=str, default='/path/to/input/T1w/Image', help='Original T1 image')
    parser.add_argument('--output', type=str, default='/path/to/output/ACPC/Corrected/Image', help='Persudo Extracted Brain')
    parser.add_argument('--RefImg', type=str, default='../atlas/MNI152_T1.nii.gz', help='MNI152 image')

    args = parser.parse_args()

    img_acpc_corrected = _ACPC_Correction(moving_img_path=args.input, fixed_img_path=args.RefImg)
 
    ants.image_write(img_acpc_corrected, args.output)
