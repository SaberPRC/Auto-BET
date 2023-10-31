import os
import ants
import argparse
from scipy import ndimage


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _ants_registration(moving_img_path, moving_seg_path=None, fixed_img_path=None, type_of_transform='SyN'):
    moving_img = ants.image_read(moving_img_path)
    if moving_seg_path == None:
        moving_img = ants.image_read(moving_img_path)
        fixed_img = ants.image_read(fixed_img_path)
        res = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=type_of_transform)
        return res['warpedmovout']
    else:
        moving_img = ants.image_read(moving_img_path)
        moving_seg = ants.image_read(moving_seg_path)
        fixed_img = ants.image_read(fixed_img_path)

        res = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=type_of_transform)

        warped_img = res['warpedmovout']
        warped_seg = ants.apply_transforms(fixed=fixed_img, moving=moving_seg, transformlist=res['fwdtransforms'], interpolator='nearestNeighbor')

        return warped_img, warped_seg
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Setting for Persudo Brain Mask Generation')
    parser.add_argument('--input', type=str, default='/path/to/input/T1w/image', help='Original T1 image')
    parser.add_argument('--output', type=str, default='/path/to/output/persudo/brain/extraction', help='Persudo Extracted Brain')
    parser.add_argument('--RefImg', type=str, default='../atlas/MNI152_T1.nii.gz', help='MNI152 image')
    parser.add_argument('--RefSeg', type=str, default=-'../atlas/MNI152_mask.nii.gz', help='MNI152 brain')

    args = parser.parse_args()

    warped_mni_img, warped_mni_mask = _ants_registration(moving_img_path=args.RefImg, moving_seg_path=args.RefSeg, fixed_img_path=args.input)
 
    origin, spacing, direction, img = _ants_img_info(args.input)

    persudo_mask = warped_mni_mask.numpy()
    persudo_mask = ndimage.binary_dilation(persudo_mask, iterations=5)
    persudo_mask = persudo_mask.astype(np.float32)

    persudo_brain = img * persudo_mask
    persudo_brain = ants.from_numpy(persudo_brain, origin, spacing, direction)

    ants.image_write(persudo_brain, args.output)