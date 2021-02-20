#   Copyright (C) 2021 Greenweaves Software Limited
#
#   This program is free software: you can redistribute it and/or
#   modify it under the terms of the GNU Lesser General Public
#   License as published by the Free Software Foundation; either
#   version 2.1 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU  Lesser General Public License for more details
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  To contact me, Simon Crase, email simon@greenweaves.nz
#
# Segment images using supplied CellSegmenter

from argparse                   import ArgumentParser
from glob                       import glob
from hpacellseg.cellsegmentator import CellSegmentator
from hpacellseg.utils           import label_cell, label_nuclei
from imageio                    import imwrite
from matplotlib.pyplot          import figure, imread, imshow, axis, savefig, close, show
from numpy                      import dstack,uint8
import numpy as np
from os.path                    import join,basename
from random                     import sample
from time                       import time
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import base64
def binary_mask_to_ascii(mask, mask_val=1): # https://www.kaggle.com/dschettler8845/hpa-cellwise-classification-inference
    """Converts a binary mask into OID challenge encoding ascii text."""
    mask = np.where(mask==mask_val, 1, 0).astype(np.bool)

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Segment HPA data using HPA Cell segmenter')
    parser.add_argument('--path',
                        default = r'd:\data\hpa-scc',
                        help    = 'Path to data')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Identified subset of data-- e.g. train512x512')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Identifies where to store plots')
    parser.add_argument('--sample',
                        type    = int,
                        default = 3,
                        help    = 'Specifies number of images to be sampled at random and processed')
    parser.add_argument('--segments',
                        default = './segments',
                        help    = 'Identifies where to store plots')
    parser.add_argument('--NuclearModel',
                        default = './nuclei-model.pth',
                        help    = 'Identifies where to store nuclear models')
    parser.add_argument('--CellModel',
                        default = './cell-model.pth',
                        help    = 'Identifies where to store cell models')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display plots')
    args        = parser.parse_args()

    image_dir   = join(args.path,args.image_set)
    mt          = sample(glob(join(image_dir, '*_red.png')),args.sample)
    er          = [f.replace('red', 'yellow') for f in mt]
    nu          = [f.replace('red', 'blue') for f in mt]
    images      = [mt, er, nu]

    segmentator = CellSegmentator(
        nuclei_model        = args.NuclearModel,
        cell_model          = args.CellModel,
        scale_factor        = 0.25,
        device              = "cpu",
        padding             = False,
        multi_channel_model = True,
    )

    # For nuclei
    nuc_segmentations = segmentator.pred_nuclei(nu)

    # For full cells
    cell_segmentations = segmentator.pred_cells(images)

    # post-processing
    for i, pred in enumerate(cell_segmentations):
        nuclei_mask, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i])
        FOVname                = basename(mt[i]).replace('red','predictedmask')
        print (FOVname)
        imwrite(join(args.segments,FOVname), cell_mask)
        fig            = figure(figsize=(19,10))
        microtubule    = imread(mt[i])
        endoplasmicrec = imread(er[i])
        nuclei         = imread(nu[i])
        img            = dstack((microtubule, endoplasmicrec, nuclei))
        imshow(img)
        imshow(cell_mask, alpha=0.5)
        axis('off')
        savefig(join(args.figs,basename(mt[i]).replace('red','xx')))
        ll = label_cell(nuc_segmentations[i], cell_segmentations[i])[1].astype(uint8)
        bb = binary_mask_to_ascii(ll,i)
        print (bb)
        if not args.show:
            close(fig)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
