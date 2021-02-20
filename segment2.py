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
from base64                     import b64encode
from csv                        import reader
from glob                       import glob
from hpacellseg.cellsegmentator import CellSegmentator
from hpacellseg.utils           import label_cell, label_nuclei
from imageio                    import imwrite
from matplotlib.pyplot          import figure, imread, imshow, axis, savefig, close, show, title,suptitle
from numpy                      import dstack,uint8, where, bool, squeeze, asfortranarray
from os.path                    import join,basename
from pycocotools                import _mask as coco_mask
from random                     import sample
from time                       import time
from zlib                       import compress, Z_BEST_COMPRESSION

def create_descriptions(file_name='descriptions.csv'):
    with open(file_name) as description_file:
        return {int(row[0]) : row[1] for row in reader(description_file) }


# read_training_expectations
#
# Read and parse the  training image-level labels
#
# Parameters:
#     path       Path to image-level labels
#     file_name  Name of image-level labels file

def read_training_expectations(path=r'd:\data\hpa-scc',file_name='train.csv'):
    with open(join(path,file_name)) as train:
        rows = reader(train)
        next(rows)
        return {row[0]: [int(label) for label in row[1].split('|')] for row in rows}

def binary_mask_to_ascii(mask, mask_val=1): # https://www.kaggle.com/dschettler8845/hpa-cellwise-classification-inference
    """Converts a binary mask into OID challenge encoding ascii text."""
    mask = where(mask==mask_val, 1, 0).astype(bool)

    # check input mask --
    if mask.dtype != bool:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(uint8)
    mask_to_encode = asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = compress(encoded_mask, Z_BEST_COMPRESSION)
    base64_str = b64encode(binary_str)
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
                        # default = 3,
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
    parser.add_argument('--descriptions',
                        default='descriptions.csv')
    parser.add_argument('--classes',
                        default = ['all'],
                        nargs   = '+',
                        help    = 'List of classes to be processed')
    parser.add_argument('--multiplets',
                        default = False,
                        action  = 'store_true',
                        help    = 'Allow multiple assignments')
    args         = parser.parse_args()
    Descriptions = create_descriptions(file_name=args.descriptions)
    Expectations = read_training_expectations() # FIXME - add parameters

    if not args.multiplets:
        Expectations = {file_name:class_names for file_name,class_names in Expectations.items() if len(class_names)==1}
    if args.classes[0]!='all':   #FIXME
        classes = [int(c) for c in args.classes]
        Expectations = {file_name:class_names for file_name,class_names in Expectations.items() if class_names[0] in classes}

    file_list   = list(Expectations.keys())
    print (f'Processing {len(file_list)} slides')
    if args.sample!=None:
        file_list = sample(file_list,args.sample)
    image_dir   = join(args.path,args.image_set)
    mt          = [join(image_dir,f'{name}_red.png') for name in file_list]
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
        classes = Expectations[file_list[i]]
        class_descriptions = [Descriptions[class_id] for class_id in classes]
        # FIXME - this doesn't handle multiple classes
        title(f'{file_list[i]} {classes[0]} {class_descriptions[0]}')
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
