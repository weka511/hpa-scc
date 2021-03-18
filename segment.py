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
# Segment images using CellSegmentator aupplied by HPA

from argparse                   import ArgumentParser
from base64                     import b64encode
from csv                        import reader, writer
from glob                       import glob
from hpacellseg.cellsegmentator import CellSegmentator
from hpacellseg.utils           import label_cell, label_nuclei
from imageio                    import imwrite
from math                       import isqrt
from matplotlib.pyplot          import figure, imread, imshow, axis, savefig, close, show, title,suptitle,plot,xlabel,ylabel
from numpy                      import dstack,uint8, where, bool, squeeze, asfortranarray, save
from os                         import environ
from os.path                    import join,basename
from pycocotools                import _mask as coco_mask
from random                     import sample, seed
from utils                      import Timer
from zlib                       import compress, Z_BEST_COMPRESSION

# create_descriptions
#
# Read descriptions of classes

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
#
#     Returns: dict of class assigments for each image name
#
# NB:  the  list of image classes contains non-unique entries --
#      See Dan Presil's post of 2021 February 21
#      https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/221025
#      so we need to do a list(set(...)) below

def read_training_expectations(path=join(environ['DATA'],'hpa-scc'),file_name='train.csv'):

    def unique(items):
        return list(set(items))
    with open(join(path,file_name)) as train:
        rows = reader(train)
        next(rows)
        return {row[0]: unique([int(label) for label in row[1].split('|')]) for row in rows}

# binary_mask_to_ascii

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

# segment_image
#
# Apply segmentation mask to determine which part of image should actually be shown
#
# Parameters:
#     img_cell       The entire image (destructive - this will be replaced by the return value)
#     binary_mask    The mask
#     class_id       The id of the class - all other pixels in image will be zeroed
#
#     Returns a copy of the image with all pixels (outside the image) zeroed

def segment_image(img_cell,binary_mask,class_id):
    for i in range(len(img_cell)):
        for j in range(len(img_cell[i])):
            if binary_mask[i][j] != class_id:
                img_cell[i][j] = 0
    return img_cell

def segment(nuclei_model = None,
            cell_model   = None,
            scale_factor = 0.5,
            nu           = [],
            images       = []):
    segmentator = CellSegmentator(
        nuclei_model        = args.NuclearModel,
        cell_model          = args.CellModel,
        scale_factor        = scale_factor,
        device              = "cpu",
        padding             = True,      # Changed from https://github.com/CellProfiling/HPA-Cell-Segmentation
        multi_channel_model = True,
    )

    nuc_segmentations  = segmentator.pred_nuclei(nu)          # For nuclei

    cell_segmentations = segmentator.pred_cells(images)       # For full cells

    return nuc_segmentations, cell_segmentations

def apply_masks(nuc_segmentations, cell_segmentations,
                file_list     = [],
                segments      = './segments',
                figs          = './figs',
                masks         = './masks',
                mt            = [],
                er            = [],
                nu            = [],
                show          = False,
                output_writer = None):
    Failures = []
    for i, pred in enumerate(cell_segmentations):
        nuclei_mask, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i])
        imwrite(join(segments,f'{file_list[i]}_predictedmask.png'), cell_mask)
        fig                    = figure(figsize=(19,10))
        microtubule            = imread(mt[i])
        endoplasmicrec         = imread(er[i])
        nuclei                 = imread(nu[i])
        imshow(dstack((microtubule, endoplasmicrec, nuclei)))
        imshow(cell_mask, alpha=0.5)
        classes            = Expectations[file_list[i]]
        class_descriptions = [Descriptions[class_id] for class_id in classes]

        title(f'{file_list[i]}  {", ".join(class_descriptions)}')
        axis('off')
        savefig(join(figs,basename(mt[i]).replace('red','full')))
        if not show:
            close(fig)
        binary_mask        = label_cell(nuc_segmentations[i], cell_segmentations[i])[1].astype(uint8)
        number_of_segments = binary_mask.max()+1
        if output_writer!=None:
            output_writer.writerow([file_list[i],number_of_segments])
        if number_of_segments>1:
            print (f'Segmented {file_list[i]}')
        else:
            print (f'Failed {file_list[i]}')
            Failures.append(file_list[i])

        fig                = figure(figsize=(20,20))
        m1                 = isqrt(number_of_segments)
        m2                 = number_of_segments // m1 + 1
        axs                = fig.subplots(m1, m2,squeeze=False)

        for k in range(m1):
            for l in range(m2):
                axs[k][l].axis('off')
                axs[k][l].set_xticklabels([])
                axs[k][l].set_yticklabels([])
        k = 0
        l = 0
        with open(join(masks,basename(mt[i]).replace('red','full').replace('png','npy')),'wb') as binary_mask_file:
            save(binary_mask_file,binary_mask)
        with open(join(masks,basename(mt[i]).replace('red','full').replace('png','txt')),'w') as ascii_mask_file:
            for j in range(1,binary_mask.max()+1):
                axs[k][l].imshow(segment_image(dstack((microtubule, endoplasmicrec, nuclei)),binary_mask,j))
                ascii_mask = binary_mask_to_ascii(binary_mask,j)
                ascii_mask_file.write(f'{j}\n')
                ascii_mask_file.write(f'{ascii_mask}\n')
                l += 1
                if l>=len(axs[k]):
                    k += 1
                    l  = 0

            suptitle(f'{file_list[i]}  {", ".join(class_descriptions)}')
            savefig(join(figs,basename(mt[i]).replace('red','segmented')))
        if not show:
            close(fig)
    return Failures

if __name__=='__main__':
    with Timer():
        parser = ArgumentParser('Segment HPA data using HPA Cell segmenter')
        parser.add_argument('--path',
                            default = join(environ['DATA'],'hpa-scc'),
                            help    = 'Path to data')
        parser.add_argument('--image_set',
                            default = 'train512x512',
                            help    = 'Identified subset of data-- e.g. train512x512')
        parser.add_argument('--figs',
                            default = './figs',
                            help    = 'Identifies where to store plots')
        parser.add_argument('--masks',
                            default = './masks',
                            help    = 'Identifies where to store ascii masks')
        parser.add_argument('--sample',
                            type    = int,
                            help    = 'Specifies number of images to be sampled at random and processed')
        parser.add_argument('--segments',
                            default = './segments',
                            help    = 'Identifies where to store cell masks')
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
        parser.add_argument('--file_name',
                            default = 'train.csv',
                            help    = 'List of image ids and classes')
        parser.add_argument('--seed',
                            default = None,
                            help = 'Seed for random number generator')
        parser.add_argument('--image_id',
                            default = None,
                            help    = 'Used to process single file only')
        parser.add_argument('--scale_factor',
                            default = 0.1125,   #Issue 33,
                            type    = float,
                            help    = 'Used by CellSegmentator')

        parser.add_argument('--sfrange',
                            default = [],
                            type    = float,
                            nargs   = 3,
                            help    = 'Used by CellSegmentator')
        parser.add_argument('--out',
                            default = 'segments.csv',
                            help    = 'Output file for segment counts')
        args         = parser.parse_args()

        seed(args.seed)
        Descriptions = create_descriptions(file_name=args.descriptions)
        Expectations = read_training_expectations(path=args.path,file_name=args.file_name)

        if args.image_id!=None:
            file_list = [args.image_id]
        else:
            if not args.multiplets:
                Expectations = {file_name:class_names for file_name,class_names in Expectations.items() if len(class_names)==1}
            if args.classes[0]!='all':   #FIXME
                classes      = [int(c) for c in args.classes]
                Expectations = {file_name:class_names for file_name,class_names in Expectations.items() if class_names[0] in classes}

            file_list        = list(Expectations.keys())

        if args.sample!=None:
            file_list = sample(file_list,args.sample)
            print (f'Processing {args.sample} slides')
        else:
            print (f'Processing {len(file_list)} slides')

        image_dir   = join(args.path,args.image_set)
        mt          = [join(image_dir,f'{name}_red.png')    for name in file_list]
        er          = [join(image_dir,f'{name}_yellow.png') for name in file_list]
        nu          = [join(image_dir,f'{name}_blue.png')   for name in file_list]
        images      = [mt, er, nu]

        if len(args.sfrange)==0:
            nuc_segmentations, cell_segmentations = segment(nuclei_model = args.NuclearModel,
                                                            cell_model   = args.CellModel,
                                                            scale_factor = args.scale_factor,
                                                            nu           = nu,
                                                            images       = images)
            with open(args.out,'w') as output_file:
                Failures = apply_masks(nuc_segmentations, cell_segmentations,
                                       file_list     = file_list,
                                       segments      = args.segments,
                                       mt            = mt,
                                       er            = er,
                                       nu            = nu,
                                       figs          = args.figs,
                                       masks         = args.masks,
                                       show          = args.show,
                                       output_writer = writer(output_file))

            print (f'There were {len(Failures)} failures out of {len(file_list)} -- {100*len(Failures)/len(file_list)}% with scale factor={args.scale_factor}')
            for failure in Failures:
                print (failure)

            if args.show:
                show()

        else:
            low  = args.sfrange[0]
            high = args.sfrange[1]
            n    = int(args.sfrange[2])
            step = (high-low)/n
            xs   = []
            ys   = []
            for i in range(n+1):
                scale_factor                          = low + i * step
                nuc_segmentations, cell_segmentations = segment(nuclei_model = args.NuclearModel,
                                                                cell_model   = args.CellModel,
                                                                scale_factor = scale_factor,
                                                                nu           = nu,
                                                                images       = images)

                Failures                              = apply_masks(nuc_segmentations, cell_segmentations,
                                                                    file_list = file_list,
                                                                    segments  = args.segments,
                                                                    mt        = mt,
                                                                    er        = er,
                                                                    nu        = nu,
                                                                    figs      = args.figs,
                                                                    masks     = args.masks,
                                                                    show      = False)
                xs.append(scale_factor)
                ys.append(len(Failures)/len(file_list))

            figure(figsize=(20,20))
            plot(xs,ys)
            title(f'{len(file_list)} images, seed = {args.seed}')
            xlabel('Scale Factor')
            ylabel('Failure Rate')
            savefig('scale-factors.png')
            show()
