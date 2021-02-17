# snarfed from https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
import glob
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

save_dir = 'save'   #FIXME
mt       = glob.glob(save_dir + '/' + '*_red.png')
er       = [f.replace('red', 'yellow') for f in mt]
nu       = [f.replace('red', 'blue') for f in mt]
images   = [mt, er, nu]


NUC_MODEL = "./nuclei-model.pth"
CELL_MODEL = "./cell-model.pth"
segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device="cpu",    #FIXME
    padding=False,
    multi_channel_model=True,
)

# For nuclei
nuc_segmentations = segmentator.pred_nuclei(images[2])

# For full cells
cell_segmentations = segmentator.pred_cells(images)

# post-processing
for i, pred in enumerate(cell_segmentations):
    nuclei_mask, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i])
    FOVname = os.path.basename(mt[i]).replace('red','predictedmask')
    imageio.imwrite(os.path.join(save_dir,FOVname), cell_mask)



fig, ax = plt.subplots(1,3, figsize=(20,50))
for i in range(len(mt)):                    # FIXME
    microtubule = plt.imread(mt[i])
    endoplasmicrec = plt.imread(er[i])
    nuclei = plt.imread(nu[i])
    mask = plt.imread(mt[i].replace('red','predictedmask'))
    img = np.dstack((microtubule, endoplasmicrec, nuclei))
    ax[i].imshow(img)
    ax[i].imshow(mask, alpha=0.5)
    ax[i].axis('off')
plt.show()
