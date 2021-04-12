# hpa-scc
[Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification)

## Python Scripts--new architecture

|File|Description|
|---------------------|-------------------------------------------------------------------------------------------------|
|census.py|Compute statistics for number of images for combinations of classes|
|descriptions.csv|Text labels for classes|
|dirichlet.py|Segment using dirichlet process means|
|establish-N.R|R script used to establish value for maximum number of iterations|
|hpa-scc.wpr|Python project|
|hpascc.py|Common code - e.g. read list of descriptions, image-ids|
|rgb.txt|List of xkcd colours|
|select_images.py|Prepare worklist for pipeline|
|slice2.py|Slice and downsample dataset|
|utils.py|Utility classes: Timer and Log|

## Python Scripts--old architecture

|File|Description|
|---------------------|-------------------------------------------------------------------------------------------------|
|logs.py|Analyze log files from training/testing|
|segment.py|Segment images using HPA cellsegmentator|
|slice.py|Slice and downsample dataset|
|train2.py|Build and train neural network using sliced data|
|visualize.py|Visualize image files|
|vizcnn.py|Visualize CNN filters|

## Documentation

|File|Description|
|-----------------|-------------------------------------------------------------------------------------------------|
|hpa-scc.bib|Bibliography|
|Notes.tex|My notes|
