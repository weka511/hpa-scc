# hpa-scc
[Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification)

## Python Scripts

|File|Description|
|---------------------|-------------------------------------------------------------------------------------------------|
|descriptions.csv|Text labels for classes|
|dirichlet.py|Segment using dirichlet process means|
|hpa-scc.wpr|Python project|
|logs.py|Analyze log files from training/testing|
|rgb.txt|List of xkcd colours|
|segment.py|Segment images using HPA cellsegmentator|
|slice.py|Slice and downsample dataset|
|train2.py|Build and train neural network using sliced data|
|utils.py|Utility classes: Timer and Log|
|visualize.py|Visualize image files|
|vizcnn.py|Visualize CNN filters|

## Documentation

|File|Description|
|-----------------|-------------------------------------------------------------------------------------------------|
|hpa-scc.bib|Bibliography|
|Notes.tex|My notes|

## Spike solutions and obsolete code

|File|Description|
|---------------------|-------------------------------------------------------------------------------------------------|
|analyze-labels|Find images  that have only one label|
|CellSegmenterTest.py|Demo for HPA cellsegmentator|
|encoding.py|Example of encoding|
|otsu.py|Segment images using Otsu's method|
|split.py|Partition data into training and validation|
|spike2.py|Estimate memory usage for down-sampled data, and loading and saving times|
|spike3.py|Train network using data daved by spike2.py|
|spike4.py|Understand how to make imshow and scatter consistent|
|train.py|Build and train neural network|
|watershed.py|Naive Watershed segmentation|
|download.py|Download training images from kaggle|
|summarize.py|Organize image names by description|
|unzip.py|Unzip downloaded files|
