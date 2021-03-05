# hpa-scc
[Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification)

## Python Scripts

|File|Description|
|---------------------|-------------------------------------------------------------------------------------------------|
||Production||
|descriptions.csv|Text labels for classes|
|hpa-scc.wpr|Python project|
|logs.py|Analyze log files from training/testing|
|segment.py|Segment images using HPA cellsegmentator|
|split.py|Partition data into training and validation|
|train.py|Build and train neual network|
|visualize.py|Visualize image files|
||Spikes||
|anayze-labels|Find images  that have only one label|
|CellSegmenterTest.py|Demo for HPA cellsegmentator|
|encoding.py|Example of encoding|
|otsu.py|Segment images using Otsu's method|
|spike2.py|Estimate memory usage for down-sampled data, and loading and saving times|
|spike3.py|Train network using data daved by spike2.py|
|watershed.py|Naive Watershed segmentation|
|download.py|Download training images from kaggle|
|summarize.py|Organize image names by description|
|unzip.py|Unzip downloaded files|


## Documentation

|File|Description|
|-----------------|-------------------------------------------------------------------------------------------------|
|hpa-scc.bib|Bibliography|
|Notes.tex|My notes|
