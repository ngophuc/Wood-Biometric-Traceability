# README.md

## Dependencies

`Python 3.10.20` 

`torch==2.5.1` 

`torchvision==0.20.1`

`numpy==1.26.4`

`timm==1.0.26`

`pillow==12.2.0`

`tqdm==4.67.3`

## Installation

Download the folder and ensure the above dependencies are available. The programs can be run from inside this folder.

## Data

`ducret-autun-t1-orig` : database of source logs, 1 image per log

`ducret-autun-t1-virtual-sawn` : virtual cuts (intended for the aged logs to obtain boards with appearance change) applied to fresh logs 

`ducret-autun-t2-cut-align-full-similarity` : aged logs aligned correctly (in scale, position, orientation etc) to their fresh log counterparts using sparse matching → RANSAC → estimated affine transform

`ducret-autun-t2-virtual-sawn-full-similarity` : virtually sawn board cuts

## Instructions

Each file is named explicitly as per its purpose.

`retrieve.py` takes every virtually sawed board and searches for its parent log in the database of logs in `ducret-autun-t1-orig`.

`localize.py` takes every virtually sawed board and localizes it inside its parent log using the authors’ method.

`localize_classical.py` does the same as `localize.py` except using the classical baseline method inspired from Xiaolin Li et al.

`virtual_saw.py` makes independent virtual board cuts for experiments. To keep working with the authors’ virtual cut, avoid running this program and merely use it as reference.

Every code is run without having any CLI parameters. For example, `python3 retrieve.py` to obtain results for board-to-log retrieval. Running as is uses default arguments. Tweaking parameters is possible inside the Config class inside each code. 
