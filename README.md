Description
===========

This repository contains the source code of method described in the paper: **Deep Learning Approach for Board-to-log Biometric Traceability by S. Shirodkar, D. Martinetto, P. Ngo, F. Longuetaud, F. Verjat and G. Pot**. 

## Requirements

Conda

Machine with a GPU

## Installation

Clone the repo and run the following from inside:

`conda env create -p ./spatial -f requirements.yml`

then activate the virtual environment using:

`conda activate ./spatial`

The codes `retrieve.py`, `localize.py` and `localize_classical.py` are standalone and can be run independently. To run in default config, simply run without CLI parameters. For example,

`python retrieve.py`

Tweaking algorithm parameters is possible inside the Config class inside each code. 

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
