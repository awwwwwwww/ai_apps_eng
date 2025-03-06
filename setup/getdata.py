#!/usr/bin/env python
"""This module downloads the dataset to be used for the example project."""

__author__ = "Adam Weissman"
__date__ = "4 February 2025"

import urllib.request as url
import zipfile as zp
import os

data_dir = "data"
data_file = "recipe-ingredients.zip"
download_url = "https://www.kaggle.com/api/v1/datasets/download/hugodarwood/epirecipes"

try:
    # Attempt to open the file, if file exists, skip download and unzipping
    with open(os.path.dirname(os.getcwd())+'\\'+data_dir+'\\'+data_file, 'r') as file:
      pass
except FileNotFoundError:
    # If open file failed, then file wasn't found, so download and unzip
    os.chdir(os.path.dirname(os.getcwd())+'\\'+data_dir)
    url.urlretrieve(download_url, data_file)
    with zp.ZipFile(data_file) as zip_ref:
        zip_ref.extractall()