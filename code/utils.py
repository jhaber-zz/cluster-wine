#!/usr/bin/env python
# coding: utf-8

"""
@title: Utility functions for data processing and analysis
@author: Jaren Haber, PhD
@date: September 16, 2023

"""


###############################################
# Initialize                 
###############################################

# Import packages
import requests
import shutil
from os.path import join


###############################################
# Define function(s)                 
###############################################

def get_unzip(URL, fpath):
    """
    Downloads zip-formatted file from specified URL and extracts it.
    
    Args:
        URL (str) -- points to zipped file
        fpath (str) -- path to folder in which to save output
    Returns:
        extracted zip file in main directory (one level up)
        (returns nothing directly)
        
    """
    
    zipped = requests.get(URL)
    zipped_bytes = zipped.content
    
    fname = URL.split('/')[-1] # Get name for extraction folder using URL
    outfile = join(fpath, fname)
    with open(outfile, "wb") as f:
        f.write(zipped_bytes)

    shutil.unpack_archive(outfile, extract_dir = fpath, format = 'zip')
    
    return