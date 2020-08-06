# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Download Flickr-Faces-HQ (FFHQ) dataset to current working directory."""

import os
import sys
import requests
import html
import hashlib
# import PIL.Image
# import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict

# PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # avoid "Decompressed Data Too Large" error

#----------------------------------------------------------------------------

json_spec = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842, file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')

tfrecords_specs = [
    dict(file_url='https://drive.google.com/uc?id=1LnhoytWihRRJ7CfhLQ76F8YxwxRDlZN3', file_path='tfrecords/ffhq/ffhq-r02.tfrecords', file_size=6860000,      file_md5='63e062160f1ef9079d4f51206a95ba39'),
    dict(file_url='https://drive.google.com/uc?id=1LWeKZGZ_x2rNlTenqsaTk8s7Cpadzjbh', file_path='tfrecords/ffhq/ffhq-r03.tfrecords', file_size=17290000,     file_md5='54fb32a11ebaf1b86807cc0446dd4ec5'),
    dict(file_url='https://drive.google.com/uc?id=1Lr7Tiufr1Za85HQ18yg3XnJXstiI2BAC', file_path='tfrecords/ffhq/ffhq-r04.tfrecords', file_size=57610000,     file_md5='7164cc5531f6828bf9c578bdc3320e49'),
    dict(file_url='https://drive.google.com/uc?id=1LnyiayZ-XJFtatxGFgYePcs9bdxuIJO_', file_path='tfrecords/ffhq/ffhq-r05.tfrecords', file_size=218890000,    file_md5='050cc7e5fd07a1508eaa2558dafbd9ed'),
    dict(file_url='https://drive.google.com/uc?id=1Lt6UP201zHnpH8zLNcKyCIkbC-aMb5V_', file_path='tfrecords/ffhq/ffhq-r06.tfrecords', file_size=864010000,    file_md5='90bedc9cc07007cd66615b2b1255aab8'),
    dict(file_url='https://drive.google.com/uc?id=1LwOP25fJ4xN56YpNCKJZM-3mSMauTxeb', file_path='tfrecords/ffhq/ffhq-r07.tfrecords', file_size=3444980000,   file_md5='bff839e0dda771732495541b1aff7047'),
    dict(file_url='https://drive.google.com/uc?id=1LxxgVBHWgyN8jzf8bQssgVOrTLE8Gv2v', file_path='tfrecords/ffhq/ffhq-r08.tfrecords', file_size=13766900000,  file_md5='74de4f07dc7bfb07c0ad4471fdac5e67'),
    dict(file_url='https://drive.google.com/uc?id=1M-ulhD5h-J7sqSy5Y1njUY_80LPcrv3V', file_path='tfrecords/ffhq/ffhq-r09.tfrecords', file_size=55054580000,  file_md5='05355aa457a4bd72709f74a81841b46d'),
    dict(file_url='https://drive.google.com/uc?id=1M11BIdIpFCiapUqV658biPlaXsTRvYfM', file_path='tfrecords/ffhq/ffhq-r10.tfrecords', file_size=220205650000, file_md5='bf43cab9609ab2a27892fb6c2415c11b'),
]

license_specs = {
    'json':      dict(file_url='https://drive.google.com/uc?id=1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX', file_path='LICENSE.txt',                    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'images':    dict(file_url='https://drive.google.com/uc?id=1sP2qz8TzLkzG2gjwAa4chtdB31THska4', file_path='images1024x1024/LICENSE.txt',    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'thumbs':    dict(file_url='https://drive.google.com/uc?id=1iaL1S381LS10VVtqu-b2WfF9TiY75Kmj', file_path='thumbnails128x128/LICENSE.txt',  file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'wilds':     dict(file_url='https://drive.google.com/uc?id=1rsfFOEQvkd6_Z547qhpq5LhDl2McJEzw', file_path='in-the-wild-images/LICENSE.txt', file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'tfrecords': dict(file_url='https://drive.google.com/uc?id=1SYUmqKdLoTYq-kqsnPsniLScMhspvl5v', file_path='tfrecords/ffhq/LICENSE.txt',     file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
}

#----------------------------------------------------------------------------

def download_file(session, file_spec, stats, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)
                        # with stats['lock']:
                            stats['bytes_done'] += len(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            if 'pixel_size' in file_spec or 'pixel_md5' in file_spec:
                with PIL.Image.open(tmp_path) as image:
                    if 'pixel_size' in file_spec and list(image.size) != file_spec['pixel_size']:
                        raise IOError('Incorrect pixel size', file_path)
                    if 'pixel_md5' in file_spec and hashlib.md5(np.array(image)).hexdigest() != file_spec['pixel_md5']:
                        raise IOError('Incorrect pixel MD5', file_path)
            break

        except:
            # with stats['lock']:
            stats['bytes_done'] -= data_size

            # Handle known failure cases.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                data_str = data.decode('utf-8')

                # Google Drive virus checker nag.
                links = [html.unescape(link) for link in data_str.split('"') if 'export=download' in link]
                if len(links) == 1:
                    if attempts_left:
                        file_url = requests.compat.urljoin(file_url, links[0])
                        continue

                # Google Drive quota exceeded.
                if 'Google Drive - Quota exceeded' in data_str:
                    if not attempts_left:
                        raise IOError("Google Drive download quota exceeded -- please try again later")

            # Last attempt => raise error.
            if not attempts_left:
                raise

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic
    # with stats['lock']:
    #     stats['files_done'] += 1

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

#----------------------------------------------------------------------------

def choose_bytes_unit(num_bytes):
    b = int(np.rint(num_bytes))
    if b < (100 << 0): return 'B', (1 << 0)
    if b < (100 << 10): return 'kB', (1 << 10)
    if b < (100 << 20): return 'MB', (1 << 20)
    if b < (100 << 30): return 'GB', (1 << 30)
    return 'TB', (1 << 40)

#----------------------------------------------------------------------------

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60: return '%ds' % s
    if s < 60 * 60: return '%dm %02ds' % (s // 60, s % 60)
    if s < 24 * 60 * 60: return '%dh %02dm' % (s // (60 * 60), (s // 60) % 60)
    if s < 100 * 24 * 60 * 60: return '%dd %02dh' % (s // (24 * 60 * 60), (s // (60 * 60)) % 24)
    return '>100d'


if __name__ == "__main__":
    file_spec = {}
    file_spec['file_path'] = "ffhq/ffhq-r10.tfrecords"
    file_spec['file_url'] = 'https://drive.google.com/uc?id=1LxxgVBHWgyN8jzf8bQssgVOrTLE8Gv2v'
    stats = 13766900000
    with requests.Session() as session:
        download_file(session, file_spec, stats)

#----------------------------------------------------------------------------