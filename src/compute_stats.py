# !/usr/bin/env python
__author__ = "Danelle Cline"
__copyright__ = "Copyright 2023, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Compute image statistics

@author: __author__
@status: __status__
@license: __license__
'''

from pathlib import Path
import json
import numpy as np
import codecs
import progressbar
from PIL import Image, ImageStat
assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

def compute_statistics(data_dir: Path, stats_file:Path):
    """
    This computes statistics for a collection of images
    :param data_dir: Absolute path to the directory with .png files
    :param stats_file: Absolute path to the filename to store the statistics
    :return: statistics dictionary
    """
    if not data_dir.exists():
        print(f'{data_dir} does not exist')
        return

    sum_frames = {}
    sum_concepts = {}
    means = []
    stds = []
    mean = np.array([])
    std = np.array([])

    for g in progressbar.progressbar(sorted(data_dir.glob('**/*.jpg')), prefix=f'Computing statistics for {data_dir} : '):
        g_path = Path(g)
        c = g_path.parent.name
        img = Image.open(g_path.as_posix())
        stat = ImageStat.Stat(img)
        mean = stat.mean
        std = stat.stddev
        means.append(mean)
        stds.append(std)

        if c not in sum_concepts.keys():
            sum_concepts[c] = 1
        else:
            sum_concepts[c] += 1


    if len(means) > 0:
        mean = np.mean(means, axis=(0)) / 255.
        std = np.std(stds, axis=(0)) / 255.

    print(f'Writing {stats_file}')
    json.dump({'total_concepts': sum_concepts, 'mean': mean.tolist(), 'std': std.tolist()},
              codecs.open(stats_file.as_posix(), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    return {'total_frames': sum_frames, 'total_concepts': sum_concepts, 'mean': mean.tolist(), 'std': std.tolist()}



if __name__ == '__main__':
    import pandas as pd
    stats = compute_statistics(Path.cwd() / 'data' / 'catsdogsval', Path.cwd() / 'stats.json')


