## Scaling

Images should be scaled according to the models as 224x224. 

## Data organization

Three files are required for training which should consist of compressed archives of PNG or JPEG formatted images and a train_stats.json file. 
  
```json
{
    "mean":[
        0.4180903962661585,
        0.5001509693331928,
        0.4003405137983626
    ],
    "std":[
        0.008093840050943024,
        0.007145403886840386,
        0.0071429898778710655
    ],
    "total_concepts":{
        "Benthocodon":596,
        "Coryphaenoides":6,
        "Cystechinus_loveni":160,
        "Echinocrepis_rostrata":76,
        "Elpidia":392,
        ...
    }
}
``` 


Copy this to the same folder as your training data, e.g.

~~~
│   └── data
│       ├── train.tar.gz
│       ├── val.tar.gz
│       ├── train_stats.json
~~~


## Stats file

Data statistics need to be captured for normalization which is used during training and later inference/prediction.
These should be captured in a conf.json file in the same directory as the training data. 
An example python snippet to capture that

```python

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
    stats = compute_statistics(Path.cwd() / 'data' / 'catsdogsval', Path.cwd() / 'train_stats.json')

```
