# Training a classification model with dogs and cats

This simple example will train for 2 epochs locally on a default data set of dogs and cats. 

# Setup a minimal python environment

```
git clone https://github.com/mbari-org/kclassify.git
cd kclassify && conda env create 
conda activate kclassify
```

# Run

A small collection of data is included in the repo. To train on this data, run the following command:

```
python src/train.py 
``` 

The training data and other artifacts that are generated during training in the following organization:

```
~~~
│   └── kclassify
│       └── data/ 
│                   └──train_stats.json (optional)
│                   └──catsdogstrain.tar.gz 
│                   └──catsdogsval.tar.gz
│       └── checkpoint/  
...
│                   └──0f110283-1d0d-41ed-a336-b997bfec0658/
│                   └──1c264240-62b4-4342-9f31-80b6a5d69b14/
│       └── model/  
```


Available options for hyperparameters can be found [here](arguments.md)

