# Right Laplacian Node2Vec
Experiments evaluating the performance of Right Laplacian weighting in Node2Vec sampling.

## Running the experiments
To run the experiments, you need to clone the repository, install the dependencies and execute the `run.py` python script.

### Clone the repository
To clone the repository run:

```bash
git clone https://github.com/LucaCappelletti94/right_laplacian_node2vec
```

### Installing the dependencies
After having cloned the repository, navigate in the directory and to install the dependecies, just run:

```bash
pip install pip -U
pip install -r requirements.txt
```

### Running the experiment
To retrieve the required datasets and run the experiment, execute the following command from your python shell:

```bash
python3 run.py
```

The runtime on a normal desktop computer is about 1-2 days.

All of our results are available in the [`right_laplacian_experiments.csv` document here](https://github.com/LucaCappelletti94/right_laplacian_node2vec/blob/main/right_laplacian_experiments.csv).

### Visualizing the results
The barplots results can be visualized by running the snipped in the [visualization jupyter notebook](https://github.com/LucaCappelletti94/right_laplacian_node2vec/blob/main/notebooks/Barplots%20visualization.ipynb):

```python
import pandas as pd
from barplots import barplots

_ = barplots(
    pd.read_csv("right_laplacian_experiments.csv"),
    groupby=["evaluation_type", "normalization_name", "unbalance"],
)
```