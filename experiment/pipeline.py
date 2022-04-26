from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from tqdm.auto import tqdm
import pandas as pd

def run_experiment():
    """Pipeline to execute the right laplacian experiments."""
    all_holdouts = []
    for normalization_name, normalize_by_degree in tqdm(
        (
            ("Traditional", False),
            ("Right Laplacian", True),
        ),
        leave=False,
        desc="Running Experiments",
        dynamic_ncols=True,
    ):
        holdouts, _ = evaluate_embedding_for_edge_prediction(
            embedding_method="CBOW",
            graphs=[
                "HomoSapiens",
                "DrosophilaMelanogaster",
                "ArabidopsisThaliana",
                "SaccharomycesCerevisiae",
                "RattusNorvegicus",
                "MusMusculus",
                "SusScrofa",
                "AmanitaMuscariaKoideBx008",
                "AlligatorSinensis",
                "CanisLupus"
            ],
            model_name="Perceptron",
            use_only_cpu=True,
            embedding_method_kwargs=dict(
                max_neighbours=1000,
                iterations=10,
                epochs=50,
                return_weight=4.0,
                explore_weight=0.25,
                normalize_by_degree=normalize_by_degree
            )
        )
        holdouts["normalization_name"] = normalization_name
        all_holdouts.append(holdouts)
    
    pd.concat(
        all_holdouts
    ).to_csv("right_laplacian_experiments.tsv", index=False)