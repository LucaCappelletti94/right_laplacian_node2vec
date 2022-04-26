from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen import Graph
from tqdm.auto import tqdm
import pandas as pd

def string_graph_normalization(graph: Graph) -> Graph:
    """Apply standard STRING PPI normalization."""
    return graph.filter_from_names(min_edge_weight=700)\
        .drop_singleton_nodes()\
        .sort_by_decreasing_outbound_node_degree()\
        .divide_edge_weights(1000.0)

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
            use_mirrored_strategy=False,
            embedding_method_kwargs=dict(
                max_neighbours=1000,
                iterations=10,
                epochs=50,
                return_weight=4.0,
                explore_weight=0.25,
                normalize_by_degree=normalize_by_degree
            ),
            graph_normalization_callback=string_graph_normalization
        )
        holdouts["normalization_name"] = normalization_name
        all_holdouts.append(holdouts)
    
    pd.concat(
        all_holdouts
    ).to_csv("right_laplacian_experiments.tsv", index=False)