import silence_tensorflow.auto
from grape.edge_prediction import edge_prediction_evaluation
from grape.edge_prediction import DecisionTreeEdgePrediction, RandomForestEdgePrediction, GradientBoostingEdgePrediction
from grape.embedders import WalkletsSkipGramEnsmallen, Node2VecSkipGramEnsmallen, DeepWalkSkipGramEnsmallen
from ensmallen import Graph
from tqdm.auto import tqdm
import pandas as pd


def string_graph_normalization(graph: Graph) -> Graph:
    """Apply standard STRING PPI normalization.
    
    Parameters
    --------------------
    graph: Graph
        The STRING PPI graph to normalize.
    """
    return graph.filter_from_names(min_edge_weight=700)\
        .remove_components(top_k_components=1)\
        .divide_edge_weights(1000.0)


def run_experiment(smoke_test: bool = False) -> pd.DataFrame:
    """Pipeline to execute the Degree Normalization experiments.
    
    Parameters
    --------------------
    smoke_test: bool
        Whether this run needs to be a smoke test.
    """
    all_holdouts = []
    for normalize_by_degree in tqdm(
        (True, False),
        leave=False,
        desc="Running Experiments",
        dynamic_ncols=True,
    ):
        for embedding_model in tqdm(
            (
                WalkletsSkipGramEnsmallen,
                Node2VecSkipGramEnsmallen,
                DeepWalkSkipGramEnsmallen
            ),
            leave=False,
            desc="Embedding",
            dynamic_ncols=True,
        ):
            performance = edge_prediction_evaluation(
                holdouts_kwargs=dict(
                    train_size=0.8,
                    minimum_node_degree=5,
                    maximum_node_degree=100,
                ),
                graphs=[
                    "HomoSapiens",
                    "DrosophilaMelanogaster",
                    "SaccharomycesCerevisiae",
                    "MusMusculus",
                    "SusScrofa",
                    "AmanitaMuscariaKoideBx008",
                    "AlligatorSinensis",
                    "CanisLupus"
                ],
                models=[
                    DecisionTreeEdgePrediction(),
                    RandomForestEdgePrediction(),
                    GradientBoostingEdgePrediction()
                ],
                evaluation_schema="Connected Monte Carlo",
                number_of_holdouts=10,
                node_features=embedding_model(
                    normalize_by_degree=normalize_by_degree,
                    enable_cache=True
                ),
                graph_callback=string_graph_normalization,
                smoke_test=smoke_test,
                enable_cache=True,
                verbose=True
            )
            all_holdouts.append(performance)

    results = pd.concat(
        all_holdouts
    )
    
    results.to_csv("degree_normalization_experiments.csv", index=False)

    return results