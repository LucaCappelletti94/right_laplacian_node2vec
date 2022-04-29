from typing import Optional
from downloaders import BaseDownloader
import compress_json
import gzip
import shutil
import os
import numpy as np
import pandas as pd
from ensmallen import Graph


def retrieve_coo_graph(
    graph_name: str,
    settings_graph_name: Optional[str] = None
) -> Graph:
    """Retrieves and processes the required file.

    Parameters
    ---------------------
    graph_name: str
        The required graph name
    settings_graph_name: Optional[str] = None
        The settings graph name
    """
    if settings_graph_name is None:
        settings_graph_name = graph_name
    urls = compress_json.local_load("internet_archive_urls.json")
    url = urls[graph_name]

    graph_name_lower = graph_name.lower()

    downloader = BaseDownloader(verbose=2)
    downloader.download(url)

    compressed_edge_list = f"downloads/{graph_name}/{graph_name}/{graph_name_lower}_edge_list.npy.gz"
    numpy_edge_list = f"downloads/{graph_name}/{graph_name}/{graph_name_lower}_edge_list.npy"
    tsv_edge_list = f"downloads/{graph_name}/{graph_name}/{graph_name_lower}_edge_list.tsv"
    edge_types = f"downloads/{graph_name}/{graph_name}/{graph_name_lower}_edge_types.csv"
    compressed_node_list = f"downloads/{graph_name}/{graph_name}/{graph_name_lower}_node_list.tsv.gz"
    node_list = f"downloads/{graph_name}/{graph_name}/{graph_name_lower}_node_list.tsv"

    for source, destination in (
        (compressed_edge_list, numpy_edge_list),
        (compressed_node_list, node_list)
    ):
        if not os.path.exists(destination):
            with gzip.open(source, 'rb') as f_in:
                with open(destination, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    if not os.path.exists(tsv_edge_list):
        edge_list = np.load(numpy_edge_list)
        pd.DataFrame(
            edge_list,
            columns=["subject", "object", "edge_type"],
        ).to_csv(tsv_edge_list, sep="\t", index=False)

    graph = Graph.from_csv(
        directed=False,
        node_path=node_list,
        nodes_column="node_name",
        node_types_separator="\t",
        node_list_node_types_column="node_type",
        load_node_list_in_parallel=False,
        edge_type_path=edge_types,
        edge_types_column="edge_type_name",
        edge_type_list_separator=",",
        load_edge_type_list_in_parallel=False,
        edge_path=tsv_edge_list,
        edge_list_separator="\t",
        sources_column="subject",
        destinations_column="object",
        edge_list_edge_types_column="edge_type",
        edge_list_numeric_node_ids=True,
        edge_list_is_correct=True,
        edge_list_numeric_edge_type_ids=True,
        verbose=True,
        name=graph_name
    )
    return graph


def retrieve_coo_ctd() -> Graph:
    """Return instance of CTD graph."""
    return retrieve_coo_graph("CTD")