# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing import List

from dataClass import LineEvaluationResult
from promptflow.core import log_metric, tool


@tool
def aggregate(processed_results: List[str]):
    """
    This tool aggregates the processed result of all lines and calculate the accuracy. Then log metric for the accuracy.

    :param processed_results: List of the output of line_process node.
    """

    # Initialize totals
    total_retrieve_context_relevance_score = 0
    total_groundedness_score = 0
    total_relevance_score = 0

    # Iterate over each item in the data
    for item in processed_results:
        # Add values to totals, converting them to float
        total_retrieve_context_relevance_score += float(item['retrieve_context_relevance_score'])
        total_groundedness_score += float(item['groundedness_score'])
        total_relevance_score += float(item['relevance_score'])

    # Calculate averages
    aggregated_retrieve_context_relevance_score = total_retrieve_context_relevance_score / len(processed_results)
    aggregated_groundedness_score = total_groundedness_score / len(processed_results)
    aggregated_relevance_score = total_relevance_score / len(processed_results)

    # Log metric the aggregate result
    log_metric(key="retrieve_context_relevance_score", value=aggregated_retrieve_context_relevance_score)
    log_metric(key="groundedness_score", value=aggregated_groundedness_score)
    log_metric(key="relevance_score", value=aggregated_relevance_score)

    
    
    return {"retrieve_context_relevance_score": aggregated_retrieve_context_relevance_score, "groundedness_score": aggregated_groundedness_score,"relevance_score":aggregated_relevance_score}
