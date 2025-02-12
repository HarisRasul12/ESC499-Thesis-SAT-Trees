"""
=========== Module Description ===========

This module provides functions to derive distance classes from a given dataset by computing pairwise
Euclidean distances between data points. It groups pairs of points into ordered classes (labeled “D1”, “D2”, …)
such that any two distances in the same class differ by at most a specified tolerance (epsilon). These
distance classes serve as a crucial mathematical foundation in the SAT encoding for clustering: they enable
the formulation of constraints that either encourage data points with similar pairwise distances to be clustered
together or enforce their separation. The resulting ordered grouping is later integrated into the SAT model as
soft constraints to optimize intra-cluster cohesion and inter-cluster separation.
"""

from collections import OrderedDict
from itertools import combinations
from typing import Tuple, List

import numpy as np


def create_distance_classes(dataset: np.ndarray,
                            epsilon: float = 0) -> Tuple[
    OrderedDict[str, List[Tuple[Tuple[int, int], float]]], OrderedDict[str, List[Tuple[int, int]]], List[np.ndarray]]:
    """
    Constructs non-overlapping distance classes by grouping all pairs of data points whose Euclidean distances
    differ by no more than a given epsilon. This grouping creates ordered classes (labeled “D1”, “D2”, …) that
    serve as the foundation for enforcing clustering constraints: pairs in the same class are assumed to be
    similarly “close” (or “far”) and are later used to condition SAT clauses that encourage either co-clustering
    or separation, as dictated by the clustering objective.

    Args:
        dataset: The dataset containing n-dimensional data points.
        epsilon: The maximum difference between distances to consider them similar.

    Returns:
        A tuple containing:
          - An ordered dictionary mapping class labels (e.g., "D1", "D2", ...) to lists of (pair, distance) tuples.
          - An ordered dictionary mapping class labels to lists of point index pairs.
          - A list of arrays, each array containing the point index pairs for a distance class.
    """

    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    distances = {}
    for (idx1, point1), (idx2, point2) in combinations(enumerate(dataset), 2):
        dist = euclidean_distance(point1, point2)
        distances[(idx1, idx2)] = dist

    sorted_distances = sorted(distances.items(), key=lambda item: item[1])

    distance_classes_with_dist = OrderedDict()
    distance_classes_simplified = OrderedDict()
    current_class_label = 1
    for (pair, dist) in sorted_distances:
        placed = False
        for d_class, pairs in distance_classes_with_dist.items():
            class_dist = next(iter(pairs))[1]  # Get the reference distance for this class
            if abs(class_dist - dist) <= epsilon:
                distance_classes_with_dist[d_class].append((pair, dist))
                distance_classes_simplified[d_class].append(pair)
                placed = True
                break
        if not placed:
            distance_classes_with_dist[f'D{current_class_label}'] = [(pair, dist)]
            distance_classes_simplified[f'D{current_class_label}'] = [pair]
            current_class_label += 1

    distance_pairs_array = [np.array(pairs) for pairs in distance_classes_simplified.values()]
    distance_classes = distance_pairs_array

    return distance_classes_with_dist, distance_classes_simplified, distance_classes
