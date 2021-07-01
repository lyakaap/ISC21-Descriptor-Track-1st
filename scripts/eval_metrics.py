"""Evaluation script for the Facebook AI Image Similarity Challenge. This script has a command-line
interface (CLI). See CLI documentation with: `python eval_metrics.py --help`. Note that this script
does not contain any of the input validation on submission files that the competition platform will
do (e.g., validating shape of submission, validating ID values)."""

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import typer


def argsort(seq: Sequence[Any]):
    """Like np.argsort but for 1D sequences. Based on https://stackoverflow.com/a/3382369/3853462"""
    return sorted(range(len(seq)), key=seq.__getitem__)


def precision_recall(
    y_true: np.ndarray, probas_pred: np.ndarray, num_positives: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precisions, recalls, and thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Binary label of each prediction (0 or 1). Shape [n, k] or [n*k, ]
    probas_pred : np.ndarray
        Score of each prediction (higher score == images more similar, ie not a distance)
        Shape [n, k] or [n*k, ]
    num_positives : int
        Number of positives in the groundtruth.

    Returns
    -------
    precisions, recalls, thresholds
        ordered by increasing recall, as for a precision-recall curve
    """
    probas_pred = probas_pred.flatten()
    y_true = y_true.flatten()
    # to handle duplicates scores, we sort (score, NOT(jugement)) for predictions
    # eg,the final order will be (0.5, False), (0.5, False), (0.5, True), (0.4, False), ...
    # This allows to have the worst possible AP.
    # It prevents participants from putting the same score for all predictions to get a good AP.
    order = argsort(list(zip(probas_pred, ~y_true)))
    order = order[::-1]  # sort by decreasing score
    probas_pred = probas_pred[order]
    y_true = y_true[order]

    ntp = np.cumsum(y_true)  # number of true positives <= threshold
    nres = np.arange(len(y_true)) + 1  # number of results

    precisions = ntp / nres
    recalls = ntp / num_positives
    return precisions, recalls, probas_pred


def average_precision(recalls: np.ndarray, precisions: np.ndarray):
    """
    Compute the micro-average precision score (μAP).

    Parameters
    ----------
    recalls : np.ndarray
        Recalls. Must be sorted by increasing recall, as in a PR curve.
    precisions : np.ndarray
        Precisions for each recall value.

    Returns
    -------
    μAP: float
    """

    # Check that it's ordered by increasing recall
    if not np.all(recalls[:-1] <= recalls[1:]):
        raise ValueError("recalls array must be sorted before passing in")
    return ((recalls - np.concatenate([[0], recalls[:-1]])) * precisions).sum()


def find_operating_point(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, required_x: float
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Find the highest y (and corresponding z) with x at least `required_x`.

    Returns
    -------
    x, y, z
        The best operating point (highest y) with x at least `required_x`.
        If we can't find a point with the required x value, return
        x=required_x, y=None, z=None
    """
    valid_points = x >= required_x
    if not np.any(valid_points):
        return required_x, None, None

    valid_x = x[valid_points]
    valid_y = y[valid_points]
    valid_z = z[valid_points]
    best_idx = np.argmax(valid_y)
    return valid_x[best_idx], valid_y[best_idx], valid_z[best_idx]


def evaluate_metrics(submission_df: pd.DataFrame, gt_df: pd.DataFrame):
    """Given a matching track submission dataframe and a ground truth dataframe,
    compute the competition metrics."""

    # Subset submission_df to query_ids that we have labels for in gt_df
    submission_df = submission_df[submission_df["query_id"].isin(gt_df["query_id"])]

    gt_pairs = {
        tuple(row)
        for row in gt_df[["query_id", "reference_id"]].itertuples(index=False)
        if not pd.isna(row.reference_id)
    }

    # Binary indicator for whether prediction is a true positive or false positive
    y_true = np.array(
        [
            tuple(row) in gt_pairs
            for row in submission_df[["query_id", "reference_id"]].itertuples(index=False)
        ]
    )
    # Confidence score, as if probability. Only property required is greater score == more confident.
    probas_pred = submission_df["score"].values

    p, r, t = precision_recall(y_true, probas_pred, len(gt_pairs))

    # Micro-average precision
    ap = average_precision(r, p)

    # Metrics @ Precision>=90%
    pp90, rp90, tp90 = find_operating_point(p, r, t, required_x=0.9)

    if rp90 is None:
        # Precision was never above 90%
        rp90 = 0.0

    return ap, rp90


def load_descriptor_h5(descs_submission_path):
    """Load datasets from descriptors submission hdf5 file."""
    import h5py

    with h5py.File(descs_submission_path, "r") as f:
        query = f["query"][:]
        reference = f["reference"][:]
        # Coerce IDs to native Python unicode string no matter what type they were before
        query_ids = np.array(f["query_ids"][:], dtype=object).astype(str).tolist()
        reference_ids = np.array(f["reference_ids"][:], dtype=object).astype(str).tolist()
    return query, reference, query_ids, reference_ids


def query_iterator(xq: np.ndarray):
    """Produces batches of progressively increasing sizes."""
    nq = len(xq)
    bs = 32
    i = 0
    while i < nq:
        xqi = xq[i: i + bs]  # noqa: E203
        yield xqi
        if bs < 20_000:
            bs *= 2
        i += len(xqi)


def search_with_capped_res(xq: np.ndarray, xb: np.ndarray, num_results: int):
    """Searches xq (queries) into xb (reference), with a maximum total number of results."""
    import faiss
    from faiss.contrib import exhaustive_search
    import torch

    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)

    ngpu = -1 if xb.shape[1] <= 2048 and 'A100' not in torch.cuda.get_device_name() else 0
    radius, lims, dis, ids = exhaustive_search.range_search_max_results(
        index,
        query_iterator(xq),
        1e10,  # initial radius is arbitrary
        max_results=2 * num_results,
        min_results=num_results,
        ngpu=ngpu,  # use GPU if available
    )
    assert len(ids) > 0

    n = len(dis)
    nq = len(xq)
    if n > num_results:
        # crop to num_results exactly
        o = dis.argpartition(num_results)[:num_results]
        mask = np.zeros(n, bool)
        mask[o] = True
        new_dis = dis[mask]
        new_ids = ids[mask]
        nres = [0] + [mask[lims[i]: lims[i + 1]].sum() for i in range(nq)]  # noqa: E203
        new_lims = np.cumsum(nres)
        lims, dis, ids = new_lims, new_dis, new_ids

    return lims, dis, ids


def get_matching_from_descs(
    query: np.ndarray,
    reference: np.ndarray,
    query_ids: List[str],
    reference_ids: List[str],
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conduct similarity search and convert results and distances into matching submission format.

    Parameters
    ----------
    query : np.ndarray
        2D array of query descriptor vectors
    reference : np.ndarray
        2D array of reference descriptor vectors
    query_ids : List[str]
        query image IDs corresponding to vectors in query
    reference_ids : List[str]
        reference image IDs corresponding to vectors in reference
    num_results : int
        maximum number of results from similarity search

    Returns
    -------
    pandas dataframe of results in matching submission format
    """
    # Subset search to query_ids that we have labels for in gt_df
    query_gt_mask = np.isin(query_ids, gt_df["query_id"])
    query_ids = np.array(query_ids)[query_gt_mask]
    query = query[query_gt_mask]

    nq = len(query)
    lims, dis, ids = search_with_capped_res(query, reference, num_results=nq * 10)

    matching_submission_df = pd.DataFrame(
        {"query_id": query_ids[i], "reference_id": reference_ids[ids[j]], "score": -dis[j]}
        for i in range(nq)
        for j in range(lims[i], lims[i + 1])
    )

    return matching_submission_df


def main(
    submission_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to submission file. (CSV for matching track, HDF5 descriptor track)",
    ),
    gt_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to ground truth CSV file.",
    ),
    is_matching: Optional[bool] = typer.Option(
        None,
        "--matching/--descriptor",
        "-m/-d",
        help="Indicate which competition track that the submission file is for. "
        "By default, will infer from file extension.",
    ),
):
    """Evaluate competition metrics for a submission to the Facebook AI Image Similarity
    Challenge. Note that this script does not contain any of the input validation on submission
    files that the competition platform will do (e.g., validating shape of submission, validating
    ID values)."""
    if is_matching is None:
        # Infer which type submission file is
        if submission_path.suffix.lower() == ".csv":
            is_matching = True
        elif submission_path.suffix.lower() in {".hdf", ".h5", ".hdf5"}:
            is_matching = False
        else:
            typer.echo("Unable to infer track from file extension. Please specify explicitly.")
            raise typer.Exit(code=1)

    gt_df = pd.read_csv(gt_path)

    if is_matching:
        submission_df = pd.read_csv(submission_path)
    else:
        query, reference, query_ids, reference_ids = load_descriptor_h5(submission_path)
        submission_df = get_matching_from_descs(query, reference, query_ids, reference_ids, gt_df)
        submission_df.to_csv(submission_path.parent / (submission_path.stem + '.csv'), index=False)

    ap, rp90 = evaluate_metrics(submission_df, gt_df)

    typer.echo(
        json.dumps(
            {
                "average_precision": ap,
                "recall_p90": rp90,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    typer.run(main)
