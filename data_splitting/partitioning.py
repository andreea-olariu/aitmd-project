"""Partitioning strategies used to simulate (non-)IID federated clients.

Every function receives the labels of the samples (and, where needed, a
"group" per sample, e.g. the subject / camera / hospital that produced it)
and returns a list with one array of sample indices per client.

All functions are pure NumPy, so you can import them in your own code:

    from partitioning import dirichlet_split
    parts = dirichlet_split(labels, num_clients=3, alpha=0.3, seed=42)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. IID – the "ideal" (and unrealistic) baseline
# ---------------------------------------------------------------------------
def iid_split(labels: np.ndarray, num_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Shuffle all samples and give every client an equal-sized random chunk."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    return [np.sort(part) for part in np.array_split(indices, num_clients)]


# ---------------------------------------------------------------------------
# 2. Label-distribution skew with a Dirichlet prior
# ---------------------------------------------------------------------------
def dirichlet_split(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    seed: int = 42,
    max_retries: int = 100,
) -> list[np.ndarray]:
    """For every class, draw client proportions p ~ Dir(alpha) and deal the
    samples of that class accordingly.

    * alpha -> infinity : every client sees the same label distribution (IID)
    * alpha = 1.0       : moderate skew
    * alpha <= 0.1      : extreme skew, most clients see only 1-2 classes
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    for _ in range(max_retries):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = rng.permutation(np.where(labels == c)[0])
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cut_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(idx_c, cut_points)):
                client_indices[client_id].extend(chunk.tolist())
        if min(len(ci) for ci in client_indices) >= min_samples:
            return [np.sort(np.array(ci, dtype=int)) for ci in client_indices]

    raise RuntimeError(
        f"Could not give every client >= {min_samples} samples after {max_retries} "
        "tries. Increase --alpha, decrease --min-samples or use fewer clients."
    )


# ---------------------------------------------------------------------------
# 3. Pathological split – every client only owns K classes
# ---------------------------------------------------------------------------
def pathological_split(
    labels: np.ndarray,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
) -> list[np.ndarray]:
    """Each client receives samples from exactly `classes_per_client` classes
    (McMahan et al., 2017). Classes are assigned round-robin so that every
    class is owned by at least one client whenever that is possible; the
    samples of a class are then shared equally between its owners."""
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))
    if classes_per_client > len(classes):
        raise ValueError(
            f"classes_per_client={classes_per_client} > number of classes ({len(classes)})"
        )

    # Round-robin assignment: client i gets classes i*K, i*K+1, ... (mod C)
    owners: dict[int, list[int]] = {int(c): [] for c in classes}
    for client_id in range(num_clients):
        for k in range(classes_per_client):
            c = int(classes[(client_id * classes_per_client + k) % len(classes)])
            if client_id not in owners[c]:
                owners[c].append(client_id)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for c, client_ids in owners.items():
        if not client_ids:  # class not owned by anyone (more classes than slots)
            continue
        idx_c = rng.permutation(np.where(labels == c)[0])
        for client_id, chunk in zip(client_ids, np.array_split(idx_c, len(client_ids))):
            client_indices[client_id].extend(chunk.tolist())

    return [np.sort(np.array(ci, dtype=int)) for ci in client_indices]


# ---------------------------------------------------------------------------
# 4. Quantity skew – same label distribution, very different dataset sizes
# ---------------------------------------------------------------------------
def quantity_split(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    seed: int = 42,
) -> list[np.ndarray]:
    """Client sizes follow Dir(alpha) proportions (small alpha = a few "big
    hospitals" and many "small clinics"); samples are drawn at random, so the
    label distribution stays roughly the same on every client."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n < num_clients * min_samples:
        raise ValueError("Not enough samples for the requested min_samples per client")

    proportions = rng.dirichlet(np.full(num_clients, alpha))
    # Guarantee min_samples to everyone, distribute the rest by proportions
    sizes = min_samples + np.floor(proportions * (n - num_clients * min_samples)).astype(int)
    sizes[-1] = n - sizes[:-1].sum()
    cut_points = np.cumsum(sizes)[:-1]
    indices = rng.permutation(n)
    return [np.sort(part) for part in np.split(indices, cut_points)]


# ---------------------------------------------------------------------------
# 5. Natural / group split – use a real-world attribute as the client
# ---------------------------------------------------------------------------
def group_split(groups: np.ndarray, num_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Samples that share a group value (subject, camera, device, hospital, ...)
    always end up on the same client. This is the most realistic setting:
    in practice each client *is* a person / device / institution.

    If there are more groups than clients, groups are packed greedily (largest
    first) onto the currently smallest client, which keeps sizes balanced."""
    rng = np.random.default_rng(seed)
    unique_groups, counts = np.unique(groups, return_counts=True)
    if len(unique_groups) < num_clients:
        raise ValueError(
            f"Only {len(unique_groups)} distinct groups ({list(unique_groups)}) for "
            f"{num_clients} clients. Use --num-clients {len(unique_groups)} or less."
        )

    # Shuffle first so that ties in group size are broken randomly
    order = rng.permutation(len(unique_groups))
    unique_groups, counts = unique_groups[order], counts[order]
    order = np.argsort(-counts, kind="stable")

    client_groups: list[list] = [[] for _ in range(num_clients)]
    client_sizes = np.zeros(num_clients, dtype=int)
    for g_idx in order:
        target = int(np.argmin(client_sizes))
        client_groups[target].append(unique_groups[g_idx])
        client_sizes[target] += counts[g_idx]

    return [np.sort(np.where(np.isin(groups, cg))[0]) for cg in client_groups]


# ---------------------------------------------------------------------------
# Extra "real world" corruptions applied after partitioning
# ---------------------------------------------------------------------------
def apply_feature_skew(
    images: np.ndarray, client_id: int, num_clients: int, seed: int = 42
) -> np.ndarray:
    """Simulate a different acquisition device per client (e.g. a different
    scanner / camera): client 0 gets darker, low-noise images and the last
    client brighter, noisier images. Works on uint8 images of any shape.
    For numeric CSV data, see apply_sensor_skew."""
    rng = np.random.default_rng(seed + client_id)
    t = client_id / max(num_clients - 1, 1)  # 0 ... 1
    brightness = 0.7 + 0.6 * t  # 0.7x ... 1.3x
    noise_std = 20.0 * t  # 0 ... 20 grey levels
    out = images.astype(np.float32) * brightness
    out += rng.normal(0.0, noise_std, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_sensor_skew(
    values: np.ndarray, client_id: int, num_clients: int, seed: int = 42
) -> np.ndarray:
    """Feature skew for numeric (sensor / skeleton) data: every client has a
    differently calibrated device. Each column gets a client-specific gain
    (0.9x ... 1.1x) and offset, plus measurement noise that grows from client 0
    (clean) to the last client. `values` has shape (rows, columns)."""
    rng = np.random.default_rng(seed + client_id)
    t = client_id / max(num_clients - 1, 1)  # 0 ... 1
    scale = values.std(axis=0, keepdims=True) + 1e-8
    gain = 1.0 + rng.uniform(-0.1, 0.1, size=(1, values.shape[1])) * t
    offset = rng.normal(0.0, 0.2, size=(1, values.shape[1])) * scale * t
    noise = rng.normal(0.0, 0.1 * t, size=values.shape) * scale
    return (values * gain + offset + noise).astype(values.dtype)


def apply_label_noise(
    labels: np.ndarray, noise_rate: float, num_classes: int, seed: int = 42
) -> np.ndarray:
    """Flip a fraction `noise_rate` of labels to a different, random class –
    simulates a client with an unreliable annotator."""
    if noise_rate <= 0:
        return labels
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    flip = rng.random(len(labels)) < noise_rate
    offsets = rng.integers(1, num_classes, size=flip.sum())
    noisy[flip] = (labels[flip] + offsets) % num_classes
    return noisy
