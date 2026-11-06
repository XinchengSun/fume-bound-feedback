"""Read original INRIA MAT or canonical NPZ without destroying multi-hot labels.

No relabeling by argmax, test-as-validation, feature normalization, resplitting,
confidence filtering, or train/validation data merging occurs here.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import TensorDataset

SPLITS = ('train', 'val', 'test')
MAT_KEYS = {
    'train': ('tr_img', 'tr_txt', 'tr_img_lab'),
    'val': ('val_img', 'val_txt', 'val_img_lab'),
    'test': ('te_img', 'te_txt', 'te_img_lab'),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _read(path):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as source:
            return {s: tuple(source[f'{s}_{name}'].copy() for name in ('image', 'text', 'labels'))
                    for s in SPLITS}
    if path.suffix.lower() != '.mat':
        raise ValueError('Use original INRIA .mat or canonical .npz; see README')
    from scipy.io import loadmat
    try:
        source = loadmat(path)
        return {s: tuple(np.asarray(source[k]) for k in MAT_KEYS[s]) for s in SPLITS}
    except NotImplementedError:
        import h5py
        with h5py.File(path, 'r') as source:
            out = {}
            for split in SPLITS:
                rows = []
                for key in MAT_KEYS[split]:
                    ds = source[key]
                    if 'MATLAB_class' not in ds.attrs:
                        raise ValueError('Ambiguous HDF5 orientation: export canonical NPZ')
                    rows.append(np.asarray(ds).T)  # MATLAB v7.3 stores reversed axes.
                out[split] = tuple(rows)
            return out


def load_data(path, classes=None, label_base=None):
    raw = _read(path)
    for split, (image, text, y) in raw.items():
        if image.ndim != 2 or text.ndim != 2 or len(image) != len(text) or len(image) < 1:
            raise ValueError(f'{split}: nonempty aligned [N,D] features required')
        if not np.isfinite(image).all() or not np.isfinite(text).all() or not np.isfinite(y).all():
            raise ValueError(f'{split}: nonfinite features/labels')
        if len(y) != len(image):
            raise ValueError(f'{split}: label row count mismatch; no implicit transposition')
    yt = np.asarray(raw['train'][2])
    matrix = yt.ndim == 2 and yt.shape[1] > 1
    if matrix:
        k = yt.shape[1]
        if classes is not None and classes != k:
            raise ValueError('classes does not match the label matrix width')
    else:
        if label_base not in (0, 1) or classes is None or classes < 2:
            raise ValueError('Index labels require explicit --label-base 0/1 and --classes K')
        k = classes
    datasets, receipt = {}, {}
    dims = (raw['train'][0].shape[1], raw['train'][1].shape[1])
    for split, (image, text, y) in raw.items():
        if (image.shape[1], text.shape[1]) != dims:
            raise ValueError('Feature dimensions differ between splits')
        if matrix:
            if y.ndim != 2 or y.shape[1] != k or not np.isin(y, [0, 1]).all():
                raise ValueError('All splits must contain unnormalized hard multi-hot matrices')
            labels = y.astype(np.float32, copy=True)
        else:
            if not (y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)):
                raise ValueError('Mixed label encodings across splits')
            if not np.equal(y, np.round(y)).all():
                raise ValueError('Class indices must be integers')
            ids = y.reshape(-1).astype(np.int64)-label_base
            if (ids < 0).any() or (ids >= k).any():
                raise ValueError('Class index outside the declared range')
            labels = np.eye(k, dtype=np.float32)[ids]
        if (labels.sum(1) == 0).any():
            raise ValueError(f'{split}: unlabeled row; no sample is silently dropped')
        image = np.ascontiguousarray(image, dtype=np.float32)
        text = np.ascontiguousarray(text, dtype=np.float32)
        if not np.isfinite(image).all() or not np.isfinite(text).all():
            raise ValueError('Features overflowed float32')
        datasets[split] = TensorDataset(torch.from_numpy(image), torch.from_numpy(text),
                                       torch.from_numpy(labels))
        receipt[split] = dict(rows=len(image), image_sha256=array_hash(image),
                              text_sha256=array_hash(text), labels_sha256=array_hash(labels),
                              min_labels=int(labels.sum(1).min()), max_labels=int(labels.sum(1).max()))
    # Use ONLY training data. RMS changes noise scale, not input features.
    scales = tuple(float(t.double().square().mean().sqrt())
                   for t in datasets['train'].tensors[:2])
    receipt.update(path=str(Path(path).resolve()), file_sha256=sha256_file(path),
                   image_dim=dims[0], text_dim=dims[1], classes=k,
                   label_encoding='multihot_matrix' if matrix else f'index_base_{label_base}',
                   train_noise_rms=list(scales), resplit=False, feature_normalization=False)
    return datasets, receipt
