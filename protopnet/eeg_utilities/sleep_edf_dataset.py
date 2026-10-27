"""Carga épocas Sleep-EDF ya preprocesadas, sin aplicar transformaciones."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class SleepEDFDataset(Dataset):
    """Lee los .npz de una carpeta en RAM (solo archivos del nivel indicado).

    Cada archivo contiene X: (N, 3, 3000), y: (N,) y subj: un identificador
    escalar o un vector (N,) con el sujeto de cada época. El mismo sujeto
    debe tener el mismo identificador en todas sus noches/archivos.
    """

    class_names = ("W", "N1", "N2", "N3", "REM", "Movement")

    def __init__(self, root):
        files = sorted(Path(root).glob("*.npz"))
        if not files:
            raise ValueError(f"No se encontraron archivos .npz en {root}")

        signals, labels = [], []
        self.subject_ids = []
        self.sample_ids = []
        for path in files:
            # No se necesita pickle para matrices numéricas o cadenas Unicode.
            with np.load(path, allow_pickle=False) as data:
                missing = {"X", "y", "subj"} - set(data.files)
                if missing:
                    raise ValueError(f"{path.name}: faltan claves {sorted(missing)}")
                x, y, subjects = data["X"], data["y"], data["subj"]

            if x.ndim != 3 or x.shape[1:] != (3, 3000) or len(x) == 0:
                raise ValueError(f"{path.name}: X debe tener forma (N, 3, 3000), N > 0")
            if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
                raise ValueError(f"{path.name}: X debe contener señales numéricas reales")
            if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
                raise ValueError(f"{path.name}: y debe ser un vector entero de longitud N")
            if not np.isin(y, np.arange(1, 7)).all():
                raise ValueError(f"{path.name}: las etiquetas deben estar entre 1 y 6")

            # Acepta un sujeto por archivo o un sujeto por época.
            if subjects.size == 1:
                subjects = np.repeat(subjects.reshape(-1), len(x))
            elif subjects.shape != (len(x),):
                raise ValueError(f"{path.name}: subj debe ser escalar o tener forma (N,)")
            if subjects.dtype.kind not in "iuUS":
                raise ValueError(f"{path.name}: subj debe contener identificadores enteros o cadenas")
            subject_ids = subjects.astype(str).tolist()
            if any(not subject.strip() for subject in subject_ids):
                raise ValueError(f"{path.name}: hay identificadores de sujeto vacíos")

            # Solo cambia el tipo a float32; no normaliza, recorta ni remuestrea.
            x = x.astype(np.float32, copy=False)
            if not np.isfinite(x).all():
                raise ValueError(f"{path.name}: X contiene NaN o infinitos")
            signals.append(x)
            labels.append(y.astype(np.int64) - 1)
            self.subject_ids.extend(subject_ids)
            self.sample_ids.extend(f"{path.name}:{i}" for i in range(len(x)))

        self.x = torch.from_numpy(np.concatenate(signals))
        self.y = torch.from_numpy(np.concatenate(labels))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "img": self.x[index],
            "target": self.y[index],
            "sample_id": self.sample_ids[index],
            "subject_id": self.subject_ids[index],
        }


def split_by_subject(dataset, train_subjects, val_subjects, test_subjects):
    """Devuelve tres Subset; exige sujetos disjuntos y cobertura completa.

    Las listas explícitas permiten conservar la misma partición en Colab.
    Los identificadores se comparan como cadenas, igual que en el dataset.
    """
    groups = [set(map(str, ids)) for ids in (train_subjects, val_subjects, test_subjects)]
    if any(not group for group in groups):
        raise ValueError("Cada partición debe contener al menos un sujeto")
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Un sujeto no puede aparecer en más de una partición")
    available = set(dataset.subject_ids)
    requested = set.union(*groups)
    if requested != available:
        raise ValueError(
            f"Sujetos sin asignar: {sorted(available - requested)}; "
            f"sujetos desconocidos: {sorted(requested - available)}"
        )
    return tuple(
        Subset(dataset, [i for i, subject in enumerate(dataset.subject_ids) if subject in group])
        for group in groups
    )
