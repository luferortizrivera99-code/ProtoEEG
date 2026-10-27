# Etapa 1: datos Sleep-EDF

El cargador original de entrenamiento está en
`protopnet/eeg_utilities/custom_dataset.py`: `EEG_DataSet` lee un diccionario
`.pth` y etiquetas `.npy`; `EEG_ConcatDataset` añade transformaciones.
`protopnet/spikenet_helpers.py` contiene otra versión utilizada por las
utilidades de evaluación. El entrenador consume `img`, `target` y `sample_id`.

El nuevo `SleepEDFDataset` es independiente: lee `.npz` del nivel superior de
la carpeta indicada y conserva las señales preprocesadas. No utiliza las
transformaciones originales. `X` debe ser `(N, 3, 3000)`, `y` debe ser `(N,)`
con enteros de 1 a 6 y `subj` un entero/cadena escalar, un array de un elemento
o un vector `(N,)`. Guarda los sujetos como enteros o cadenas Unicode de NumPy,
no como arrays de objetos que requieran pickle.

Cada elemento contiene:

| Clave | Contenido |
| --- | --- |
| `img` | Tensor float32 `(3, 3000)`, canales Fpz-Cz, Pz-Oz, EOG horizontal |
| `target` | Tensor int64 escalar: W=0, N1=1, N2=2, N3=3, REM=4, Movement=5 |
| `sample_id` | Nombre del archivo y posición de la época, únicos dentro de la carpeta |
| `subject_id` | Identificador original convertido a cadena |

El orden de canales se conserva; el archivo no contiene información con la
que comprobarlo. La duración y frecuencia de muestreo proceden del
preprocesamiento: 3000 muestras a 100 Hz corresponden a 30 segundos.

## Uso en Colab

Con Drive montado y el directorio actual situado en la raíz del repositorio:

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
from torch.utils.data import DataLoader
from protopnet.eeg_utilities.sleep_edf_dataset import SleepEDFDataset, split_by_subject

dataset = SleepEDFDataset('/content/drive/MyDrive/SleepCassette_procesado')
print('Sujetos disponibles:', sorted(set(dataset.subject_ids)))

# Comprobación inicial de un lote; este loader no define una partición.
batch = next(iter(DataLoader(dataset, batch_size=16, shuffle=False)))
assert batch['img'].shape[1:] == (3, 3000)
assert batch['img'].dtype == torch.float32
assert batch['target'].dtype == torch.int64
assert torch.all((batch['target'] >= 0) & (batch['target'] <= 5))
print(batch['img'].shape, batch['target'].tolist(), batch['subject_id'])

# Completar con los identificadores reales impresos arriba.
train_subjects = ['sujeto_train']
val_subjects = ['sujeto_val']
test_subjects = ['sujeto_test']
train_ds, val_ds, test_ds = split_by_subject(
    dataset, train_subjects, val_subjects, test_subjects
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
```

Las listas deben cubrir todos los sujetos sin solapamiento; si no lo hacen,
la función falla con un mensaje explicativo. Dos noches del mismo sujeto
deben tener el mismo `subj`: el cargador no puede inferir que dos IDs distintos
corresponden a una misma persona. No dividir épocas al azar.

Esta versión carga todo en RAM: las señales float32 requieren aproximadamente
36 000 bytes por época (3.6 GB por 100 000 épocas), más metadatos y memoria
temporal durante la concatenación. `num_workers=0` evita procesos adicionales.
Si no cabe en Colab, habrá que abordar la carga por bloques en otra etapa.

Se probó con datos sintéticos, no con archivos privados. Queda comprobar el
formato real de `subj`, el orden de los canales, la memoria disponible y la
distribución de clases por partición. No es necesario que cada lote contenga
las seis clases.

No se conecta todavía al entrenamiento: compartir las claves del diccionario
no hace que el backbone original acepte `(B, 3, 3000)` ni que su clasificador
binario resuelva seis clases. Esos cambios quedan para etapas posteriores.
