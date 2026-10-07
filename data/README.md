# Data

No image, feature file or model is in this repository. Download the datasets yourself and keep
them in `data/` (git ignores everything in this folder except this file).

| Dataset | Source | Licence / terms | Use in leafdoctor |
|---|---|---|---|
| PlantVillage | Kaggle `emmarex/plantdisease` (15 folders), or the full 38-class set from the PlantVillage project | Check the terms on the page of your copy | The one source for train, val and test |
| New Plant Diseases Dataset (Augmented) | Kaggle `vipoooool/new-plant-diseases-dataset` | Kaggle dataset terms | Optional. Train only, through `split --augmented`. It is an offline-augmented copy of PlantVillage |
| PlantDoc | GitHub `pratikkayal/PlantDoc-Dataset` | CC BY 4.0, as the repository states | External field test set (`train --field`) |

## Expected layout

```
data/
  PlantVillage/<class folder>/<image>.jpg
  New Plant Diseases Dataset(Augmented)/train/<class folder>/<image>.jpg
  PlantDoc/test/<class folder>/<image>.jpg
```

Each class folder name must map to one of the 38 canonical classes in `src/leafdoctor/classes.py`.
Folder names that do not map are listed as `unknown_folders` by `leafdoctor manifest`. For PlantDoc,
rename the folders (for example `Tomato leaf late blight` to `Tomato___Late_blight`) or add aliases
to `EXTRA_ALIASES`.

## Manifest columns

| Column | Meaning |
|---|---|
| `path` | Absolute path of the image |
| `label` | Canonical class name |
| `source` | Dataset name that you give to `--source` |
| `group_id` | Leaf id. Images with the same leaf id always go to the same split |
| `phash` | 16 hex digits: the dihedral-invariant difference hash |

## Commands

```bash
leafdoctor manifest data/PlantVillage --source plantvillage --out artifacts/pv.csv
leafdoctor manifest "data/New Plant Diseases Dataset(Augmented)/train" --source npdd --out artifacts/npdd.csv
leafdoctor manifest data/PlantDoc/test --source plantdoc --out artifacts/plantdoc.csv
leafdoctor split artifacts/pv.csv --augmented artifacts/npdd.csv
```

## Synthetic data

`leafdoctor synth` and `leafdoctor demo` draw synthetic leaves (`src/leafdoctor/synthetic.py`).
The tests and the offline demo use only synthetic images. They need no download and no network.
