import os

import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from leafdoctor.manifest import scan_folder  # noqa: E402
from leafdoctor.split import group_split, merge_groups  # noqa: E402
from leafdoctor.synthetic import generate_dataset  # noqa: E402

CLASSES = ("Tomato___healthy", "Tomato___Early_blight", "Corn_(maize)___Common_rust_", "Potato___Late_blight")


@pytest.fixture(scope="session")
def lab_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab")
    generate_dataset(root, leaves_per_class=10, views_per_leaf=3, size=48, seed=1, classes=CLASSES)
    return root


@pytest.fixture(scope="session")
def lab_rows(lab_dir):
    return merge_groups(scan_folder(lab_dir, "lab").rows, 3)


@pytest.fixture(scope="session")
def field_rows(tmp_path_factory):
    root = tmp_path_factory.mktemp("field")
    generate_dataset(root, leaves_per_class=3, views_per_leaf=1, size=48, seed=2, field=True, classes=CLASSES)
    return scan_folder(root, "field").rows


@pytest.fixture(scope="session")
def splits(lab_rows):
    return group_split(lab_rows, seed=7).splits
