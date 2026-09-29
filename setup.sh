#!/bin/bash
# Install psiz itself, editable and without dependencies, into the entry's interpreter. psiz reads
# its own version through importlib.metadata (PsychologicalEmbedding.get_config, which Keras calls
# while fitting), so importing it from src/ is not enough: without installed metadata that raises
# PackageNotFoundError. Editable keeps the checkout as the code that runs, so switching branches
# switches the program. --no-deps leaves the pinned TensorFlow alone (setup.cfg asks for < 2.9), and
# --no-build-isolation builds with the setuptools_scm pinned in requirements.txt.
set -e
cd "$(dirname "$0")"
"${PYTHON:-python3}" -m pip install --no-deps --no-build-isolation -e .
