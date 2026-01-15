# This file marks the 'utils' directory as a Python package.
# It can be used to define package-level variables,
# or to import certain functions/classes directly into the package namespace.

# For example, to make `sca_to_csv` directly accessible via `from utils import sca_to_csv`:
from .sca_to_csv import sca_to_csv
from .sca_csv_batch import sca_to_csv as sca_batch_to_csv # Renamed to avoid conflict
from .html_to_csv import html_to_csv # Assuming html_to_csv is a function in html_to_csv.py
