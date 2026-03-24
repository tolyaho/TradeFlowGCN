"""Download the CEPII Gravity Database.

The CEPII Gravity dataset provides bilateral trade flows at the
country-pair-year level together with standard gravity covariates
(distance, contiguity, common language, colonial ties, etc.).

Reference: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Default URL for the CEPII Gravity CSV (V202211 release)
DEFAULT_URL = (
    "http://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"
)


def download_gravity_data(
    url: str = DEFAULT_URL,
    raw_dir: str | Path = "data/raw",
    *,
    force: bool = False,
) -> Path:
    """Download and extract the CEPII Gravity dataset.

    Parameters
    ----------
    url : str
        URL of the zipped CSV file.
    raw_dir : str or Path
        Directory to save the extracted CSV.
    force : bool
        If True, re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the extracted CSV file.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    csv_candidates = list(raw_dir.glob("Gravity_V*.csv"))
    if csv_candidates and not force:
        logger.info("CEPII Gravity data already exists: %s", csv_candidates[0])
        return csv_candidates[0]

    logger.info("Downloading CEPII Gravity data from %s ...", url)
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()

    # Read ZIP into memory and extract
    total_bytes = 0
    content = io.BytesIO()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        content.write(chunk)
        total_bytes += len(chunk)
        logger.info("  Downloaded %.1f MB ...", total_bytes / 1e6)

    content.seek(0)
    with zipfile.ZipFile(content) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV files found in the downloaded archive.")
        for csv_name in csv_names:
            logger.info("Extracting %s ...", csv_name)
            zf.extract(csv_name, raw_dir)

    extracted = raw_dir / csv_names[0]
    logger.info("CEPII Gravity data saved to %s", extracted)
    return extracted
