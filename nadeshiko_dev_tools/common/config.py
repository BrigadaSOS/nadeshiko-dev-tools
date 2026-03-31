from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingConfig:
    input_folder: Path
    dryrun: bool = False
    parallel: bool = False
    pool_size: int = 6
