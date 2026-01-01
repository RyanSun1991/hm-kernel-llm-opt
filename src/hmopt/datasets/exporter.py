"""Export training dataset entries from stored runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

from hmopt.datasets.formatter import format_example
from hmopt.storage.db import models


def export_dataset(session: Session, run_ids: List[str], output_path: Path) -> Path:
    examples = []
    for rid in run_ids:
        examples.append(format_example(session, rid))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return output_path
