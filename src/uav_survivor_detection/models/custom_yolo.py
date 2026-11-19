"""Register custom modules so Ultralytics can build our enhanced model YAMLs.

Ultralytics rebuilds the network from its YAML at the start of ``model.train()``,
so modules inserted into an already-built model in Python are silently dropped
(this is why earlier "GAM+CBAM" checkpoints had exactly the baseline parameter
count). Defining the enhancements in the YAML and registering any module the
parser does not know about makes them survive training.
"""

from pathlib import Path

import ultralytics.nn.tasks as _tasks
from ultralytics.nn.modules.conv import CBAM

ROOT = Path(__file__).resolve().parents[3]
ENHANCED_YAML = ROOT / "configs" / "models" / "yolo11n-p2-cbam.yaml"


def register_custom_modules() -> None:
    """Make CBAM resolvable by ultralytics.nn.tasks.parse_model."""
    if not hasattr(_tasks, "CBAM"):
        _tasks.CBAM = CBAM


register_custom_modules()
