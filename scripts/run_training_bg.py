"""Background training runner — redirects stdout/stderr to log files internally."""
import subprocess
import sys
from pathlib import Path

config = sys.argv[1]
log_dir = sys.argv[2]

Path(log_dir).mkdir(parents=True, exist_ok=True)

stdout_log = open(f"{log_dir}/train_stdout.log", "w", buffering=1)
stderr_log = open(f"{log_dir}/train_stderr.log", "w", buffering=1)

result = subprocess.run(
    [sys.executable, "scripts/train.py", "--config", config],
    cwd=r"d:\CAPSTONE SOFTWARE",
    stdout=stdout_log,
    stderr=stderr_log,
)

stdout_log.close()
stderr_log.close()
sys.exit(result.returncode)
