import kagglehub
import shutil
from pathlib import Path

MODEL_HANDLE = "google/mobilenet-v1/tensorflow2/100-224-classification"

# Download to KaggleHub cache directory.
cache_path = Path(kagglehub.model_download(MODEL_HANDLE)).resolve()

# Local target directory.
target_dir = Path("./model").resolve()
target_dir.mkdir(parents=True, exist_ok=True)

# Copy model files to the target directory.
# dirs_exist_ok=True allows copying into an existing directory.
shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)

print("KaggleHub cache path:", cache_path)
print("Model copied to:", target_dir)
