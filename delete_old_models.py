import os
import shutil
from pathlib import Path

print("Scanning Hugging Face cache to find and delete old Qwen3-VL models...")

# Method 1: Using huggingface_hub cache manager
try:
    from huggingface_hub import scan_cache_dir
    cache_info = scan_cache_dir()
    deleted_repos = []
    for repo in cache_info.repos:
        if "Qwen3-VL" in repo.repo_id or "Qwen3" in repo.repo_id:
            print(f"\nFound cached model: {repo.repo_id} ({repo.size_on_disk_str})")
            print(f"Path: {repo.repo_path}")
            try:
                # Delete the repository using its path
                shutil.rmtree(repo.repo_path)
                print(f"✓ Successfully deleted {repo.repo_id} from Hugging Face cache.")
                deleted_repos.append(repo.repo_id)
            except Exception as delete_err:
                print(f"✗ Failed to delete {repo.repo_id}: {delete_err}")
    if not deleted_repos:
        print("No Qwen3-VL models found in Hugging Face cache using cache manager.")
except Exception as e:
    print(f"Hugging Face cache manager check skipped: {e}")

# Method 2: Manual scan fallback for common Windows directories
user_profile = os.getenv("USERPROFILE")
default_cache_paths = []
if user_profile:
    default_cache_paths.append(Path(user_profile) / ".cache" / "huggingface" / "hub")
default_cache_paths.append(Path("C:/Users/mohit/.cache/huggingface/hub"))

print("\nRunning filesystem check for Qwen3 model directories...")
for cache_path in default_cache_paths:
    if cache_path.exists():
        print(f"Scanning directory: {cache_path}")
        # Search for folders matching Qwen3-VL patterns
        for folder in cache_path.iterdir():
            folder_name = folder.name.lower()
            if folder.is_dir() and ("qwen3" in folder_name or "qwen3-vl" in folder_name):
                print(f"Found Qwen3 folder on disk: {folder}")
                try:
                    shutil.rmtree(folder)
                    print(f"✓ Deleted folder: {folder}")
                except Exception as ex:
                    print(f"✗ Failed to delete {folder}: {ex}")
    else:
        print(f"Directory not found: {cache_path}")

print("\nDone! Old Qwen3-VL models have been cleared.")
