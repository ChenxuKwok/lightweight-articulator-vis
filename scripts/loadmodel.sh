# pip install huggingface_hub  # if you haven't
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="cheoljun95/Speech-Articulatory-Coding",
    local_dir="./sparc_models",           # put them wherever you like
    local_dir_use_symlinks=False          # copy, don’t symlink
)
PY