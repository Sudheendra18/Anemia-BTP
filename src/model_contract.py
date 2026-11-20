"""Versioned input and dataset identities for training and capture inference."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def input_format(df, image_col):
    """Identify the ROI format of the images actually used for training."""
    if image_col == "image_path":
        return "raw"
    paths = df[image_col].map(lambda value: Path(str(value)))
    masked = paths.map(lambda path: path.name == "roi_masked.png")
    if "roi_masked_path" in df:
        masked_paths = df["roi_masked_path"].map(lambda value: Path(str(value)))
        masked |= paths.eq(masked_paths)
    formats = set(masked.map({True: "masked", False: "rectangular"}))
    if "roi_input_format" in df:
        declared = set(df["roi_input_format"].dropna().astype(str))
        if declared and declared != formats:
            raise ValueError("Declared ROI format does not match image paths")
    if len(formats) != 1:
        raise ValueError("Do not mix masked and rectangular ROI images in one experiment")
    return formats.pop()


def training_identity(config, train_df, val_df, image_col, hashes=None):
    hashes = {} if hashes is None else hashes

    def rows(df):
        result = []
        for _, row in df.iterrows():
            path = str(row[image_col])
            if path not in hashes:
                hashes[path] = file_hash(path)
            result.append([str(row["image_id"]), str(row["patient_id"]), int(row["label"]), path, hashes[path]])
        return result
    payload = dict(version=1, training=config["training"], label_map=config["dataset"]["label_map"],
                   seed=config.get("random_seed", 42), image_col=image_col,
                   train=rows(train_df), validation=rows(val_df),
                   code={name: file_hash(Path(__file__).with_name(name))
                         for name in ("dataset.py", "model.py", "train.py", "model_contract.py")})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def check_run_manifest(folder, expected):
    path = folder / "run_manifest.json"
    artifacts = [folder / name for name in ("best_model.pt", "metrics.json", "resume_state.pt")]
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Training data/settings changed for {folder}. Choose a new output folder or use --fresh.")
    elif any(p.exists() for p in artifacts):
        raise ValueError(f"Cannot verify old training results in {folder}. Choose a new output folder or use --fresh.")
    weights = folder / "best_model.pt"
    if weights.exists():
        contract_path = folder / "checkpoint_contract.json"
        if not contract_path.is_file():
            raise ValueError(f"Missing checkpoint contract in {folder}; use a new output folder or --fresh")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (contract.get("checkpoint_sha256") != file_hash(weights)
                or contract.get("training_fingerprint") != expected["fingerprint"]
                or contract.get("input_format") != expected["input_format"]):
            raise ValueError(f"Saved weights do not match this training run in {folder}; use a new output folder or --fresh")
