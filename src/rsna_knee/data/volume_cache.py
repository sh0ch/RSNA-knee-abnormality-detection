"""Discover and publish packed uint8 volume caches as a Kaggle Dataset."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rsna_knee.utils.paths import default_volume_cache_dir, is_kaggle_kernel, project_root

DEFAULT_VOLUME_CACHE_SLUG = "simonhochwebde/rsna-knee-volume-cache"
_MOUNT_NAME = "rsna-knee-volume-cache"
VOLUME_CACHE_PARTS = 4


def volume_cache_part_slugs(
    n_parts: int = VOLUME_CACHE_PARTS,
    *,
    slug: str = DEFAULT_VOLUME_CACHE_SLUG,
) -> list[str]:
    """``owner/name-p1`` … ``owner/name-pN`` (four ~3 GiB tars instead of one ~12 GiB)."""
    n_parts = max(1, int(n_parts))
    owner, _, name = slug.partition("/")
    return [f"{owner}/{name}-p{i}" for i in range(1, n_parts + 1)]


def volume_cache_mount_names(n_parts: int = VOLUME_CACHE_PARTS) -> tuple[str, ...]:
    names = [_MOUNT_NAME, *[s.split("/", 1)[-1] for s in volume_cache_part_slugs(n_parts)]]
    return tuple(dict.fromkeys(names))


def kaggle_argv() -> list[str]:
    """Kaggle CLI as ``kaggle ...``, not ``python -m kaggle``.

    Notebooks ship a non-CLI ``kaggle`` package (and ``/kaggle/working/kaggle``
    can shadow the API), so ``python -m kaggle`` raises ``No module named
    kaggle.__main__``.
    """
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    print("kaggle CLI not on PATH — pip install -U kaggle", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "kaggle"],
        check=True,
    )
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    raise RuntimeError(
        "kaggle CLI not found after pip install. "
        "In a cell run: import shutil; print(shutil.which('kaggle'))"
    )


def _run_kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [*kaggle_argv(), *args]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def volume_cache_shape_name(
    volume_shape: tuple[int, int, int],
    max_series: int,
) -> str:
    depth, height, width = volume_shape
    return f"d{depth}_h{height}_w{width}_s{max_series}"


def _kaggle_input_root() -> Path:
    return Path("/kaggle/input")


def discover_volume_cache_roots() -> list[Path]:
    """Attached Dataset mounts (short name and ``datasets/<owner>/<slug>``)."""
    if not is_kaggle_kernel():
        return []
    root = _kaggle_input_root()
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path)
        if key in seen or not path.is_dir():
            return
        seen.add(key)
        dirs.append(path)

    _add(root / _MOUNT_NAME)
    owner, _, _base = DEFAULT_VOLUME_CACHE_SLUG.partition("/")
    for name in volume_cache_mount_names():
        _add(root / name)
        _add(root / "datasets" / owner / name)
    datasets = root / "datasets"
    if datasets.is_dir():
        try:
            owners = list(datasets.iterdir())
        except OSError:
            owners = []
        for owner_dir in owners:
            try:
                children = list(owner_dir.iterdir())
            except OSError:
                children = []
            for child in children:
                if child.name.startswith(_MOUNT_NAME):
                    _add(child)
    return dirs


def discover_volume_cache_read_dirs(shape_name: str) -> list[Path]:
    """Directories that may contain ``{study_uid}.npy`` for this volume shape."""
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in discover_volume_cache_roots():
        for cand in (root / shape_name, root):
            key = str(cand)
            if key in seen or not cand.is_dir():
                continue
            seen.add(key)
            dirs.append(cand)
    return dirs


def cached_npy_stems(dirs: list[Path]) -> set[str]:
    stems: set[str] = set()
    for folder in dirs:
        if not folder.is_dir():
            continue
        try:
            entries = folder.iterdir()
        except OSError:
            continue
        for path in entries:
            if path.suffix == ".npy" and not path.name.startswith("."):
                stems.add(path.stem)
    return stems


def find_local_volume_cache_dir(
    volume_shape: tuple[int, int, int] = (16, 256, 256),
    max_series: int = 3,
) -> Path | None:
    """Writable cache folder that already has ``.npy`` files (for publish)."""
    shape = volume_cache_shape_name(volume_shape, max_series)
    roots: list[Path] = []
    env_root = default_volume_cache_dir()
    if env_root is not None:
        roots.append(env_root)
    if is_kaggle_kernel():
        roots.extend(
            [
                Path("/tmp/rsna_volume_cache"),
                Path("/kaggle/tmp/rsna_volume_cache"),
                Path("/kaggle/working/volume_cache"),
            ]
        )
    else:
        roots.append(project_root() / "volume_cache")
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        for cand in (root / shape, root):
            if cand.is_dir() and cached_npy_stems([cand]):
                return cand
    return None


def push_volume_cache_dataset(
    cache_dir: Path | str,
    *,
    slug: str = DEFAULT_VOLUME_CACHE_SLUG,
) -> Path:
    """
    Upload packed ``.npy`` volumes as a private Kaggle Dataset.

    Does not copy the files (they are ~14 GiB). Writes ``dataset-metadata.json``
    into ``cache_dir`` and points the Kaggle CLI at that folder. Needs internet
    (interactive kernel or local CLI).
    """
    cache_dir = Path(cache_dir)
    if cache_dir.is_dir() and not cached_npy_stems([cache_dir]):
        nested = [
            child
            for child in cache_dir.iterdir()
            if child.is_dir() and cached_npy_stems([child])
        ]
        if len(nested) == 1:
            cache_dir = nested[0]
    n_files = len(cached_npy_stems([cache_dir]))
    if n_files == 0:
        raise FileNotFoundError(f"No .npy volumes in {cache_dir}")

    meta = {
        "id": slug,
        "title": slug.split("/", 1)[-1],
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (cache_dir / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Publishing {n_files} npy files from {cache_dir} → {slug}", flush=True)

    status = _run_kaggle(
        "datasets", "status", slug, check=False, capture=True
    )
    status_text = (status.stdout or "") + (status.stderr or "")
    missing = status.returncode != 0 or any(
        s in status_text.lower()
        for s in ("403", "404", "not found", "forbidden", "does not exist")
    )
    if missing:
        _run_kaggle(
            "datasets", "create", "-p", str(cache_dir), "--dir-mode", "tar"
        )
        return cache_dir

    note = f"uint8 volume cache {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} ({n_files} studies)"
    _run_kaggle(
        "datasets",
        "version",
        "-p",
        str(cache_dir),
        "-m",
        note,
        "--dir-mode",
        "tar",
    )
    return cache_dir


def _npy_files(cache_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in cache_dir.iterdir()
        if path.suffix == ".npy" and not path.name.startswith(".")
    )


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _resolve_npy_dir(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    if cache_dir.is_dir() and not cached_npy_stems([cache_dir]):
        nested = [
            child
            for child in cache_dir.iterdir()
            if child.is_dir() and cached_npy_stems([child])
        ]
        if len(nested) == 1:
            return nested[0]
    return cache_dir


def shard_npy_files(cache_dir: Path | str, n_parts: int = VOLUME_CACHE_PARTS) -> list[list[Path]]:
    """Split ``*.npy`` into ``n_parts`` contiguous groups (sorted by filename)."""
    n_parts = max(1, int(n_parts))
    files = _npy_files(_resolve_npy_dir(Path(cache_dir)))
    if not files:
        raise FileNotFoundError(f"No .npy volumes in {cache_dir}")
    size, extra = divmod(len(files), n_parts)
    shards: list[list[Path]] = []
    idx = 0
    for part in range(n_parts):
        take = size + (1 if part < extra else 0)
        shards.append(files[idx : idx + take])
        idx += take
    return shards


def stage_volume_cache_part(
    cache_dir: Path | str,
    part: int,
    *,
    n_parts: int = VOLUME_CACHE_PARTS,
    stage_root: Path | str | None = None,
) -> Path:
    """Hardlink one shard into a small folder for ``kaggle datasets create``."""
    n_parts = max(1, int(n_parts))
    if part < 1 or part > n_parts:
        raise ValueError(f"part must be 1..{n_parts}, got {part}")
    src_dir = _resolve_npy_dir(Path(cache_dir))
    files = shard_npy_files(src_dir, n_parts)[part - 1]
    if not files:
        raise FileNotFoundError(f"Part {part}/{n_parts} is empty")
    if stage_root is None:
        stage_root = (
            Path("/kaggle/working/_volume_cache_parts")
            if is_kaggle_kernel()
            else project_root() / "_volume_cache_parts"
        )
    dest = Path(stage_root) / f"p{part}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for src in files:
        _link_or_copy(src, dest / src.name)
    print(f"Staged part {part}/{n_parts}: {len(files)} npy → {dest}", flush=True)
    return dest


def push_volume_cache_part(
    cache_dir: Path | str,
    part: int,
    *,
    n_parts: int = VOLUME_CACHE_PARTS,
    slug: str | None = None,
) -> Path:
    """Upload one of ``n_parts`` Datasets (``…-p1`` … ``…-pN``). Re-run after a timeout."""
    slugs = volume_cache_part_slugs(n_parts)
    part_slug = slug or slugs[part - 1]
    staged = stage_volume_cache_part(cache_dir, part, n_parts=n_parts)
    return push_volume_cache_dataset(staged, slug=part_slug)


def push_volume_cache_parts(
    cache_dir: Path | str,
    *,
    n_parts: int = VOLUME_CACHE_PARTS,
    start_part: int = 1,
    only_part: int | None = None,
) -> list[Path]:
    """Upload shards sequentially. Use ``only_part=2`` to resume after part 1."""
    n_parts = max(1, int(n_parts))
    if only_part is not None:
        parts = [int(only_part)]
    else:
        parts = list(range(int(start_part), n_parts + 1))
    uploaded: list[Path] = []
    for part in parts:
        print(f"=== Volume cache upload {part}/{n_parts} ===", flush=True)
        uploaded.append(push_volume_cache_part(cache_dir, part, n_parts=n_parts))
    print(
        "Attach Datasets "
        + ", ".join(volume_cache_part_slugs(n_parts))
        + " then restart Jupyter.",
        flush=True,
    )
    return uploaded
