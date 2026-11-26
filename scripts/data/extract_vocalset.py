"""Verify and extract VocalSet11.zip.

Source: https://zenodo.org/records/1442513 (DOI 10.5281/zenodo.1442513), CC BY 4.0,
Wilkins, Seetharaman, Wahl, Pardo (2018). Files go to data/vocalset_full/.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/downloads/VocalSet11.zip"
TARGET = ROOT / "data/vocalset_full"
EXPECTED_BYTES = 2077243579


def main():
    if not ARCHIVE.exists():
        print("archive missing:", ARCHIVE)
        return 1
    size = ARCHIVE.stat().st_size
    if size != EXPECTED_BYTES:
        print(f"size mismatch: {size} of {EXPECTED_BYTES} bytes; download is incomplete, run again to resume")
        return 1
    written = skipped = total_bytes = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or ":" in name:
                raise ValueError(f"unsafe path in archive: {name}")
            if parts[0] == "__MACOSX" or parts[-1].startswith("._") or parts[-1] == ".DS_Store":
                continue
            destination = TARGET.joinpath(*parts)
            total_bytes += member.file_size
            if destination.exists() and destination.stat().st_size == member.file_size:
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)  # zipfile checks each member's CRC
            written += 1
    wavs = sum(1 for _ in TARGET.rglob("*.wav"))
    manifest = {"archive": ARCHIVE.name, "archive_bytes": size, "files_written": written,
                "files_already_present": skipped, "wav_files": wavs, "bytes": total_bytes,
                "source": "https://zenodo.org/records/1442513", "license": "CC-BY-4.0"}
    (TARGET / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
