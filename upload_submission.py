#!/usr/bin/env python3
"""Package a LayeredDepth prediction directory and upload it resumably.

This keeps the public repository's directory-oriented command line while using
the website's current authenticated, parallel, ranged-upload protocol.

Example:
    python3 upload_submission.py \
        --email user@example.com \
        --path /path/to/predictions \
        --method_name "My method" \
        --benchmark multi_layer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import time
import zipfile
from pathlib import Path
from urllib import error as urllib_error

import layereddepth_upload as uploader


def safe_name(value: str) -> str:
    """Return a short filesystem-safe name for generated state files."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return cleaned[:80] or "submission"


def default_archive_path(source_directory: Path) -> Path:
    return source_directory.with_name(f"{source_directory.name}.layereddepth.zip")


def default_state_path(
    archive_path: Path,
    method_name: str,
    benchmark_name: str,
) -> Path:
    stem = safe_name(archive_path.stem)
    method = safe_name(method_name)
    benchmark = safe_name(benchmark_name)
    return archive_path.parent / f"layereddepth-upload-{stem}-{method}-{benchmark}.json"


def collect_archive_files(
    source_directory: Path,
    excluded_paths: set[Path],
) -> tuple[list[tuple[Path, str]], int]:
    """List regular files in stable archive order, rooted at the directory."""
    normalized_exclusions = {
        path.resolve(strict=False) for path in excluded_paths
    }
    members: list[tuple[Path, str]] = []
    total_size = 0

    for root, directory_names, file_names in os.walk(
        source_directory,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        root_path = Path(root)
        for file_name in file_names:
            path = root_path / file_name
            if path.resolve(strict=False) in normalized_exclusions:
                continue
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise uploader.UploadError(f"Could not stat {path}: {exc}") from exc
            arcname = path.relative_to(source_directory).as_posix()
            members.append((path, arcname))
            total_size += size

    if not members:
        raise uploader.UploadError(
            f"Submission directory contains no regular files: {source_directory}"
        )
    return members, total_size


def print_archive_progress(
    completed: int,
    total: int,
    member_number: int,
    member_count: int,
) -> None:
    percent = 100.0 * completed / total if total else 100.0
    print(
        f"\rPackaging: {percent:6.2f}%  "
        f"{uploader.format_bytes(completed)} / {uploader.format_bytes(total)}  "
        f"({member_number}/{member_count} files)",
        end="",
        flush=True,
    )


def create_or_reuse_archive(
    source_directory: Path,
    archive_path: Path,
    compression_name: str,
    rebuild: bool,
) -> Path:
    """Atomically create a Zip64 archive, or reuse the stable existing one."""
    if archive_path.exists() and not rebuild:
        uploader.inspect_zip(archive_path, verify_contents=False)
        print(
            f"Reusing existing archive: {archive_path}\n"
            "Use --rebuild-archive with a new state file if the directory changed.",
            flush=True,
        )
        return archive_path

    if archive_path.exists() and archive_path.is_dir():
        raise uploader.UploadError(f"Archive path is a directory: {archive_path}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = archive_path.with_name(
        f".{archive_path.name}.tmp-{os.getpid()}"
    )
    members, source_size = collect_archive_files(
        source_directory,
        {archive_path, temporary_path},
    )

    # ZIP_STORED has almost no CPU cost and gives the parallel uploader random
    # access immediately after packaging. Deflate level 1 is available when
    # local disk or transfer size matters more than packaging time.
    compression = (
        zipfile.ZIP_STORED
        if compression_name == "stored"
        else zipfile.ZIP_DEFLATED
    )
    compression_level = None if compression == zipfile.ZIP_STORED else 1

    estimated_overhead = max(64 * 1024 * 1024, len(members) * 1024)
    free_bytes = shutil.disk_usage(archive_path.parent).free
    required_bytes = source_size + estimated_overhead
    if free_bytes < required_bytes:
        raise uploader.UploadError(
            f"Not enough free space to build the archive atomically. "
            f"Need approximately {uploader.format_bytes(required_bytes)}, "
            f"but {uploader.format_bytes(free_bytes)} is available in "
            f"{archive_path.parent}. Use --archive-path on a larger volume."
        )

    print(
        f"Packaging {len(members)} files from {source_directory}\n"
        f"Input size: {uploader.format_bytes(source_size)}\n"
        f"Archive: {archive_path} ({compression_name})",
        flush=True,
    )

    completed = 0
    last_update = 0.0
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=compression,
            compresslevel=compression_level,
            allowZip64=True,
        ) as archive:
            for index, (path, arcname) in enumerate(members, start=1):
                archive.write(path, arcname)
                completed += path.stat().st_size
                now = time.monotonic()
                if now - last_update >= 0.25 or index == len(members):
                    print_archive_progress(
                        completed,
                        source_size,
                        index,
                        len(members),
                    )
                    last_update = now
        print(flush=True)
        os.replace(temporary_path, archive_path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    uploader.inspect_zip(archive_path, verify_contents=False)
    print(
        f"Archive complete: {uploader.format_bytes(archive_path.stat().st_size)}",
        flush=True,
    )
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="LayeredDepth account email")
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Directory containing the prediction files",
    )
    parser.add_argument(
        "--method_name",
        "--method-name",
        dest="method_name",
        default="dummy",
        help="Method name shown on the LayeredDepth submission",
    )
    parser.add_argument(
        "--benchmark",
        default="first_layer",
        choices=["first_layer", "multi_layer"],
    )
    parser.add_argument(
        "--submission-id",
        type=int,
        help="Upload to an existing owned submission instead of creating one",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        help="Staged ZIP path (default: beside the input directory)",
    )
    parser.add_argument(
        "--compression",
        choices=["stored", "deflate"],
        default="stored",
        help="ZIP compression (stored is fastest; deflate uses less space)",
    )
    parser.add_argument(
        "--rebuild-archive",
        action="store_true",
        help="Rebuild the staged ZIP; requires a new/nonexistent state file",
    )
    parser.add_argument("--base-url", default=uploader.DEFAULT_BASE_URL)
    parser.add_argument("--code", help="Verification code (prompted securely if omitted)")
    parser.add_argument(
        "--reuse-code",
        action="store_true",
        help="Use a still-valid code without requesting another email",
    )
    parser.add_argument(
        "--cookie-file",
        type=Path,
        help="Persist the authenticated session for subsequent resumes",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Persistent upload state; reuse the same file to resume",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=float,
        help="Chunk payload size (default: server recommendation, currently 12 MiB)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Parallel requests (default: server recommendation, currently 4)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument(
        "--verify-zip",
        action="store_true",
        help="CRC-test every staged ZIP member before uploading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_directory = args.path.expanduser().resolve()
    if not source_directory.is_dir():
        raise uploader.UploadError(
            f"Submission path must be a directory: {source_directory}"
        )

    archive_path = (
        args.archive_path.expanduser().resolve()
        if args.archive_path
        else default_archive_path(source_directory)
    )
    state_path = (
        args.state_file.expanduser().resolve()
        if args.state_file
        else default_state_path(archive_path, args.method_name, args.benchmark)
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    existing_state: dict[str, object] = {}
    if state_path.exists():
        try:
            existing_state = json.loads(state_path.read_text())
        except (OSError, ValueError) as exc:
            raise uploader.UploadError(
                f"Could not read state file {state_path}: {exc}"
            ) from exc

    if args.rebuild_archive and state_path.exists():
        raise uploader.UploadError(
            "Refusing to rebuild an archive associated with existing upload state. "
            "Choose a new --state-file (and normally a new --archive-path) so "
            "server receipts cannot be mixed with different bytes."
        )
    if state_path.exists() and not archive_path.exists():
        raise uploader.UploadError(
            f"The staged archive recorded by {state_path} is missing: "
            f"{archive_path}. Do not regenerate bytes for an in-progress upload; "
            "restore that archive or start with a new state file."
        )

    archive_path = create_or_reuse_archive(
        source_directory=source_directory,
        archive_path=archive_path,
        compression_name=args.compression,
        rebuild=args.rebuild_archive,
    )
    member_count, uncompressed_size = uploader.inspect_zip(
        archive_path,
        args.verify_zip,
    )
    total_size = archive_path.stat().st_size
    archive_mtime_ns = archive_path.stat().st_mtime_ns
    recorded_mtime_ns = int(existing_state.get("archive_mtime_ns", 0) or 0)
    if recorded_mtime_ns and recorded_mtime_ns != archive_mtime_ns:
        raise uploader.UploadError(
            "The staged ZIP changed since this upload state was created. "
            "Refusing to mix existing server receipts with different local bytes. "
            "Restore the original ZIP or use a new --state-file."
        )

    cookie_path = (
        args.cookie_file.expanduser().resolve() if args.cookie_file else None
    )
    if cookie_path is not None:
        cookie_path.parent.mkdir(parents=True, exist_ok=True)

    client = uploader.LayeredDepthClient(
        base_url=args.base_url,
        email=args.email,
        timeout=args.timeout,
        max_retries=args.max_retries,
        cookie_file=cookie_path,
    )
    client.login(args.code, reuse_code=args.reuse_code)

    saved_submission_id = int(existing_state.get("submission_id", 0) or 0)
    if args.submission_id and saved_submission_id:
        if args.submission_id != saved_submission_id:
            raise uploader.UploadError(
                f"State file belongs to submission #{saved_submission_id}, not "
                f"#{args.submission_id}."
            )
    submission_id = int(args.submission_id or saved_submission_id or 0)
    if not submission_id:
        submission_id = client.create_submission(args.method_name, args.benchmark)

    state = uploader.load_or_create_upload_state(
        state_path=state_path,
        file_path=archive_path,
        submission_id=submission_id,
        total_size=total_size,
    )
    upload_id = str(state["upload_id"])

    initial_status = client.upload_status(submission_id, upload_id)
    capabilities = initial_status.get("capabilities") or {}
    recommended_chunk_size = int(
        capabilities.get("recommended_chunk_size", uploader.DEFAULT_CHUNK_SIZE)
    )
    maximum_chunk_size = int(
        capabilities.get("max_chunk_size", uploader.DEFAULT_CHUNK_SIZE)
    )
    chunk_size = int(
        state.get("chunk_size")
        or (
            args.chunk_size_mib * 1024 * 1024
            if args.chunk_size_mib is not None
            else recommended_chunk_size
        )
    )
    workers = int(
        args.workers
        or capabilities.get("recommended_parallel_uploads", uploader.DEFAULT_WORKERS)
    )
    if chunk_size <= 0 or chunk_size > maximum_chunk_size:
        raise uploader.UploadError(
            f"Chunk size must be between 1 byte and "
            f"{uploader.format_bytes(maximum_chunk_size)}."
        )
    if workers < 1 or workers > 6:
        raise uploader.UploadError("--workers must be between 1 and 6.")
    if capabilities.get("binary_multipart_public") is not True:
        raise uploader.UploadError(
            "The server has not enabled the public binary-multipart transport."
        )

    state.update(
        {
            "submission_id": submission_id,
            "source_directory": str(source_directory),
            "archive_path": str(archive_path),
            "archive_mtime_ns": archive_mtime_ns,
            "archive_compression": args.compression,
            "chunk_size": chunk_size,
            "workers": workers,
            "upload_mode": "parallel-direct",
            "transport": "binary-multipart-v1",
        }
    )
    uploader.save_upload_state(state_path, state)

    print(
        f"Directory: {source_directory}\n"
        f"ZIP: {archive_path}\n"
        f"Compressed: {uploader.format_bytes(total_size)}\n"
        f"Uncompressed: {uploader.format_bytes(uncompressed_size)} "
        f"in {member_count} members\n"
        f"Submission: #{submission_id}\n"
        f"State: {state_path}",
        flush=True,
    )

    if args.status_only:
        print(json.dumps(initial_status, indent=2, sort_keys=True))
        return 0

    if args.monitor_only:
        uploader.monitor_submission(
            client,
            submission_id,
            upload_id,
            args.poll_seconds,
        )
        return 0

    payload = uploader.upload_file(
        client=client,
        file_path=archive_path,
        submission_id=submission_id,
        upload_id=upload_id,
        chunk_size=chunk_size,
        workers=workers,
    )
    status = str(payload.get("submission_status", "unknown"))
    print(f"Server status after upload: {status}", flush=True)

    if not args.no_monitor:
        uploader.monitor_submission(
            client,
            submission_id,
            upload_id,
            args.poll_seconds,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Re-run the same command to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except (
        uploader.UploadError,
        urllib_error.URLError,
        TimeoutError,
        socket.timeout,
        ConnectionError,
        OSError,
    ) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
