#!/usr/bin/env python3
"""Core parallel, resumable LayeredDepth ZIP uploader.

The script logs in with LayeredDepth's email-code flow, uploads a ZIP through
the public binary-multipart protocol, and monitors evaluation status. The
directory-oriented ``upload_submission.py`` wrapper packages prediction files
and calls this module; this file can also be run directly for an existing ZIP.

Example:
    python3 upload_layereddepth.py \
        /path/to/submission.zip \
        --email user@example.com \
        --create-submission "My submission" \
        --state-file ./layereddepth-upload.json
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import getpass
import hashlib
import http.cookiejar
import json
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


DEFAULT_BASE_URL = "https://layereddepth.cs.princeton.edu/"
DEFAULT_CHUNK_SIZE = 12 * 1024 * 1024
DEFAULT_WORKERS = 4
BASE64_FALLBACK_CHUNK_SIZE = 6 * 1024 * 1024
PUBLIC_MULTIPART_REQUEST_LIMIT = 13_107_200
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
TERMINAL_STATUSES = {"finished", "failed"}


class UploadError(RuntimeError):
    """A LayeredDepth request or upload failed."""


class BinaryIngressRejected(UploadError):
    """The public Apache/ModSecurity layer rejected binary content."""


@dataclasses.dataclass
class HttpResponse:
    status_code: int
    body: bytes
    headers: Any
    url: str

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{value} B"


def response_payload(response: HttpResponse) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip().replace("\n", " ")
        payload = {"error": text[:500] or f"HTTP {response.status_code}"}
    if not isinstance(payload, dict):
        return {"error": f"Unexpected response: {payload!r}"}
    return payload


def status_has_chunk(payload: dict[str, Any], offset: int, size: int) -> bool:
    for chunk in payload.get("received_chunks") or []:
        if int(chunk.get("offset", -1)) == offset and int(chunk.get("size", -1)) == size:
            return True
    chunk_end = offset + size
    for interval in payload.get("received_intervals") or []:
        start = int(interval.get("offset", -1))
        end = start + int(interval.get("size", 0))
        if start <= offset and end >= chunk_end:
            return True
    return False


def received_intervals(
    payload: dict[str, Any], total_size: int
) -> list[tuple[int, int]]:
    """Return normalized, merged byte ranges confirmed by the server."""
    ranges: list[tuple[int, int]] = []
    items = payload.get("received_intervals") or payload.get("received_chunks") or []
    for item in items:
        try:
            start = int(item["offset"])
            size = int(item["size"])
        except (KeyError, TypeError, ValueError):
            continue
        end = min(start + size, total_size)
        start = max(start, 0)
        if start < end:
            ranges.append((start, end))

    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def missing_chunks(
    payload: dict[str, Any], total_size: int, chunk_size: int
) -> list[tuple[int, int]]:
    """Split only unreceived gaps into requests no larger than chunk_size.

    A binary chunk may be stored as two base64 fallback receipts. Resuming on a
    fixed chunk grid can then collide with the first fallback receipt and cause
    HTTP 409. Gap-based scheduling works for arbitrary receipt boundaries.
    """
    chunks: list[tuple[int, int]] = []
    cursor = 0
    for start, end in received_intervals(payload, total_size):
        while cursor < start:
            size = min(chunk_size, start - cursor)
            chunks.append((cursor, size))
            cursor += size
        cursor = max(cursor, end)
    while cursor < total_size:
        size = min(chunk_size, total_size - cursor)
        chunks.append((cursor, size))
        cursor += size
    return chunks


def multipart_file_body(
    data: bytes,
    *,
    field_name: str,
    filename: str,
) -> tuple[bytes, str]:
    """Encode one binary file field exactly as the browser's FormData does."""
    boundary = "----LayeredDepthUploader" + uuid.uuid4().hex
    safe_filename = Path(filename).name.replace("\\", "_").replace('"', "_")
    safe_filename = safe_filename.replace("\r", "_").replace("\n", "_")
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{safe_filename}"\r\n'
        "Content-Type: application/octet-stream\r\n"
        "\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix + data + suffix, boundary


class LayeredDepthClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        timeout: float,
        max_retries: int,
        cookie_file: Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.email = email.strip().lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.cookie_file = cookie_file
        if cookie_file is not None:
            self.cookie_jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
            if cookie_file.exists():
                try:
                    self.cookie_jar.load(ignore_discard=True, ignore_expires=True)
                except (OSError, http.cookiejar.LoadError):
                    pass
        else:
            self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib_request.build_opener(
            urllib_request.HTTPCookieProcessor(self.cookie_jar)
        )
        self.default_headers = {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0.0.0 Safari/537.36"
            ),
            "X-Requested-With": "XMLHttpRequest",
        }

    def save_cookies(self) -> None:
        if self.cookie_file is not None:
            self.cookie_jar.save(ignore_discard=True, ignore_expires=True)
            os.chmod(self.cookie_file, 0o600)

    def url(self, path: str) -> str:
        return urllib_parse.urljoin(self.base_url, path.lstrip("/"))

    def request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
    ) -> HttpResponse:
        url = self.url(path)
        if query:
            url += ("&" if "?" in url else "?") + urllib_parse.urlencode(query)
        request_headers = dict(self.default_headers)
        if headers:
            request_headers.update(headers)
        request = urllib_request.Request(
            url,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return HttpResponse(
                    status_code=response.status,
                    body=response.read(),
                    headers=response.headers,
                    url=response.geturl(),
                )
        except urllib_error.HTTPError as exc:
            return HttpResponse(
                status_code=exc.code,
                body=exc.read(),
                headers=exc.headers,
                url=exc.geturl(),
            )

    def csrf_token(self) -> str:
        tokens = [
            cookie.value
            for cookie in self.cookie_jar
            if cookie.name == "csrftoken"
        ]
        if not tokens:
            raise UploadError("The server did not set a CSRF cookie.")
        return tokens[-1]

    def post_headers(self, referer_path: str) -> dict[str, str]:
        parsed = urllib_parse.urlparse(self.base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return {
            "X-CSRFToken": self.csrf_token(),
            "Referer": self.url(referer_path),
            "Origin": origin,
        }

    def begin_login(self) -> None:
        response = self.request("GET", "/auth/")
        if response.status_code != 200:
            raise UploadError(
                f"Could not load login page: HTTP {response.status_code}"
            )

        token = self.csrf_token()
        form_data = urllib_parse.urlencode(
            {
                "action": "login",
                "email": self.email,
                "csrfmiddlewaretoken": token,
            }
        ).encode()
        headers = self.post_headers("/auth/")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        response = self.request(
            "POST",
            "/auth/request_code/",
            data=form_data,
            headers=headers,
        )
        payload = response_payload(response)
        if response.status_code != 200 or not payload.get("ok"):
            raise UploadError(
                f"Could not request login code (HTTP {response.status_code}): "
                f"{payload.get('error', payload)}"
            )
        self.save_cookies()
        print(f"Verification code sent to {self.email}.", flush=True)

    def finish_login(self, code: str) -> None:
        token = self.csrf_token()
        form_data = urllib_parse.urlencode(
            {
                "action": "login",
                "email": self.email,
                "code": code.strip(),
                "csrfmiddlewaretoken": token,
            }
        ).encode()
        headers = self.post_headers("/auth/")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        response = self.request(
            "POST",
            "/auth/verify/",
            data=form_data,
            headers=headers,
        )
        payload = response_payload(response)
        if response.status_code != 200 or not payload.get("ok"):
            raise UploadError(
                f"Login verification failed (HTTP {response.status_code}): "
                f"{payload.get('error', payload)}"
            )
        self.save_cookies()
        print("LayeredDepth login succeeded.", flush=True)

    def login(self, code: str | None = None, reuse_code: bool = False) -> None:
        response = self.request("GET", "/submissions/")
        if response.status_code == 200 and "/submissions/" in response.url:
            self.save_cookies()
            print("Restored existing LayeredDepth login.", flush=True)
            return
        if reuse_code:
            response = self.request("GET", "/auth/")
            if response.status_code != 200:
                raise UploadError(
                    f"Could not load login page: HTTP {response.status_code}"
                )
        else:
            self.begin_login()
        if not code:
            code = getpass.getpass("LayeredDepth verification code: ")
        self.finish_login(code)

    def create_submission(self, name: str, benchmark_name: str) -> int:
        response = self.request("GET", "/submissions/new/")
        if response.status_code != 200:
            raise UploadError(
                f"Could not load the new-submission page: HTTP {response.status_code}"
            )
        token = self.csrf_token()
        form_data = urllib_parse.urlencode(
            {
                "name": name,
                "benchmark_name": benchmark_name,
                "csrfmiddlewaretoken": token,
            }
        ).encode()
        headers = self.post_headers("/submissions/new/")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        response = self.request(
            "POST",
            "/submissions/new/",
            data=form_data,
            headers=headers,
        )
        match = re.search(r"/submission/(\d+)/?", response.url)
        if response.status_code != 200 or not match:
            raise UploadError(
                f"Could not create submission (HTTP {response.status_code}, "
                f"final URL {response.url})."
            )
        submission_id = int(match.group(1))
        print(f"Created LayeredDepth submission #{submission_id}.", flush=True)
        return submission_id

    def upload_status(
        self,
        submission_id: int,
        upload_id: str,
    ) -> dict[str, Any]:
        response = self.request(
            "GET",
            f"/submission/{submission_id}/upload/status/",
            query={"upload_id": upload_id},
        )
        payload = response_payload(response)
        if response.status_code != 200 or not payload.get("ok"):
            raise UploadError(
                f"Could not query upload status (HTTP {response.status_code}): "
                f"{payload.get('error', payload)}"
            )
        return payload

    def send_chunk(
        self,
        submission_id: int,
        upload_id: str,
        offset: int,
        total_size: int,
        chunk: bytes,
        filename: str,
    ) -> dict[str, Any]:
        request_body, boundary = multipart_file_body(
            chunk,
            field_name="chunk",
            filename=f"{Path(filename).name}.part",
        )
        if len(request_body) > PUBLIC_MULTIPART_REQUEST_LIMIT:
            raise UploadError(
                f"The binary multipart request is {format_bytes(len(request_body))}, "
                f"above the public proxy limit of "
                f"{format_bytes(PUBLIC_MULTIPART_REQUEST_LIMIT)}. "
                "Use a smaller --chunk-size-mib."
            )
        headers = self.post_headers(f"/submission/{submission_id}/")
        headers.update(
            {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-LayeredDepth-Upload": "chunked",
                "X-Upload-Mode": "parallel-direct",
                "X-Upload-ID": upload_id,
                "X-Upload-Offset": str(offset),
                "X-Upload-Total-Size": str(total_size),
                "X-Upload-Final": "0",
                "X-Upload-Filename": filename,
                "X-Upload-Encoding": "identity",
                "X-Upload-Chunk-SHA256": hashlib.sha256(chunk).hexdigest(),
            }
        )
        try:
            return self._post_chunk_with_retries(
                submission_id=submission_id,
                upload_id=upload_id,
                offset=offset,
                chunk_size=len(chunk),
                data=request_body,
                headers=headers,
                raise_on_html_500=True,
            )
        except BinaryIngressRejected:
            print(
                f"\nBinary chunk at {format_bytes(offset)} was rejected or reset; "
                "using 6 MiB base64 subchunks.",
                flush=True,
            )
            return self._send_base64_fallback(
                submission_id=submission_id,
                upload_id=upload_id,
                offset=offset,
                total_size=total_size,
                chunk=chunk,
                filename=filename,
            )

    def _send_base64_fallback(
        self,
        submission_id: int,
        upload_id: str,
        offset: int,
        total_size: int,
        chunk: bytes,
        filename: str,
    ) -> dict[str, Any]:
        last_payload: dict[str, Any] = {"ok": True}
        for relative_offset in range(0, len(chunk), BASE64_FALLBACK_CHUNK_SIZE):
            raw_part = chunk[
                relative_offset : relative_offset + BASE64_FALLBACK_CHUNK_SIZE
            ]
            encoded_part = base64.b64encode(raw_part)
            request_body, boundary = multipart_file_body(
                encoded_part,
                field_name="chunk",
                filename=f"{Path(filename).name}.b64",
            )
            if len(request_body) > PUBLIC_MULTIPART_REQUEST_LIMIT:
                raise UploadError(
                    f"Fallback request exceeds the public limit: {len(request_body)} bytes."
                )
            part_offset = offset + relative_offset
            headers = self.post_headers(f"/submission/{submission_id}/")
            headers.update(
                {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "X-LayeredDepth-Upload": "chunked",
                    "X-Upload-Mode": "parallel-direct",
                    "X-Upload-ID": upload_id,
                    "X-Upload-Offset": str(part_offset),
                    "X-Upload-Total-Size": str(total_size),
                    "X-Upload-Final": "0",
                    "X-Upload-Filename": filename,
                    "X-Upload-Encoding": "base64",
                    "X-Upload-Chunk-SHA256": hashlib.sha256(raw_part).hexdigest(),
                }
            )
            last_payload = self._post_chunk_with_retries(
                submission_id=submission_id,
                upload_id=upload_id,
                offset=part_offset,
                chunk_size=len(raw_part),
                data=request_body,
                headers=headers,
                raise_on_html_500=False,
            )
        return last_payload

    def _post_chunk_with_retries(
        self,
        submission_id: int,
        upload_id: str,
        offset: int,
        chunk_size: int,
        data: bytes,
        headers: dict[str, str],
        raise_on_html_500: bool,
    ) -> dict[str, Any]:
        last_error = "unknown upload error"

        for attempt in range(self.max_retries + 1):
            html_server_error = False
            transport_reset = False
            try:
                response = self.request(
                    "POST",
                    f"/submission/{submission_id}/upload/",
                    data=data,
                    headers=headers,
                )
                payload = response_payload(response)
                content_type = str(response.headers.get("Content-Type", "")).lower()
                html_server_error = (
                    response.status_code == 500 and "application/json" not in content_type
                )

                if response.status_code == 200 and payload.get("ok"):
                    return payload

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{payload.get('error', payload)}"
                )
                if response.status_code == 413:
                    raise UploadError(
                        f"The public proxy rejected a {format_bytes(len(data))} "
                        "multipart request with HTTP 413. Retry with a smaller "
                        "--chunk-size-mib."
                    )
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise UploadError(last_error)
            except (
                urllib_error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                transport_reset = True

            # The server may have committed the ranged chunk even if the
            # response was lost. Receipts make retries idempotent.
            try:
                status = self.upload_status(submission_id, upload_id)
                if status_has_chunk(status, offset, chunk_size):
                    return {
                        "ok": True,
                        "uploaded_bytes": int(status.get("uploaded_bytes", 0)),
                        "total_size": int(status.get("total_size", 0)),
                        "chunk_offset": offset,
                        "chunk_size": chunk_size,
                        "submission_status": status.get("submission_status"),
                        "error_message": status.get("error_message", ""),
                    }
            except (
                UploadError,
                urllib_error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
            ):
                pass

            if raise_on_html_500 and (html_server_error or transport_reset):
                raise BinaryIngressRejected(last_error)

            if attempt >= self.max_retries:
                break
            delay = min(30.0, 2.0**attempt) + random.random()
            print(
                f"\nChunk at {format_bytes(offset)} failed ({last_error}); "
                f"retrying in {delay:.1f}s...",
                flush=True,
            )
            time.sleep(delay)

        raise UploadError(
            f"Chunk at offset {offset} failed after "
            f"{self.max_retries + 1} attempts: {last_error}"
        )

    def finalize_upload(
        self,
        submission_id: int,
        upload_id: str,
        total_size: int,
        filename: str,
    ) -> dict[str, Any]:
        boundary = "----LayeredDepthFinalize" + uuid.uuid4().hex
        request_body = f"--{boundary}--\r\n".encode("ascii")
        headers = self.post_headers(f"/submission/{submission_id}/")
        headers.update(
            {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-LayeredDepth-Upload": "chunked",
                "X-Upload-Mode": "parallel-direct",
                "X-Upload-ID": upload_id,
                "X-Upload-Offset": "0",
                "X-Upload-Total-Size": str(total_size),
                "X-Upload-Finalize": "1",
                "X-Upload-Filename": filename,
                "X-Upload-Encoding": "identity",
            }
        )
        last_error = "unknown finalization error"
        for attempt in range(self.max_retries + 1):
            response = self.request(
                "POST",
                f"/submission/{submission_id}/upload/",
                data=request_body,
                headers=headers,
            )
            payload = response_payload(response)
            if response.status_code == 200 and payload.get("ok"):
                return payload
            last_error = f"HTTP {response.status_code}: {payload.get('error', payload)}"
            if response.status_code not in RETRYABLE_STATUS_CODES:
                raise UploadError(last_error)
            if attempt < self.max_retries:
                time.sleep(min(30.0, 2.0**attempt) + random.random())
        raise UploadError(f"Upload finalization failed: {last_error}")


def inspect_zip(path: Path, verify_contents: bool) -> tuple[int, int]:
    if path.suffix.lower() != ".zip":
        raise UploadError(f"Submission must be a .zip file: {path}")
    if not path.is_file():
        raise UploadError(f"Submission file does not exist: {path}")

    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            uncompressed_size = sum(member.file_size for member in members)
            if verify_contents:
                print(
                    "Testing every ZIP member. This can take a long time for a large archive.",
                    flush=True,
                )
                bad_member = archive.testzip()
                if bad_member:
                    raise UploadError(f"Corrupt ZIP member: {bad_member}")
    except zipfile.BadZipFile as exc:
        raise UploadError(f"Invalid ZIP archive: {exc}") from exc

    return len(members), uncompressed_size


def load_or_create_upload_state(
    state_path: Path,
    file_path: Path,
    submission_id: int,
    total_size: int,
) -> dict[str, Any]:
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError) as exc:
            raise UploadError(f"Could not read state file {state_path}: {exc}") from exc

        expected = {
            "file_path": str(file_path),
            "file_size": total_size,
            "submission_id": submission_id,
        }
        mismatches = [
            key for key, value in expected.items() if state.get(key) != value
        ]
        if mismatches:
            raise UploadError(
                f"State file {state_path} belongs to a different upload "
                f"(mismatched: {', '.join(mismatches)})."
            )
        upload_id = str(state.get("upload_id", ""))
        if upload_id:
            return state

    upload_id = "upload_" + uuid.uuid4().hex
    state = {
        "upload_id": upload_id,
        "file_path": str(file_path),
        "file_size": total_size,
        "submission_id": submission_id,
        "created_at": int(time.time()),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def save_upload_state(state_path: Path, state: dict[str, Any]) -> None:
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(state_path)


def print_progress(uploaded: int, total: int, started_at: float) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    speed = uploaded / elapsed
    percent = 100.0 * uploaded / total if total else 0.0
    remaining = max(total - uploaded, 0)
    eta = remaining / speed if speed > 0 else 0
    message = (
        f"\r{percent:6.2f}%  {format_bytes(uploaded)} / {format_bytes(total)}  "
        f"{format_bytes(int(speed))}/s  ETA {eta / 3600:.2f}h"
    )
    print(message, end="", flush=True)


def upload_file(
    client: LayeredDepthClient,
    file_path: Path,
    submission_id: int,
    upload_id: str,
    chunk_size: int,
    workers: int,
) -> dict[str, Any]:
    total_size = file_path.stat().st_size
    status = client.upload_status(submission_id, upload_id)
    uploaded_bytes = int(status.get("uploaded_bytes", 0))
    submission_status = str(status.get("submission_status", ""))

    if uploaded_bytes == total_size and submission_status != "not_submitted":
        print(
            f"Server already has the complete file; status is {submission_status}.",
            flush=True,
        )
        return status
    if uploaded_bytes > total_size:
        raise UploadError(
            f"Server reports {uploaded_bytes} uploaded bytes, larger than local file "
            f"size {total_size}. Use a new --state-file."
        )

    chunks = missing_chunks(status, total_size, chunk_size)
    completed_bytes = total_size - sum(size for _, size in chunks)

    print(
        f"Upload ID: {upload_id}\n"
        f"Binary multipart: {format_bytes(chunk_size)} chunks, {workers} workers\n"
        f"Server receipts: {format_bytes(completed_bytes)} of {format_bytes(total_size)}; "
        f"{len(chunks)} chunks remain.",
        flush=True,
    )
    started_at = time.monotonic()
    initial_completed = completed_bytes
    progress_lock = threading.Lock()
    file_descriptor = os.open(file_path, os.O_RDONLY)

    def send_one(offset: int, size: int) -> None:
        nonlocal completed_bytes
        chunk = os.pread(file_descriptor, size, offset)
        if len(chunk) != size:
            raise UploadError(
                f"Unexpected EOF at byte offset {offset}: read {len(chunk)} of {size}."
            )
        client.send_chunk(
            submission_id=submission_id,
            upload_id=upload_id,
            offset=offset,
            total_size=total_size,
            chunk=chunk,
            filename=file_path.name,
        )
        with progress_lock:
            completed_bytes += size
            print_progress(
                completed_bytes - initial_completed,
                total_size - initial_completed,
                started_at,
            )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [executor.submit(send_one, offset, size) for offset, size in chunks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        os.close(file_descriptor)

    print("\nAll chunk receipts are complete. Finalizing...", flush=True)
    client.finalize_upload(
        submission_id=submission_id,
        upload_id=upload_id,
        total_size=total_size,
        filename=file_path.name,
    )
    print("Upload complete.", flush=True)
    return client.upload_status(submission_id, upload_id)


def monitor_submission(
    client: LayeredDepthClient,
    submission_id: int,
    upload_id: str,
    poll_seconds: float,
) -> dict[str, Any]:
    last_status = ""
    while True:
        payload = client.upload_status(submission_id, upload_id)
        status = str(payload.get("submission_status", "unknown"))
        if status != last_status:
            print(f"Submission #{submission_id}: {status}", flush=True)
            last_status = status
        if status in TERMINAL_STATUSES:
            if status == "failed":
                message = payload.get("error_message") or "No server error was provided."
                raise UploadError(f"Evaluation failed: {message}")
            return payload
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Local submission ZIP")
    parser.add_argument("--email", required=True, help="LayeredDepth account email")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--submission-id", type=int)
    target.add_argument(
        "--create-submission",
        metavar="NAME",
        help="Create a fresh private submission through the public website",
    )
    parser.add_argument("--benchmark-name", default="multi_layer")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--code", help="Verification code (prompted securely if omitted)")
    parser.add_argument(
        "--reuse-code",
        action="store_true",
        help="Use the newest unexpired code without requesting another email",
    )
    parser.add_argument(
        "--cookie-file",
        type=Path,
        help="Persist the authenticated browser-style session for safe restarts",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=float,
        help="Binary chunk size in MiB (default: server recommendation, currently 12)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Parallel upload workers (default: server recommendation, currently 4)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Log in and monitor an existing upload/evaluation",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Log in, print one upload/evaluation status response, and exit",
    )
    parser.add_argument(
        "--verify-zip",
        action="store_true",
        help="Read and CRC-test the entire ZIP before uploading",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Upload-ID state file (default: layereddepth-upload-<submission-id>.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    file_path = args.zip_path.expanduser().resolve()
    member_count, uncompressed_size = inspect_zip(file_path, args.verify_zip)
    total_size = file_path.stat().st_size

    client = LayeredDepthClient(
        base_url=args.base_url,
        email=args.email,
        timeout=args.timeout,
        max_retries=args.max_retries,
        cookie_file=args.cookie_file.expanduser().resolve() if args.cookie_file else None,
    )
    client.login(args.code, reuse_code=args.reuse_code)

    explicit_state_path = args.state_file.expanduser().resolve() if args.state_file else None
    existing_state: dict[str, Any] = {}
    if explicit_state_path and explicit_state_path.exists():
        existing_state = json.loads(explicit_state_path.read_text())

    if args.create_submission:
        saved_submission_id = int(existing_state.get("submission_id", 0) or 0)
        submission_id = saved_submission_id or client.create_submission(
            args.create_submission,
            args.benchmark_name,
        )
    else:
        submission_id = int(args.submission_id)

    state_path = explicit_state_path or (
        Path.cwd() / f"layereddepth-upload-{submission_id}.json"
    )
    state = load_or_create_upload_state(
        state_path=state_path,
        file_path=file_path,
        submission_id=submission_id,
        total_size=total_size,
    )
    upload_id = str(state["upload_id"])

    initial_status = client.upload_status(submission_id, upload_id)
    capabilities = initial_status.get("capabilities") or {}
    server_chunk_size = int(
        capabilities.get("recommended_chunk_size", DEFAULT_CHUNK_SIZE)
    )
    max_chunk_size = int(capabilities.get("max_chunk_size", DEFAULT_CHUNK_SIZE))
    chunk_size = int(
        state.get("chunk_size")
        or (
            args.chunk_size_mib * 1024 * 1024
            if args.chunk_size_mib is not None
            else server_chunk_size
        )
    )
    workers = int(
        args.workers
        or capabilities.get("recommended_parallel_uploads", DEFAULT_WORKERS)
    )
    if chunk_size <= 0 or chunk_size > max_chunk_size:
        raise UploadError(
            f"Chunk size must be between 1 byte and {format_bytes(max_chunk_size)}."
        )
    if workers < 1 or workers > 6:
        raise UploadError("--workers must be between 1 and 6.")
    if capabilities.get("binary_multipart_public") is not True:
        raise UploadError("The public binary-multipart transport is not enabled.")

    state.update(
        {
            "submission_id": submission_id,
            "chunk_size": chunk_size,
            "workers": workers,
            "upload_mode": "parallel-direct",
            "transport": "binary-multipart-v1",
        }
    )
    save_upload_state(state_path, state)

    print(
        f"ZIP: {file_path}\n"
        f"Compressed: {format_bytes(total_size)}\n"
        f"Uncompressed: {format_bytes(uncompressed_size)} in {member_count} members\n"
        f"Submission: #{submission_id}\n"
        f"State: {state_path}",
        flush=True,
    )

    if args.status_only:
        payload = initial_status
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.monitor_only:
        monitor_submission(
            client,
            submission_id,
            upload_id,
            args.poll_seconds,
        )
        return 0

    payload = upload_file(
        client=client,
        file_path=file_path,
        submission_id=submission_id,
        upload_id=upload_id,
        chunk_size=chunk_size,
        workers=workers,
    )
    status = str(payload.get("submission_status", "unknown"))
    print(f"Server status after upload: {status}", flush=True)

    if not args.no_monitor:
        monitor_submission(
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
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (
        UploadError,
        urllib_error.URLError,
        TimeoutError,
        socket.timeout,
        ConnectionError,
        OSError,
    ) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
