"""Versioned native-tool provisioning and read-only environment diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Callable, Mapping
from contextlib import closing
from importlib import import_module
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, Self
from urllib.request import urlopen
from uuid import uuid4

import psutil
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

DOWNLOAD_CHUNK_SIZE = 64 * 1024
UNIX_FILE_TYPE_MASK = 0o170000
UNIX_SYMBOLIC_LINK = 0o120000
MAXIMUM_ARCHIVE_EXPANSION_RATIO = 25
MINIMUM_EXTRACTION_LIMIT = 64 * 1024 * 1024
MAXIMUM_EXTRACTION_LIMIT = 20 * 1024 * 1024 * 1024


class ToolModel(BaseModel):
    """Reject undeclared fields in signed-off provisioning metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchiveLayout(ToolModel):
    """Name the executable paths expected after extraction."""

    executables: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def paths_are_safe(self) -> Self:
        """Require normalized archive-relative paths."""
        if any(not _safe_member_path(value) for value in self.executables.values()):
            msg = "executable layout paths must be normalized and relative"
            raise ValueError(msg)
        return self


class ToolArtifact(ToolModel):
    """Pin one native artifact for an exact operating-system target."""

    platform: str = Field(pattern=r"^[a-z0-9_]+-[a-z0-9_]+$")
    url: HttpUrl
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive: Literal["zip", "tar", "tar.gz", "tar.xz"]
    layout: ArchiveLayout
    signature_url: HttpUrl | None = None


class ToolSpec(ToolModel):
    """Record native-tool provenance and optional managed artifacts."""

    tool_id: Literal["ffmpeg", "brush", "splat-transform"]
    version: str = Field(min_length=1, max_length=80)
    source_url: HttpUrl
    license_name: str = Field(min_length=1, max_length=100)
    artifacts: tuple[ToolArtifact, ...]


class SampleVideoSpec(ToolModel):
    """Record a redistributable prepared input without silently downloading it."""

    sample_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    version: str = Field(min_length=1, max_length=80)
    url: HttpUrl
    license_name: str = Field(min_length=1, max_length=100)
    attribution: str = Field(min_length=1, max_length=300)
    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,100}$")
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ToolManifest(ToolModel):
    """Describe all reviewed Phase 4 native inputs."""

    schema_version: Literal[1] = 1
    tools: tuple[ToolSpec, ...]
    samples: tuple[SampleVideoSpec, ...] = ()

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> Self:
        """Keep lookup keys stable and unambiguous."""
        identifiers = [tool.tool_id for tool in self.tools]
        if len(identifiers) != len(set(identifiers)):
            msg = "tool identifiers must be unique"
            raise ValueError(msg)
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            msg = "sample identifiers must be unique"
            raise ValueError(msg)
        return self

    def tool(self, tool_id: str) -> ToolSpec:
        """Return one reviewed tool or fail without substitution."""
        match = next((tool for tool in self.tools if tool.tool_id == tool_id), None)
        if match is None:
            msg = f"unknown managed tool: {tool_id}"
            raise KeyError(msg)
        return match


def load_tool_manifest() -> ToolManifest:
    """Load the checked-in native provenance and platform matrix."""
    content = (
        files("better_backgrounds")
        .joinpath("assets/native-tools-v1.json")
        .read_text(encoding="utf-8")
    )
    return ToolManifest.model_validate_json(content)


def platform_key() -> str:
    """Return the exact manifest key for this host."""
    reported_system = platform.system().lower()
    system = {"darwin": "macos"}.get(reported_system, reported_system)
    machine = platform.machine().lower()
    architectures = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    return f"{system}-{architectures.get(machine, machine)}"


def _safe_member_path(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and ":" not in name
        and not path.is_absolute()
        and path.as_posix() == name.rstrip("/")
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _destination(root: Path, name: str) -> Path:
    if not _safe_member_path(name):
        msg = f"unsafe archive member: {name}"
        raise ValueError(msg)
    destination = root.joinpath(*PurePosixPath(name).parts).resolve()
    if not destination.is_relative_to(root.resolve()):
        msg = f"unsafe archive member: {name}"
        raise ValueError(msg)
    return destination


def safe_extract_archive(
    archive: Path,
    destination: Path,
    archive_type: Literal["zip", "tar", "tar.gz", "tar.xz"],
) -> None:
    """Extract regular files and directories without traversal or links."""
    destination.mkdir(parents=True, exist_ok=False)
    try:
        if archive_type == "zip":
            _extract_zip(archive, destination)
        else:
            _extract_tar(archive, destination)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        _validate_extracted_size(archive, sum(member.file_size for member in members))
        for member in members:
            target = _destination(destination, member.filename)
            unix_mode = member.external_attr >> 16
            if unix_mode and (unix_mode & UNIX_FILE_TYPE_MASK) == UNIX_SYMBOLIC_LINK:
                msg = f"archive links are not allowed: {member.filename}"
                raise ValueError(msg)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, target.open("xb") as output:
                shutil.copyfileobj(input_file, output, DOWNLOAD_CHUNK_SIZE)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        _validate_extracted_size(archive, sum(member.size for member in members))
        for member in members:
            target = _destination(destination, member.name)
            if member.issym() or member.islnk():
                msg = f"archive links are not allowed: {member.name}"
                raise ValueError(msg)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                msg = f"unsupported archive member: {member.name}"
                raise ValueError(msg)
            input_file = source.extractfile(member)
            if input_file is None:
                msg = f"unreadable archive member: {member.name}"
                raise ValueError(msg)
            target.parent.mkdir(parents=True, exist_ok=True)
            with input_file, target.open("xb") as output:
                shutil.copyfileobj(input_file, output, DOWNLOAD_CHUNK_SIZE)


def _validate_extracted_size(archive: Path, extracted_size: int) -> None:
    limit = min(
        MAXIMUM_EXTRACTION_LIMIT,
        max(MINIMUM_EXTRACTION_LIMIT, archive.stat().st_size * MAXIMUM_ARCHIVE_EXPANSION_RATIO),
    )
    if extracted_size > limit:
        msg = "archive expands beyond its safe extraction budget"
        raise ValueError(msg)


ResourceOpener = Callable[[str], BinaryIO]


def _urlopen(url: str) -> BinaryIO:
    return urlopen(url, timeout=30)  # noqa: S310


class ToolInstaller:
    """Verify, extract, validate, and atomically publish native tools."""

    def __init__(self, root: Path, *, opener: ResourceOpener = _urlopen) -> None:
        """Use one application-managed root and injectable download boundary."""
        self.root = root
        self._opener = opener

    def install(self, tool_id: str, version: str, artifact: ToolArtifact) -> Path:
        """Install an exact artifact or reuse its verified publication."""
        reusable = self._reuse(tool_id, version, artifact)
        if reusable is not None:
            return reusable
        target = self.root / tool_id / version / artifact.platform
        target.parent.mkdir(parents=True, exist_ok=True)
        archive = target.parent / f".{target.name}.{uuid4().hex}.download"
        try:
            self._download(artifact, archive)
            return self._publish(tool_id, version, artifact, archive)
        finally:
            archive.unlink(missing_ok=True)

    def install_archive(
        self,
        tool_id: str,
        version: str,
        artifact: ToolArtifact,
        archive: Path,
    ) -> Path:
        """Install a transferred archive after the same integrity checks as a download."""
        if archive.stat().st_size != artifact.size or _file_digest(archive) != artifact.sha256:
            msg = "managed tool archive failed integrity verification"
            raise ValueError(msg)
        reusable = self._reuse(tool_id, version, artifact)
        if reusable is not None:
            return reusable
        return self._publish(tool_id, version, artifact, archive)

    def _reuse(
        self,
        tool_id: str,
        version: str,
        artifact: ToolArtifact,
    ) -> Path | None:
        target = self.root / tool_id / version / artifact.platform
        marker = target / ".complete.json"
        expected_marker = {"schema_version": 1, "sha256": artifact.sha256}
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == expected_marker:
                self._validate_layout(target, artifact.layout)
                _make_declared_executables_runnable(target, artifact.layout)
                return target
        except OSError, ValueError, TypeError:
            pass
        return None

    def _publish(
        self,
        tool_id: str,
        version: str,
        artifact: ToolArtifact,
        archive: Path,
    ) -> Path:
        target = self.root / tool_id / version / artifact.platform
        expected_marker = {"schema_version": 1, "sha256": artifact.sha256}
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.{uuid4().hex}.part")
        try:
            safe_extract_archive(archive, staging, artifact.archive)
            self._validate_layout(staging, artifact.layout)
            _make_declared_executables_runnable(staging, artifact.layout)
            (staging / ".complete.json").write_text(
                json.dumps(expected_marker, sort_keys=True),
                encoding="utf-8",
            )
            if target.exists():
                shutil.rmtree(target)
            staging.replace(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return target

    def _download(self, artifact: ToolArtifact, destination: Path) -> None:
        digest = hashlib.sha256()
        written = 0
        with closing(self._opener(str(artifact.url))) as source, destination.open("xb") as output:
            while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
                written += len(chunk)
                if written > artifact.size:
                    msg = "managed tool download exceeded its declared size"
                    raise ValueError(msg)
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written != artifact.size or digest.hexdigest() != artifact.sha256:
            msg = "managed tool download failed integrity verification"
            raise ValueError(msg)

    @staticmethod
    def _validate_layout(root: Path, layout: ArchiveLayout) -> None:
        for name, relative_path in layout.executables.items():
            executable = root.joinpath(*PurePosixPath(relative_path).parts)
            if not executable.is_file():
                msg = f"unexpected archive layout: missing {name} at {relative_path}"
                raise ValueError(msg)


def _is_windows() -> bool:
    return os.name == "nt"


def _make_declared_executables_runnable(root: Path, layout: ArchiveLayout) -> None:
    """Grant owner execution only to manifest-declared POSIX executables."""
    if _is_windows():
        return
    for relative_path in layout.executables.values():
        executable = root.joinpath(*PurePosixPath(relative_path).parts)
        executable.chmod(executable.stat().st_mode | 0o100)


def artifact_filename(artifact: ToolArtifact) -> str:
    """Return the immutable release filename used for offline transfer."""
    path = artifact.url.path
    if path is None:
        msg = "managed tool URL has no artifact filename"
        raise ValueError(msg)
    return PurePosixPath(path).name


class SampleInstaller:
    """Download and atomically publish checksummed prepared video inputs."""

    def __init__(self, root: Path, *, opener: ResourceOpener = _urlopen) -> None:
        """Use an application-managed root and injectable network boundary."""
        self.root = root
        self._opener = opener

    def install(self, sample: SampleVideoSpec) -> Path:
        """Install one exact sample or reuse its verified cached file."""
        target = self.root / sample.sample_id / sample.version / sample.filename
        if (
            target.is_file()
            and target.stat().st_size == sample.size
            and _file_digest(target) == sample.sha256
        ):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.part")
        digest = hashlib.sha256()
        written = 0
        try:
            with closing(self._opener(str(sample.url))) as source, temporary.open("xb") as output:
                while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
                    written += len(chunk)
                    if written > sample.size:
                        msg = "sample download exceeded its declared size"
                        raise ValueError(msg)
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if written != sample.size or digest.hexdigest() != sample.sha256:
                msg = "sample download failed integrity verification"
                raise ValueError(msg)
            target.unlink(missing_ok=True)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class ToolDiagnostic(ToolModel):
    """Describe whether one pinned executable can be invoked."""

    tool_id: str
    available: bool
    path: Path | None = None
    version_output: str | None = Field(default=None, max_length=500)
    problem: str | None = Field(default=None, max_length=500)


class GpuDiagnostic(ToolModel):
    """Report best-effort local adapter and driver evidence."""

    adapter: str | None = Field(default=None, max_length=300)
    driver: str | None = Field(default=None, max_length=300)
    render_apis: tuple[str, ...] = ()
    problem: str | None = Field(default=None, max_length=500)


class DoctorReport(ToolModel):
    """Report host capabilities without changing application state."""

    schema_version: Literal[1] = 1
    platform: str
    python: str
    cpu: str
    memory_bytes: int = Field(ge=0)
    disk_free_bytes: int = Field(ge=0)
    gpu: GpuDiagnostic
    tools: tuple[ToolDiagnostic, ...]
    reconstruction_supported: bool
    sample_mode_supported: bool = True


def _version_probe_argument(tool_id: str) -> str:
    return {
        "ffmpeg": "-version",
        "ffprobe": "-version",
    }.get(tool_id, "--version")


def _diagnose_pycolmap() -> ToolDiagnostic:
    try:
        module = import_module("pycolmap")
        version = str(module.__dict__["__version__"])
        path_value = module.__dict__.get("__file__")
    except (AttributeError, ImportError, OSError) as error:
        return ToolDiagnostic(
            tool_id="pycolmap",
            available=False,
            problem=f"PyCOLMAP could not be imported: {error}"[:500],
        )
    expected = "4.1.0"
    if version != expected:
        return ToolDiagnostic(
            tool_id="pycolmap",
            available=False,
            path=Path(path_value) if path_value else None,
            version_output=f"PyCOLMAP {version}",
            problem=f"Expected PyCOLMAP {expected}.",
        )
    return ToolDiagnostic(
        tool_id="pycolmap",
        available=True,
        path=Path(path_value) if path_value else None,
        version_output=f"PyCOLMAP {version}",
    )


def diagnose_environment(
    executable_paths: Mapping[str, Path | str],
    *,
    storage_root: Path,
) -> DoctorReport:
    """Run small bounded version probes and report actionable failures."""
    diagnostics: list[ToolDiagnostic] = []
    for tool_id in ("ffmpeg", "ffprobe", "pycolmap", "brush", "splat-transform"):
        if tool_id == "pycolmap":
            diagnostics.append(_diagnose_pycolmap())
            continue
        executable = executable_paths.get(tool_id)
        if executable is None:
            diagnostics.append(
                ToolDiagnostic(
                    tool_id=tool_id,
                    available=False,
                    problem="Not installed in the managed tool directory.",
                )
            )
            continue
        command = [str(executable), _version_probe_argument(tool_id)]
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as error:
            diagnostics.append(
                ToolDiagnostic(
                    tool_id=tool_id,
                    available=False,
                    path=Path(executable),
                    problem=str(error)[:500],
                )
            )
        else:
            output = (completed.stdout or completed.stderr).splitlines()
            diagnostics.append(
                ToolDiagnostic(
                    tool_id=tool_id,
                    available=True,
                    path=Path(executable),
                    version_output=(output[0] if output else "version command succeeded")[:500],
                )
            )
    storage_root.mkdir(parents=True, exist_ok=True)
    free = psutil.disk_usage(str(storage_root)).free
    return DoctorReport(
        platform=platform_key(),
        python=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        memory_bytes=psutil.virtual_memory().total,
        disk_free_bytes=free,
        gpu=_diagnose_gpu(),
        tools=tuple(diagnostics),
        reconstruction_supported=all(item.available for item in diagnostics) and free >= 10**10,
    )


def _diagnose_gpu() -> GpuDiagnostic:
    system = platform.system().lower()
    if system == "windows":
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -First 1 Name,DriverVersion | ConvertTo-Json -Compress",
        ]
        apis = ("Direct3D 11", "Direct3D 12 / WebGPU when supported by the driver")
    elif system == "darwin":
        command = ["system_profiler", "SPDisplaysDataType", "-json"]
        apis = ("Metal / WebGPU when supported by Qt WebEngine",)
    else:
        command = ["lspci", "-mm"]
        apis = ("OpenGL", "Vulkan / WebGPU when supported by the driver")
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return GpuDiagnostic(
            render_apis=apis,
            problem=f"GPU adapter query unavailable: {error}"[:500],
        )
    output = completed.stdout.strip()
    if system == "windows":
        try:
            value = json.loads(output)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return GpuDiagnostic(
                adapter=str(value.get("Name") or "Unknown")[:300],
                driver=str(value.get("DriverVersion") or "Unknown")[:300],
                render_apis=apis,
            )
    return GpuDiagnostic(adapter=output[:300] or None, render_apis=apis)


def managed_executable_paths(
    root: Path,
    manifest: ToolManifest | None = None,
) -> dict[str, Path]:
    """Resolve only complete manifest-pinned installations for this host."""
    actual_manifest = manifest or load_tool_manifest()
    current_platform = platform_key()
    result: dict[str, Path] = {}
    for tool in actual_manifest.tools:
        artifact = next(
            (item for item in tool.artifacts if item.platform == current_platform),
            None,
        )
        if artifact is None:
            continue
        installation = root / tool.tool_id / tool.version / current_platform
        marker = installation / ".complete.json"
        try:
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
        except OSError, ValueError, TypeError:
            continue
        if marker_value != {"schema_version": 1, "sha256": artifact.sha256}:
            continue
        for executable_id, relative_path in artifact.layout.executables.items():
            executable = installation.joinpath(*PurePosixPath(relative_path).parts)
            if executable.is_file() and (_is_windows() or os.access(executable, os.X_OK)):
                result[executable_id] = executable
    return result


def project_executable_paths(
    project_root: Path | None = None,
    manifest: ToolManifest | None = None,
) -> dict[str, Path]:
    """Resolve lockfile-pinned development tools without consulting PATH."""
    actual_manifest = manifest or load_tool_manifest()
    try:
        version = actual_manifest.tool("splat-transform").version
    except KeyError:
        return {}
    root = project_root or Path(__file__).resolve().parents[2]
    lockfile = root / "package-lock.json"
    package = root / "node_modules/@playcanvas/splat-transform/package.json"
    executable = (
        root / "node_modules/.bin" / ("splat-transform.cmd" if _is_windows() else "splat-transform")
    )
    try:
        locked = json.loads(lockfile.read_text(encoding="utf-8"))["packages"][
            "node_modules/@playcanvas/splat-transform"
        ]["version"]
        installed = json.loads(package.read_text(encoding="utf-8"))["version"]
    except KeyError, OSError, TypeError, ValueError:
        return {}
    if locked != version or installed != version or not executable.is_file():
        return {}
    if not _is_windows() and not os.access(executable, os.X_OK):
        return {}
    return {"splat-transform": executable}


def resolved_executable_paths(
    managed_root: Path,
    *,
    project_root: Path | None = None,
    manifest: ToolManifest | None = None,
) -> dict[str, Path]:
    """Combine managed binaries with explicitly lockfile-pinned project tools."""
    actual_manifest = manifest or load_tool_manifest()
    return {
        **project_executable_paths(project_root, actual_manifest),
        **managed_executable_paths(managed_root, actual_manifest),
    }
