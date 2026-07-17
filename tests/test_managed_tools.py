"""Managed native-tool and environment diagnostic tests."""

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path  # noqa: TC003

import pytest

from better_backgrounds import managed_tools
from better_backgrounds.managed_tools import (
    ArchiveLayout,
    ToolArtifact,
    ToolInstaller,
    ToolManifest,
    ToolSpec,
    artifact_filename,
    platform_key,
    project_executable_paths,
    safe_extract_archive,
)


def artifact(content: bytes) -> ToolArtifact:
    """Describe one test archive."""
    return ToolArtifact(
        platform="windows-x86_64",
        url="https://example.invalid/tool.zip",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        archive="zip",
        layout=ArchiveLayout(executables={"tool": "bin/tool.exe"}),
    )


def test_manifest_rejects_duplicate_tools() -> None:
    """Keep tool identifiers unambiguous."""
    tool = ToolSpec(
        tool_id="ffmpeg",
        version="1.0",
        source_url="https://example.invalid/source",
        license_name="LGPL-2.1-or-later",
        artifacts=(),
    )

    with pytest.raises(ValueError, match="unique"):
        ToolManifest(tools=(tool, tool))


def test_platform_key_normalizes_apple_silicon_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match the reviewed macOS manifest key instead of Python's Darwin label."""
    monkeypatch.setattr(managed_tools.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(managed_tools.platform, "machine", lambda: "arm64")

    assert platform_key() == "macos-arm64"


@pytest.mark.parametrize(
    ("tool_id", "argument"),
    [("ffmpeg", "-version"), ("brush", "--version")],
)
def test_version_probe_uses_each_cli_contract(tool_id: str, argument: str) -> None:
    """Keep doctor probes compatible with the pinned native CLIs."""
    assert managed_tools._version_probe_argument(tool_id) == argument  # noqa: SLF001


def test_safe_extraction_rejects_archive_traversal(tmp_path: Path) -> None:
    """Never publish files outside the managed staging directory."""
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.exe", b"bad")

    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_archive(archive, tmp_path / "out", "zip")

    assert not (tmp_path / "escape.exe").exists()


def test_safe_extraction_rejects_tar_links(tmp_path: Path) -> None:
    """Do not follow links embedded in native-tool archives."""
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("bin/tool")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        output.addfile(member, io.BytesIO())

    with pytest.raises(ValueError, match="link"):
        safe_extract_archive(archive, tmp_path / "out", "tar")


def test_installer_rejects_unexpected_layout(tmp_path: Path) -> None:
    """Require every manifest-owned executable before atomic publication."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        output.writestr("README", b"missing executable")
    content = buffer.getvalue()
    installer = ToolInstaller(tmp_path / "tools", opener=lambda _url: io.BytesIO(content))

    with pytest.raises(ValueError, match="layout"):
        installer.install("ffmpeg", "1.0", artifact(content))

    assert not (tmp_path / "tools" / "ffmpeg" / "1.0" / "windows-x86_64").exists()


def test_installer_accepts_only_exact_offline_archive(tmp_path: Path) -> None:
    """Apply download integrity rules to manually transferred artifacts."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        output.writestr("bin/tool.exe", b"tool")
    content = buffer.getvalue()
    specification = artifact(content)
    archive = tmp_path / artifact_filename(specification)
    archive.write_bytes(content)
    installer = ToolInstaller(tmp_path / "tools")

    installed = installer.install_archive("ffmpeg", "1.0", specification, archive)

    assert installed.joinpath("bin/tool.exe").is_file()
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        installer.install_archive("ffmpeg", "1.0", specification, archive)


def test_installer_grants_posix_execution_to_declared_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make downloaded tools runnable when archive modes are not preserved."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        output.writestr("bin/tool.exe", b"tool")
    content = buffer.getvalue()
    monkeypatch.setattr(managed_tools, "_is_windows", lambda: False)
    installer = ToolInstaller(tmp_path / "tools", opener=lambda _url: io.BytesIO(content))

    installed = installer.install("ffmpeg", "1.0", artifact(content))

    assert installed.joinpath("bin/tool.exe").stat().st_mode & 0o100


def test_project_tool_requires_matching_lockfile_and_installed_version(tmp_path: Path) -> None:
    """Resolve the npm CLI only when both reviewed version records agree."""
    executable = (
        tmp_path
        / "node_modules/.bin"
        / (
            "splat-transform.cmd" if managed_tools._is_windows() else "splat-transform"  # noqa: SLF001
        )
    )
    executable.parent.mkdir(parents=True)
    executable.write_text("shim", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | 0o100)
    package = tmp_path / "node_modules/@playcanvas/splat-transform/package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"version":"2.7.1"}', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{"node_modules/@playcanvas/splat-transform":{"version":"2.7.1"}}}',
        encoding="utf-8",
    )
    manifest = ToolManifest(
        tools=(
            ToolSpec(
                tool_id="splat-transform",
                version="2.7.1",
                source_url="https://github.com/playcanvas/splat-transform/tree/v2.7.1",
                license_name="MIT",
                artifacts=(),
            ),
        ),
    )

    assert project_executable_paths(tmp_path, manifest) == {
        "splat-transform": executable,
    }

    package.write_text('{"version":"2.8.0"}', encoding="utf-8")

    assert project_executable_paths(tmp_path, manifest) == {}
