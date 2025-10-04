"""Central definitions for dependency versions and helper utilities."""

from __future__ import annotations

from typing import List, Sequence

TRANSFORMERS_VERSION = "4.46.0"
BITSANDBYTES_VERSION = "0.44.1"
TORCH_VERSION = "2.5.1"


def _collect_requirement_lines(
    include_bitsandbytes: bool,
    include_cuda_note: bool,
    internet_note: str | None,
    extra_lines: Sequence[str] | None,
) -> List[str]:
    lines = [f"- transformers=={TRANSFORMERS_VERSION}"]

    if include_bitsandbytes:
        lines.append(
            f"- bitsandbytes=={BITSANDBYTES_VERSION} (for 4-bit quantization)"
        )

    lines.append(f"- torch=={TORCH_VERSION}")

    if include_bitsandbytes and include_cuda_note:
        lines.append(
            "- CUDA GPU with 4GB+ VRAM recommended (CPU mode available with use_4bit=False)"
        )

    if internet_note:
        lines.append(internet_note)

    if extra_lines:
        lines.extend(extra_lines)

    return lines


def format_transformer_requirements(
    *,
    include_bitsandbytes: bool = False,
    include_cuda_note: bool = False,
    internet_note: str | None = "- Internet connection to download model(s) from HuggingFace Hub",
    extra_lines: Sequence[str] | None = None,
) -> str:
    """Return a formatted requirements message for transformer-based modules.

    Args:
        include_bitsandbytes: Whether to include the bitsandbytes dependency line.
        include_cuda_note: Whether to include the CUDA guidance line when
            bitsandbytes is required.
        internet_note: Custom text for the HuggingFace connectivity guidance. Set
            to ``None`` to omit the line entirely.
        extra_lines: Optional iterable of additional requirement lines.

    Returns:
        A newline-separated string beginning with ``"Requirements:"`` followed
        by bullet points of dependencies.
    """

    requirement_lines = _collect_requirement_lines(
        include_bitsandbytes=include_bitsandbytes,
        include_cuda_note=include_cuda_note,
        internet_note=internet_note,
        extra_lines=extra_lines,
    )

    return "\n".join(["Requirements:", *requirement_lines])


def format_install_instruction(package: str, version: str) -> str:
    """Create a consistent ``pip install`` instruction string."""

    return f"Install with: pip install {package}=={version}"
