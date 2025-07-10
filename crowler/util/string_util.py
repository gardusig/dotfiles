from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from crowler.instruction.instruction_model import Instruction
from builtins import print

# ─────────────────────────── regex ──────────────────────────────
DEFAULT_FENCE = "~~~"
_FENCE_RE = re.escape(DEFAULT_FENCE)
QUOTES = "\"'`"


CODE_BLOCK_RE = re.compile(
    rf"""
    {_FENCE_RE}                   # opening fence, optional indent
    (?P<quote>[{QUOTES}])              #   opening quote
    (?P<path>[^\r\n]+?)                #   path (no EOL)
    (?P=quote)                         #   closing quote
    (?P<body>.*?)                 # body (non-greedy)
    {_FENCE_RE}                   # closing fence, its own line
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)


# ────────────────────────── public API ──────────────────────────
def parse_code_response(
    response: str,
    root: Union[str, Path, None] = None,
) -> OrderedDict[str, str]:
    files: OrderedDict[str, str] = OrderedDict()
    base = Path(root).resolve() if root else None

    for match in CODE_BLOCK_RE.finditer(response):
        raw_path = match.group("path").strip().strip(QUOTES)
        code = match.group("body")

        if not raw_path:
            continue

        norm = Path(raw_path).as_posix()

        if base and not (base / norm).resolve().is_relative_to(base):
            print("Rejected path outside root sandbox: %s", norm)
            continue

        if norm in files:
            print("Duplicate file path in response: %s (overwriting)", norm)

        files[norm] = code
        print("Extracted %s (len=%d)", norm, len(code))

    if not files:
        print("No file/code blocks found in model response.")
    else:
        print("Decoded %d file(s) from model response.", len(files))

    return files


def get_instruction_strings(
    instructions: Optional[list[Instruction]] = None,
) -> list[str]:
    instruction_strings: list[str] = []
    if instructions:
        for instruction in instructions:
            instruction_strings.extend(instruction.instructions)
    return instruction_strings
