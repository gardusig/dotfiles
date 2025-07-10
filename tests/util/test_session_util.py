import pytest
from pathlib import Path

import sys

import crowler.util.session_util as session_util


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    # Patch CACHE_DIR to tmp_path/.cache/cli_history for isolation
    cache_dir = tmp_path / ".cache" / "cli_history"
    monkeypatch.setattr(session_util, "CACHE_DIR", cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield


@pytest.mark.parametrize(
    "env,pty,wt_session,ppid,expected_raw",
    [
        # HISTORY_SESSION_ID takes precedence
        ({"HISTORY_SESSION_ID": "mysession"}, None, None, 1234, "mysession"),
        # _try_pty returns a value if HISTORY_SESSION_ID is not set
        ({}, "/dev/pts/9", None, 1234, "/dev/pts/9"),
        # WT_SESSION used if HISTORY_SESSION_ID and _try_pty are not set
        ({}, None, "wt-session-abc", 1234, "wt-session-abc"),
        # Fallback to parent PID as string
        ({}, None, None, 5678, "5678"),
    ],
)
def test_get_session_id(monkeypatch, env, pty, wt_session, ppid, expected_raw):
    # Patch environment variables
    monkeypatch.setattr(
        session_util.os,
        "getenv",
        lambda k: (
            env.get(k) if k in env else (wt_session if k == "WT_SESSION" else None)
        ),
    )
    # Patch _try_pty
    monkeypatch.setattr(session_util, "_try_pty", lambda: pty)
    # Patch os.getppid
    monkeypatch.setattr(session_util.os, "getppid", lambda: ppid)
    # Patch _hash to check input
    called = {}

    def fake_hash(text):
        called["raw"] = text
        return "hashed"

    monkeypatch.setattr(session_util, "_hash", fake_hash)
    result = session_util.get_session_id()
    assert result == "hashed"
    assert called["raw"] == expected_raw


def test__hash_returns_md5_prefix():
    # Check that _hash returns the first 8 chars of md5
    result = session_util._hash("foobar")
    import hashlib

    expected = hashlib.md5("foobar".encode()).hexdigest()[:8]
    assert result == expected


def test_create_session_file(monkeypatch):
    # Patch get_session_id to return a fixed value
    monkeypatch.setattr(session_util, "get_session_id", lambda: "deadbeef")
    name = "testfile"
    expected_path = session_util.CACHE_DIR / f"{name}.deadbeef.json"
    result = session_util.create_session_file(name)
    assert result == expected_path


@pytest.mark.parametrize(
    "fileno_side_effect,ttyname_side_effect,expected",
    [
        # Success path: os.ttyname returns a string
        (0, lambda fd: "/dev/pts/3", "/dev/pts/3"),
        # Failure: fileno raises
        (lambda: (_ for _ in ()).throw(OSError("fail")), None, None),
        # Failure: os.ttyname raises
        (0, lambda fd: (_ for _ in ()).throw(OSError("fail")), None),
    ],
)
def test__try_pty(monkeypatch, fileno_side_effect, ttyname_side_effect, expected):
    # Patch sys.stdin.fileno
    class DummyStdin:
        def fileno(self):
            if callable(fileno_side_effect):
                return fileno_side_effect()
            return fileno_side_effect

    monkeypatch.setattr(session_util.sys, "stdin", DummyStdin())
    # Patch os.ttyname
    if ttyname_side_effect is not None:
        monkeypatch.setattr(session_util.os, "ttyname", ttyname_side_effect)
    else:
        # Remove os.ttyname to simulate error
        monkeypatch.delattr(session_util.os, "ttyname", raising=False)
    result = session_util._try_pty()
    assert result == expected
