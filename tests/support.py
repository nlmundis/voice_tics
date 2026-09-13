"""Fixtures the test modules share: a hermetic working directory, and scratch
files that are gone when the module finishes.

WHY THE WORKING DIRECTORY MOVES
    Both entry points read ``./voice_tics.toml`` when no ``--config`` is given,
    which is the supported zero-config behaviour. A suite that calls ``main()``
    from the repo root therefore reads whatever config the developer happens to
    keep there, so a personal ``[lint] error`` list or a ``[baseline] samples``
    glob turned ``make check`` red for reasons that had nothing to do with the
    code. Each module that calls ``main()`` runs inside an empty temporary
    directory instead.

WHY SCRATCH FILES LIVE IN THAT DIRECTORY
    ``NamedTemporaryFile(delete=False)`` with no cleanup left fifteen files in
    TMPDIR on every run. Writing them under the module's own directory means
    one ``cleanup()`` removes them all, including after a failing test.

Not named ``test_*`` so discovery does not collect it.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Union

_stack: List[tempfile.TemporaryDirectory] = []
_previous_cwd: List[str] = []


def enter_hermetic_cwd() -> None:
    """Make an empty temporary directory the working directory.

    Call from ``setUpModule``; pair with ``leave_hermetic_cwd``.
    """
    tmp = tempfile.TemporaryDirectory(prefix="voice_tics-tests-")
    _stack.append(tmp)
    _previous_cwd.append(os.getcwd())
    os.chdir(tmp.name)


def leave_hermetic_cwd() -> None:
    """Restore the previous working directory and delete everything written.

    Call from ``tearDownModule``.
    """
    os.chdir(_previous_cwd.pop())
    _stack.pop().cleanup()


def scratch_file(content: Union[str, bytes], suffix: str = ".md",
                 name: Optional[str] = None) -> str:
    """Write ``content`` to a new file in the current hermetic directory.

    Args:
        content: Text is written as UTF-8; bytes are written as given, for
            tests that need an undecodable file.
        suffix: File extension, used when ``name`` is not given.
        name: An exact file name, for tests that need a specific one.

    Returns:
        The file's absolute path. It is removed by ``leave_hermetic_cwd``.
    """
    if not _stack:
        raise RuntimeError("scratch_file needs enter_hermetic_cwd first")
    directory = _stack[-1].name
    if name is None:
        fd, path = tempfile.mkstemp(suffix=suffix, dir=directory)
        os.close(fd)
    else:
        path = os.path.join(directory, name)
    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
    with open(path, mode, **kwargs) as fh:
        fh.write(content)
    return path
