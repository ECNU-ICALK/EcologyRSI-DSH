"""Exclusive local runtime ownership shared by HTTP and CLI."""
import os
import fcntl
from pathlib import Path

class RuntimeOwnerLease:
    """Process-scoped advisory lock for one persistent event ledger."""

    def __init__(self, db_path: str | Path) -> None:
        self._fd: int | None = None
        self.lock_path: Path | None = None
        if str(db_path) == ":memory:":
            return
        resolved_db = Path(db_path).expanduser().resolve()
        resolved_db.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = resolved_db.with_name(resolved_db.name + ".sidecar.lock")
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                "another EcologyRSI-DSH sidecar already owns this database: "
                + str(resolved_db)
            ) from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        owner = f"pid={os.getpid()}\ndatabase={resolved_db}\n".encode("utf-8")
        os.ftruncate(fd, 0)
        os.write(fd, owner)
        os.fsync(fd)

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
