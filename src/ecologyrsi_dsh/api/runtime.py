"""Blocking local HTTP server lifecycle."""

from pathlib import Path


def serve(*, host: str = "127.0.0.1", port: int = 8765, db: str = "ecologyrsi-dsh.sqlite3") -> None:
    from ..server import EvolutionHTTPServer

    server = EvolutionHTTPServer((host, port), Path(db).expanduser())
    try:
        print(f"EcologyRSI-DSH local API: http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
