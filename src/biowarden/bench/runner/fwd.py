"""Inside the sandbox: forward 127.0.0.1:PORT to a unix socket (stdlib only, run with python3.12).

The sandbox has no network, only loopback. The harness talks HTTP to 127.0.0.1:PORT; this relay
passes the bytes to the benchmark proxy's unix socket, which is bind-mounted into the sandbox.
usage: python3.12 fwd.py PORT /path/to/proxy.sock
"""

import socket
import sys
import threading


def pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def serve(port: int, path: str) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(128)
    while True:
        client, _ = listener.accept()
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(path)
        except OSError:
            client.close()
            continue
        threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
        threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2])
