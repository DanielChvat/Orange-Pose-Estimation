import argparse
import functools
import http.server
import os
import socketserver


class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
        ".ply": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the Gaussian splat overlay viewer with SharedArrayBuffer-safe headers.")
    parser.add_argument("--dir", type=str, default="vis/gaussian_splat_overlay")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.dir)
    handler = functools.partial(CrossOriginIsolatedHandler, directory=root)
    with ReusableThreadingTCPServer((args.host, args.port), handler) as httpd:
        print(f"[INFO] Serving {root}")
        print(f"[INFO] Open http://{args.host}:{args.port}/index.html")
        print("[INFO] Press Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
