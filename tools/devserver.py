#!/usr/bin/env python3
"""Local preview server for the site.

Identical to `python3 -m http.server` except it tells the browser never to cache.
Plain http.server lets Chrome hold on to a stale ES module for minutes, which
makes edits to scripts/ask/*.js look like they did nothing.

    python3 tools/devserver.py [port]        # default 8000
"""

import functools
import http.server
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter: only non-200s
        if not args or str(args[1]).startswith("2"):
            return
        super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(NoCacheHandler, directory=ROOT)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {ROOT} at http://localhost:{port} (no-store)")
        httpd.serve_forever()
