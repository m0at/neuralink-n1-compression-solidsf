#!/usr/bin/env bash
set -euo pipefail
CC="${CC:-cc}"
$CC -O3 -flto -Wl,-dead_strip -std=c99 n1_codec.c -lm -o encode
strip -u -r encode 2>/dev/null || strip encode 2>/dev/null || true
printf '#!/bin/sh\nexec "${0%%/*}/encode" --decode "$@"\n' > decode
chmod +x encode decode

