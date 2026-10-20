#!/usr/bin/env bash
set -euo pipefail
CC="${CC:-cc}"
if [ "$(uname -s)" = "Darwin" ]; then
    $CC -O3 -flto -Wl,-dead_strip -std=c99 n1_codec.c -lm -o encode || $CC -O3 -std=c99 n1_codec.c -lm -o encode
    strip -u -r encode 2>/dev/null || strip encode 2>/dev/null || true
else
    $CC -O3 -flto -fdata-sections -ffunction-sections -Wl,--gc-sections -s -std=c99 n1_codec.c -lm -o encode || $CC -O3 -std=c99 n1_codec.c -lm -o encode
    strip -s encode 2>/dev/null || strip encode 2>/dev/null || true
fi
printf '#!/bin/sh\nexec "${0%%/*}/encode" --decode "$@"\n' > decode
chmod +x encode decode
