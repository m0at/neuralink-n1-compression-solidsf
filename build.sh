#!/usr/bin/env bash
set -euo pipefail
CC="${CC:-cc}"
$CC -O3 -std=c99 n1_codec.c -lm -o encode
$CC -O3 -std=c99 -DDECODE_MODE n1_codec.c -lm -o decode
strip encode decode 2>/dev/null || true
