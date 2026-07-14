#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec python launch_qtp_exhibit.py "$@"
