#!/usr/bin/env bash
# package.sh -- build the release archive with the name DERIVED from __version__.
#
# The archive shipped as "v0934" for nine releases while the CHANGELOG climbed to v0.9.42, because
# the name was typed at packaging time. It is now computed, and check_version.py fails the build if
# __version__, the CHANGELOG and the directory name disagree.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"

"$PY" check_version.py || { echo "version check FAILED -- fix before packaging" >&2; exit 1; }
read -r VERSION PKGDIR ARCHIVE < <("$PY" check_version.py --print-names)

HERE="$(basename "$PWD")"
if [ "$HERE" != "$PKGDIR" ]; then
  echo "this directory is $HERE but __version__ $VERSION wants $PKGDIR -- rename it first" >&2
  exit 1
fi

rm -f "../$ARCHIVE"
( cd .. && zip -qr "$ARCHIVE" "$PKGDIR" -x '*__pycache__*' '*/output/*' )
echo "built ../$ARCHIVE  (version $VERSION)"
