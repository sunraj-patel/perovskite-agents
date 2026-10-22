#!/bin/bash
# Diagnostic wrapper: CrossOver 26.3 wineloader with explicit WINEPREFIX, no bottle.
CX="/Users/sunraj/Applications/CrossOver.app/Contents/SharedSupport/CrossOver"
export WINESERVER="$CX/bin/wineserver" WINELOADER="$CX/bin/wineloader"
export WINEDLLPATH="$CX/lib/wine"
export DYLD_FALLBACK_LIBRARY_PATH="$CX/lib64:$CX/lib:/usr/lib"
exec "$CX/bin/wineloader" "$@"
