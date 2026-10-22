#!/bin/bash
# Diagnostic: F1 22 wswine wine64 (Wine 10 Sikarugir) + its own Frameworks on the dylib path.
APP="/Users/sunraj/Applications/F1 22 AnkerGames.app/Contents"
export DYLD_FALLBACK_LIBRARY_PATH="$APP/Frameworks:$APP/Frameworks/wswine.bundle/lib:/usr/lib"
exec "$APP/Frameworks/wswine.bundle/bin/wine64" "$@"
