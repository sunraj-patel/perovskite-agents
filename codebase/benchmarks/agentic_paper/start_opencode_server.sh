#!/usr/bin/env bash
# Start the locked-down `opencode serve` used for the OpenCode free-model routes.
#
# Why this exists: the free-tier gateway only accepts requests that still carry
# OpenCode's tool list, so file tools cannot simply be removed. Instead:
#  * the server runs in a fresh EMPTY sandbox directory that is made a git repo
#    root (otherwise OpenCode treats "/" as the project root and the model could
#    grep/read the whole disk, including this benchmark's reference answers);
#  * read/glob/grep/list are deny-by-default (an allow rule for a non-existent
#    path keeps the tool registered but unusable); bash/edit/web/task/skill and
#    external directories are denied;
#  * temperature 0 for the plan agent.
SB="$(mktemp -d "${TMPDIR:-/tmp}/opencode_bench_sandbox.XXXXXX")"
cp "$(dirname "$0")/config/opencode_serve/opencode.json" "$SB/opencode.json"
cd "$SB" && git init -q . && git -c user.email=b@b -c user.name=b commit -q --allow-empty -m init
echo "sandbox: $SB" >&2
exec opencode serve --pure --port "${PEROVAGENTS_OPENCODE_PORT:-4096}" --hostname 127.0.0.1
