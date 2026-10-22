"""One controlled SCAPS run: explicit wine + explicit (cloned) prefix. Diagnostic only."""
import json, os, re, signal, subprocess, sys, time

wine, prefix, tag, timeout = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out_dir = os.path.dirname(os.path.abspath(__file__))
sdir = os.path.join(prefix, "drive_c", "Program Files (x86)", "Scaps3309")
iv = os.path.join(sdir, "results", f"{tag}_gold.iv")
assert not os.path.exists(iv), "stale IV present"
env = dict(os.environ, WINEPREFIX=prefix)
so, se = open(f"{out_dir}/{tag}.stdout", "w"), open(f"{out_dir}/{tag}.stderr", "w")
t0 = time.time()
proc = subprocess.Popen([wine, "scaps3309.exe", f"{tag}_gold.script"], cwd=sdir, env=env,
                        stdout=so, stderr=se, start_new_session=True)
pgid = os.getpgid(proc.pid)
status, cpu_samples = "timeout", []
while time.time() - t0 < timeout:
    if os.path.exists(iv) and "deduced from calculated IV-curve" in open(iv, errors="ignore").read():
        status = "iv_written"; break
    if proc.poll() is not None:
        status = f"exited_{proc.returncode}"; break
    if int(time.time() - t0) % 20 == 0:
        ps = subprocess.run(["ps", "-axo", "pid,pcpu,etime,command"], capture_output=True, text=True).stdout
        cpu_samples.append([l.strip()[:160] for l in ps.splitlines() if "scaps3309" in l and "ps -ax" not in l])
    time.sleep(1)
elapsed = round(time.time() - t0, 1)
time.sleep(1.5)
rc_before_kill = proc.poll()
try: os.killpg(pgid, signal.SIGKILL)
except ProcessLookupError: pass
try: proc.wait(timeout=15)
except Exception: pass
so.close(); se.close()
time.sleep(1)
left = subprocess.run(["pgrep", "-fl", f"scaps3309|{re.escape(prefix)}"], capture_output=True, text=True).stdout.strip()
txt = open(iv, errors="ignore").read() if os.path.exists(iv) else ""
g = lambda p: (lambda m: float(m.group(1)) if m else None)(re.search(p, txt))
res = dict(tag=tag, wine=wine, prefix=prefix, status=status, elapsed_s=elapsed,
           returncode_before_kill=rc_before_kill, iv_exists=bool(txt),
           iv_mtime=time.ctime(os.path.getmtime(iv)) if txt else None, iv_bytes=len(txt),
           Voc=g(r"Voc =\s+([\d.eE+-]+)\s+Volt"), Jsc=g(r"Jsc =\s+([\d.eE+-]+)\s+mA"),
           FF=g(r"FF =\s+([\d.eE+-]+)\s+%"), PCE=g(r"eta =\s+([\d.eE+-]+)\s+%"),
           convergence_failure="Convergence failure" in txt, leftover_procs=left,
           cpu_samples=cpu_samples[:6])
json.dump(res, open(f"{out_dir}/{tag}.json", "w"), indent=2)
print(json.dumps(res, indent=2))
