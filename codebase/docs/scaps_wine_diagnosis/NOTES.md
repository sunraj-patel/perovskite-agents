# SCAPS live-run regression, diagnosis (18 Oct 2026)

Known-good case: `gold_q13_pin.def` (md5 9c5761ea...) + `gold_q13_pin.script`, run 8 Oct 19:28,
Voc 1.3056 V, Jsc 27.129 mA/cm2, FF 90.28 %, PCE 31.98 %. (`live_19771.iv`, 9 Oct 19:03, used
`perovagents_probe.def`, which has since been overwritten by the availability probe.)
Same def, script (IV name only differs), scaps3309.exe (md5 f6415ef0...), 240 s timeout, APFS clones
of the production prefix, runs strictly sequential, explicit wine + prefix every time.

| Run | Runtime | Env change | Result |
|---|---|---|---|
| A  | F1 22 wine64, wine-10.0 Sikarugir | none (as the code ran) | timeout, 100% CPU, no IV; stderr: "Wine cannot find the FreeType font library" |
| B  | CrossOver 26.3 wine-11.0 (wineloader wrapper) | CX lib paths | IV in 53 s, identical to gold |
| B2 | same, fresh clone | | IV in 51 s, identical |
| A2 | F1 22 wine-10.0 | + its own `Contents/Frameworks` on DYLD_FALLBACK_LIBRARY_PATH | IV in 46 s, identical |
| A3 | same, fresh clone | | IV in 46 s, IV body byte-identical to the 8 Oct original |

Conclusion: the regression is the missing FreeType/dylib path, not the Wine version. Wine 10 and
Wine 11 both work. Fix: `config.scaps_env()`.
