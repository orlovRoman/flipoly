# Research integration

Base: main c6b3c17.

Integrated sources:
- trade-economics f3ed15c: scenario sensitivity research, not confirmed LIVE costs.
- strike-baseline 44d061c: M1 is a proxy-target prediction baseline. Lack of
  a significant CT/CS improvement is not an equivalence result. NO economics
  remain synthetic; changing proxy strike changes about 9.67% of labels.
- paper-ct-outsider a3d040d: production-pipeline tests, reservation updates,
  and real concurrency/recovery checks. Integration fixes the missing typing.Any
  import in execution/worker.py.
- price-time-map a198f26: independent universe, cross-universe comparison and
  recentered day-block permutation outputs. Its acceptance remains provisional:
  no holdout exists and the study is descriptive gross research.

Repository-wide .gitattributes rules are preserved. The strike-baseline binary
rules are additive. Trade-economics/CURRENT.md is a Windows-compatible pointer
to the final historical scenario run.

Price-time-map is now merged as research artifacts, not as a trading rule:
- With 2000 bootstrap repetitions and 309 Holm tests, the minimum attainable
  first adjusted p-value is 309/2001 (about 0.1544). Zero survivors at 0.05 is
  therefore forced by resolution, not evidence that all cells lack an edge.
- Time comparisons use all rows, including missing quotes, rather than only
  markets with usable quotes at both times; missing data become zero positions.
- Their sign-flip p-values randomize individual markets, unlike the day-block CIs.
- The branch's replacement of .gitignore was resolved by preserving the
  repository rules and adding only freeze-chunk patterns.

These findings do not invalidate the descriptive decline of CT T-5 gross PnL
from 100.10 to 40.56 when expanding the universe. The universes overlap and
must not be described as independent replications.

No trading configuration, model promotion or service deployment is performed
by this integration task. Historical research artifacts are not regenerated.

Validation on the integrated code:
- tests/trading and tests/research: 386 passed, 2 skipped, 11 sklearn warnings.
- CT policy/replay/pipeline subset: 56 passed, 1 skipped.
- strike-baseline synthetic_checks.py: 14 checks passed.
- trade-economics subset before subsequent merges: 21 passed.
