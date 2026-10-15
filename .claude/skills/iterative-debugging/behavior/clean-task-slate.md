# Clean Task Slate: BEHAVIOR-1K

Use this before rerunning a seed, a stage or a whole campaign.

1. Decide the scope. One seed is
   `$RUN_ROOT/behavior/<suite>/stage<N>/seed_<NN>/`; archiving or deleting that folder is a
   complete clean slate for that seed. A whole campaign is `$RUN_ROOT/behavior/<suite>/`.
   Never delete individual attempt directories: they are the evidence, including the failures.
2. Rerunning a Stage 1 seed also means deciding what to do with the skills it contributed. If
   they came only from that seed, pull them back out of `skill-library/` before the rerun.
3. **A Stage 2 seed is never rerun to get a better number.** Rerun it only when the run itself
   was invalid, for example a launch crash or a second simulator on the card, and say so in
   `campaign-state.md`.
4. Make sure no Isaac process is left: `pgrep -af 'cap-harness run' | grep -v pgrep` must be
   empty; a lingering Kit process holds several GB of GPU memory and makes the next launch fail.
   Kill by PID, never by pattern from a shell whose own command line contains the pattern.
5. Check RAM and GPU headroom (`free -g`, `nvidia-smi`): at least 12 GB of RAM and the GPU free
   of foreign simulators.
6. Verify SAM3 (8114) and Contact-GraspNet (8115) respond.
7. If the freeze has happened, re-verify it:
   `( cd "$RUN_ROOT/behavior/<suite>" && sha256sum -c frozen-manifest.sha256 )`.
8. Keep learning seeds 26-35 apart from evaluation seeds 1-25; diagnostics use 36-50 and are
   never reported.
9. Workers write inside their own seed directory only; only the coordinator edits the library,
   and only during Stage 1.
