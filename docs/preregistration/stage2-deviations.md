# Stage 2 deviations

Changes after freeze 2 (2026-11-01, root `026780b7217957bdfd79a58f734b5468ef8ba794037fa2f7ec33d2b71ac8f36c`). The registration text, the sets file, the plan
manifest and the code were not changed after freeze 2: every held-out job verified them against `freeze-stage2.txt` before its first trial, and every
trial record carries the code digest `d84525f183e8fdf71eb9da3eaaf51a8de1324104de107b025b88dc68069883f0`. What differs from the registered procedure:

1. Workers. The Design section says "8 workers per job". The four OpenRouter held-out jobs (DeepSeek and gpt-oss, primary and default cell) ran with 12
   workers (`WORKERS=12`) so that the DeepSeek job, which needed 3 h 42 min, would fit the 3 h 50 min limit of the partition. The pre-launch probe had been
   run at 8 workers (40 trials, no 429 or 5xx answer for either model). The twelve-worker jobs ended with every planned trial scored, none lost, none
   skipped for the budget and every provider matching the pin. The Haiku jobs ran with 8. The number of workers does not enter the analysis.
2. Order. The registration says Haiku first, then DeepSeek, then gpt-oss, primary before default. Haiku's primary job ran first and its counts were read
   (`stage2 health`, no outcome rate). After that the Haiku default job and the DeepSeek and gpt-oss primary jobs ran at the same time, and the two
   default jobs of the open models last.
3. Commit. No commit or tag exists for either freeze (the owner had not authorised one when they were written). The first commit holds exactly the
   frozen files; `freeze_hashes.py --stage2 --verify` on it must reproduce both manifests, as the registration states.

Nothing else changed. The analysis was run once, after the last job had ended, from the frozen snapshot (`runs/stage2-code`) with the code digest above.
