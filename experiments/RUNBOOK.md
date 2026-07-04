# Runbook — one experiment run (one arm, one seed)

Manual, copy-paste procedure. You are physically present anyway (power meter,
ambient temp), so no orchestration script — just do the steps in order and
don't skip the cooldown gate.

**Placeholders:** `ARM` = `naive` | `sisa` · `SEED` = 42–46 · `HOST_IP` = this
machine's LAN address. Pi is reachable as `admin@rasp5node.local` with key
`~/.ssh/rasp5node` (ssh-add it first if passphrase-protected), repo cloned at
`~/master-thesis` on the Pi.

## One-time setup

```sh
docker compose run --rm preprocessor          # full partition caches (host)
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml build"
```

## 1. Prepare (host)

```sh
python3 experiments/prepare_edge_data.py --seed SEED
scp data/.cache/msc/partition_2_of_3.npz data/.cache/msc/manifest.json \
    admin@rasp5node.local:master-thesis/data/.cache/msc/
ssh admin@rasp5node.local "rm -rf msc-experiment/checkpoints/* /dev/shm/sisa_timings.jsonl /dev/shm/recovery_manifest.json /dev/shm/hardware_telemetry_*.csv"
rm -rf results/msc/global_checkpoints
mkdir -p results/msc/runs/ARM_seedSEED
```

## 2. Cooldown gate + instruments

```sh
experiments/cooldown_gate.sh                  # blocks until SoC <= 40°C for 2 min
ssh admin@rasp5node.local "nohup ./msc-experiment/monitor.sh >/dev/null 2>&1 &"
```

Record ambient temperature. Reset and start FNB58 logging.

## 3. Phase 1 — poisoned federated training (10 rounds)

```sh
ssh admin@rasp5node.local "echo phase1 > /dev/shm/run_marker"
SEED=SEED NUM_ROUNDS=10 RESULTS_SUFFIX=_p1 docker compose -f docker-compose.host.yml up -d
# On the Pi (CLIENT_MODE=standard for naive, sisa for sisa):
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=HOST_IP SEED=SEED CLIENT_MODE=ARM_CLIENT POISON_MODE=flip docker compose -f docker-compose.edge.yml up -d"
# Trigger the run (blocks until all 10 rounds finish):
docker compose -f docker-compose.host.yml run --rm runner
ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"
mv results/msc/global_checkpoints results/msc/runs/ARM_seedSEED/phase1_checkpoints
```

## 4. Phase 3 — recovery on the Pi (primary measurement window)

```sh
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml stop superexec-clientapp-3"
ssh admin@rasp5node.local "echo phase3 > /dev/shm/run_marker"
# naive -> edge_nodes.naive_retrain ; sisa -> edge_nodes.sisa_recover
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml run --rm -v \$PWD:/app -w /app --entrypoint python superexec-clientapp-3 -m edge_nodes.RECOVERY_MODULE"
ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"
```

## 5. Phase 4 — rejoin (5 rounds, resumed from the poisoned global model)

```sh
ssh admin@rasp5node.local "echo phase4 > /dev/shm/run_marker"
cp results/msc/runs/ARM_seedSEED/phase1_checkpoints/round_10.npz results/msc/resume_from.npz
SEED=SEED NUM_ROUNDS=5 RESULTS_SUFFIX=_p4 INIT_FROM_CHECKPOINT=/results/resume_from.npz \
    docker compose -f docker-compose.host.yml up -d superexec-serverapp
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=HOST_IP SEED=SEED CLIENT_MODE=ARM_CLIENT POISON_MODE=drop docker compose -f docker-compose.edge.yml up -d superexec-clientapp-3"
docker compose -f docker-compose.host.yml run --rm runner
ssh admin@rasp5node.local "echo done > /dev/shm/run_marker"
```

Stop FNB58 logging; export its CSV.

## 6. Teardown + collect

```sh
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml down; pkill -f monitor.sh"
docker compose -f docker-compose.host.yml down

RUN=results/msc/runs/ARM_seedSEED
scp "admin@rasp5node.local:/dev/shm/hardware_telemetry_*.csv" \
    admin@rasp5node.local:/dev/shm/recovery_manifest.json \
    admin@rasp5node.local:msc-experiment/checkpoints/recovered_model.pt "$RUN/"
scp admin@rasp5node.local:/dev/shm/sisa_timings.jsonl "$RUN/" 2>/dev/null   # sisa arm only
mv results/msc/fedavg_p1.json "$RUN/results_phase1.json"
mv results/msc/fedavg_p4.json "$RUN/results_phase4.json"
mv results/msc/global_checkpoints "$RUN/phase4_checkpoints"
rm -f results/msc/resume_from.npz
```

Copy the FNB58 export into `$RUN/power_fnb58.csv` and note the ambient
temperature in a `$RUN/notes.txt`.

## 7. Evaluate + analyze (after runs exist)

```sh
python3 -m analysis.evaluate_model "$RUN/recovered_model.pt"
python3 -m analysis.evaluate_model "$RUN/phase4_checkpoints/round_5.npz"
python3 -m analysis.analyze
```

## Order of runs

Counterbalance arms across seeds (A/B, B/A, A/B, B/A, A/B), full cooldown gate
between every run. Clean reference run per seed: same as Phase 1 but
`POISON_MODE=off`, `CLIENT_MODE=standard`, no recovery phases.
