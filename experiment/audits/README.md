# Dataset audits

`released_data_audit.json` is the deterministic output of:

```bash
python3 experiment/scripts/audit_dataset.py \
  --data-dir replication/MMUbiPred
```

The audit is committed because it describes the immutable benchmark inputs.
Generated training artifacts belong under the ignored `experiment/outputs/`
directory.
