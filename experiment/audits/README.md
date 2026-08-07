# Dataset audits

`released_data_audit.json` is the deterministic output of:

```bash
python3 experiment/scripts/audit_dataset.py \
  --data-dir replication/MMUbiPred
```

The committed audit describes the immutable benchmark inputs. Generated
training artifacts are stored under the ignored `experiment/outputs/`
directory.
