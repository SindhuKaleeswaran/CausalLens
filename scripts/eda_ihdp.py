"""Reproduce Phase 3 training-only EDA for replication 0."""
from causallens.data.eda import run_eda
if __name__ == '__main__':
    table, overlap = run_eda()
    print(table.to_string(index=False))
