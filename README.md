# Parallel System: MPI + OpenMP Integration

## Overview

A hybrid parallel program combining **MPI** (distributed memory, inter-process) with
**OpenMP** (shared memory, intra-process threads). Ten MPI processes cooperate to solve
four independent tasks simultaneously, then report all results back to a master process.

---

## Process Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                     10 MPI Processes                            │
│                                                                 │
│  Rank 0  ── Master ──────────────────────────────────────────  │
│              │  distributes tasks, collects & displays results  │
│              │                                                  │
│    ┌─────────┼──────────────────────────┐                      │
│    │         │            │             │                       │
│  Rank 1    Rank 2       Rank 3        Rank 4  ←──── Rank 5-9  │
│  Integer   String        File          Matrix    (sub-workers) │
│  Worker    Worker       Worker        Coordinator               │
│                                                                 │
│  [OpenMP]  [OpenMP]    [OpenMP]       [OpenMP + MPI stripes]   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Workers

### Rank 1 — Integer Worker
- **Input:** integer `n` (default: 12)
- **Operations (OpenMP parallel):**
  - `n!`  — Factorial using `#pragma omp parallel reduction(*:result)`
  - `4^n` — Power of 4 using parallel exponentiation
- **Output:** Both results sent back to master

### Rank 2 — String Worker
- **Input:** string (default: `"racecar"`)
- **Operations (OpenMP parallel):**
  - **Palindrome check** — parallel loop over half the string, `reduction(&&:)`
  - **Vowel count** — `#pragma omp parallel for reduction(+:vowel_count)`
  - **Reverse** — parallel swap with `#pragma omp parallel for`
- **Output:** Packed result string → master

### Rank 3 — File Worker
- **Input:** path to `input.txt`
- **Operations (OpenMP sections):**
  - Read all lines
  - **Even-indexed lines** → `even_lines.txt`
  - **Odd-indexed lines**  → `odd_lines.txt`
  - File writes use `#pragma omp parallel sections`
- **Output:** Line count statistics → master

### Rank 4 — Matrix Coordinator  +  Ranks 5-9 Sub-workers
- **Input:** Two 50×50 float matrices (A and B)
- **Operations:**
  | Operation | Parallelism |
  |-----------|-------------|
  | **A + 1** (scalar add) | Distributed across ranks 5-9 via MPI stripes (10 rows each), each stripe processed with OpenMP |
  | **Aᵀ** (transpose) | `#pragma omp parallel for collapse(2)` on rank 4 |
  | **A .* B** (element-wise multiply) | `#pragma omp parallel for` on rank 4 |
  | **A + B** (matrix addition) | `#pragma omp parallel for` on rank 4 |
- **Output:** Four result matrices → master, plus spot-check verification

---

## Build & Run

### Prerequisites
```bash
# Ubuntu / Debian
sudo apt install openmpi-bin libopenmpi-dev

# Fedora / RHEL
sudo dnf install openmpi openmpi-devel
```

### Compile
```bash
make
# or manually:
mpicc -O2 -fopenmp -o parallel_system parallel_system.c -lm
```

### Run
```bash
make run
# or manually:
mpirun --oversubscribe -np 10 ./parallel_system
```

---

## Files

| File | Description |
|------|-------------|
| `parallel_system.c` | Full source — all 10 process roles |
| `Makefile` | Build & run targets |
| `input.txt` | Sample 12-line file for the file worker |
| `even_lines.txt` | Generated: even-indexed lines (0,2,4,…) |
| `odd_lines.txt` | Generated: odd-indexed lines (1,3,5,…) |

---

## Parallelism Summary

| MPI Rank | Role | OpenMP Usage |
|----------|------|--------------|
| 0 | Master / display | `parallel for` (matrix fill) |
| 1 | Factorial + Power-of-4 | `reduction(*:)` |
| 2 | Palindrome / vowels / reverse | `reduction(&&:)`, `reduction(+:)`, `parallel for` |
| 3 | File line split | `parallel sections` |
| 4 | Matrix coordinator | `collapse(2)`, `parallel for` |
| 5-9 | Matrix stripe workers | `parallel for` on received stripe |

**Total logical threads:** 10 MPI processes × 4 OpenMP threads = **40 threads**

---
