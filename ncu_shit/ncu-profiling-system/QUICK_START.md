# Quick Start - NCU Profiling System

## 3 Files to Copy

```bash
compile.sh              # Compiles CUDA kernels
run_ncu_complete.sh     # Runs NCU profiling (6 metric sections)
parse_ncu.py            # Parses CSVs → JSON + LLM prompt
```

## 3 Commands to Run

```bash
./compile.sh my_kernel.cu              # 1. Compile
./run_ncu_complete.sh ./my_kernel      # 2. Profile
cat ncu_output/llm_prompt.txt          # 3. View results
```

## Outputs

```
ncu_output/
├── speed_of_light.csv         # Bottleneck identification
├── memory_analysis.csv         # Memory metrics
├── compute_analysis.csv        # Compute metrics
├── occupancy.csv              # Occupancy data
├── scheduler_stats.csv        # Scheduler stats
├── warp_stats.csv            # Warp efficiency
├── parsed_metrics.json       # ⭐ Structured data
└── llm_prompt.txt           # ⭐ Ready for GPT-4/Claude
```

## GitHub Actions (Optional)

Add `.github/workflows/ncu-profile.yml` to auto-profile on push.

Requires self-hosted runner on GPU instance:
```bash
./config.sh --url https://github.com/OWNER/REPO --token TOKEN --replace
sudo ./svc.sh install && sudo ./svc.sh start
```

## That's It!

Total setup: 5 minutes (local) or 30 minutes (with GitHub Actions)

