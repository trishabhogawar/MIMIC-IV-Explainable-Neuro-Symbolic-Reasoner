# reason_case.py — wrapper that forwards all flags to neuro_symbolic_reasoner.py
import argparse, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", required=True, help="case index (int)")
    ap.add_argument("--model-dir", default="models_lgbm_cv")
    ap.add_argument("--fold", default="1")
    ap.add_argument("--alpha", default="1.0")

    # one-shot edits (exact or concept names)
    ap.add_argument("--do", nargs="*", default=[],
                    help="e.g. med__pressor=0 score__mbp__mean=70 or mbp__mean=70")

    # legacy sweeps
    ap.add_argument("--sweep-meds", choices=["eachoff","eachon","alloff","allon"], default=None)
    ap.add_argument("--sweep-meds-include", type=str, default=None)
    ap.add_argument("--sweep-range", action="append", default=[])
    ap.add_argument("--sweep-out", type=str, default=None)

    # UX helpers
    ap.add_argument("--edit-family", action="store_true")
    ap.add_argument("--print-prec", type=int, default=3)

    # NEW: multi-med variations
    ap.add_argument("--med-scenario", action="append", default=[],
                    help="Repeatable. 'Label|med__a=0,med__b=1' or 'med__a=0,med__b=1'")
    ap.add_argument("--med-grid", type=str, default=None,
                    help="Comma list, e.g. 'med__pressor,med__sedation'")
    ap.add_argument("--grid-values", type=str, default="0,1",
                    help="Comma list of values for the grid (default '0,1')")
    ap.add_argument("--grid-max", type=int, default=128,
                    help="Cap total auto-generated scenarios (default 128)")

    args = ap.parse_args()

    cmd = [
        sys.executable, "neuro_symbolic_reasoner.py",
        "--model-dir", args.model_dir,
        "--fold", str(args.fold),
        "--idx", str(args.idx),
        "--alpha", str(args.alpha),
        "--print-prec", str(args.print_prec),
    ]

    if args.edit_family: cmd += ["--edit-family"]
    if args.do:          cmd += ["--do"] + args.do
    if args.sweep_meds:  cmd += ["--sweep-meds", args.sweep_meds]
    if args.sweep_meds_include: cmd += ["--sweep-meds-include", args.sweep_meds_include]
    for rng in args.sweep_range: cmd += ["--sweep-range", rng]
    if args.sweep_out:   cmd += ["--sweep-out", args.sweep_out]

    # forward NEW multi-med flags
    for s in args.med_scenario: cmd += ["--med-scenario", s]
    if args.med_grid:     cmd += ["--med-grid", args.med_grid]
    if args.grid_values:  cmd += ["--grid-values", args.grid_values]
    if args.grid_max is not None: cmd += ["--grid-max", str(args.grid_max)]

    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
