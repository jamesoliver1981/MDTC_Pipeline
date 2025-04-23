import json
from .Create_KPIs import melt_it, stat_func
from .Create_Output import gen_out


def run_summarise_pipeline(eval_table, zip_name):
    print(f"Running summarisation for: {zip_name}")
    d1_2 = melt_it(eval_table)
    df, stats = stat_func(d1_2, suffix="single")  # Step 2: Calculate stats (add debug = True to output updated var list)
    single_result = gen_out(df, d1_2, "single", **stats)       # Step 3: Generate output
    
    # Save summary
    with open(f'data/outputs/{zip_name}_MatchAnalysis.json', 'w') as outfile:
        json.dump(single_result, outfile, indent=4)

    print("Summarisation complete.")