from .Extract import dezip, create_out, convert_data, create_smooth
from .Feature_Gen import (
    feat_gen,
    create_points_part1,
    add_key,
    shot_prep,
    shot_prep2,
    points_prep,
    create_match,
    create_points_part2,
    mk_pts_start_end
)
from .Apply_Models import (
    apply_serve_model_1,
    apply_serve_model_2,
    apply_slice_mod,
    gen_results,
    apply_BHFocus_mod,
    apply_generic_mod,
    combine_preds
)

def run_transform_pipeline(zip_path):
    print("[1/4] Extracting zip contents...")

    extract_path = dezip(zip_path)

    print(f"[2/4] Files extracted to: {extract_path}")

    print("[3/4] Reading JSON files and building meta...")
    (
        motion,
        meta_out,
        missing_sensor,
        rightOrLeft,
        bornYear,
        gender,
        rating_lev,
        rating_typ,
        matchResult,
        matchLevel,
        matchSurface,
        matchType, 
        meta_id, 
        maxtime
    ) = create_out(extract_path)
    
    d_in = convert_data(motion, rightOrLeft)
        
    df = create_smooth(d_in, 0, int(maxtime))
    
    df_all, df_shots, mins = feat_gen(df)
    points, shots = create_points_part1(df_shots)
    
    shots2 = add_key(shots, points)
    shots_wide = shot_prep(shots2)

    shots_wide = apply_serve_model_1(shots_wide)

    shots_wide2 = shot_prep2(shots2)

    shots_wide, shots_wide2 = apply_serve_model_2(shots_wide, shots_wide2)    

    ng, points, game_pre = points_prep(points, shots_wide)

    match = create_match(points, df_all, ng)

    points_2 = create_points_part2(points, match)

    points_start_end2 = mk_pts_start_end(points_2, game_pre, df_all)

    eval_fin = shots_wide2[["Key", "TimeTrueStrike"]]

    results, dic2, slice_res = apply_slice_mod(shots_wide2)

    eval_fin = gen_results(eval_fin, results, "Slice_Res", dic2)

    eval_fin = apply_BHFocus_mod(slice_res, eval_fin)

    eval_fin_extra = apply_generic_mod(shots_wide2, dic2, eval_fin)

    eval_fin3 = combine_preds(shots_wide, eval_fin_extra, points_start_end2, points_2)
    
    print(eval_fin3.head())
    
    print("[4/4] Pipeline run complete.")

