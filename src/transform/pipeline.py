from .Extract import dezip, create_out, convert_data, create_smooth
from .Feature_Gen import feat_gen, create_points_part1, add_key, shot_prep, shot_prep2
from .Apply_Models import apply_serve_model_1, apply_serve_model_2

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
    
    print(shots_wide.head())
    
    print("[4/4] Pipeline run complete.")

