import pandas as pd
import pickle
import os
import numpy as np
import xgboost as xgb

def gen_mod_root():
    # Resolve the project root relative to this file (Apply_Models.py)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_base = os.path.join(project_root, "models")

    return models_base

def apply_serve_model_1(shots_wide):

    models_base = gen_mod_root()
    mod_Serve = pickle.load(open(os.path.join(models_base, "mod_ServeNoServe_211030_halfpt1_ext.pkl"), 'rb'))
    
    # restrict to first 31 data points now
    acc_X_first = shots_wide.iloc[:, 1:1 + 31]
    acc_Y_first = shots_wide.iloc[:, 62:62 + 31]
    acc_Z_first = shots_wide.iloc[:, 123:123 + 31]

    shots_wide["preds"] = mod_Serve.predict(xgb.DMatrix(pd.concat([acc_X_first, acc_Y_first, acc_Z_first], axis=1)))
    shots_wide["Serve_rolling"] = np.where(shots_wide.preds > 0.4, 1, 0)

    return shots_wide
    
def apply_serve_model_2(shots_wide, shots_wide2):    
    models_base = gen_mod_root()
    newservemodvars = []

    for axis in ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]:
        newservemodvars += [f"{axis}_{i}" for i in range(20, 60)]

    acc_X_half = shots_wide2.iloc[:, 2:1 + 42]
    acc_Y_half = shots_wide2.iloc[:, 63:63 + 41]
    acc_Z_half = shots_wide2.iloc[:, 124:124 + 41]
    gyr_X_half = shots_wide2.iloc[:, 185:185 + 41]
    gyr_Y_half = shots_wide2.iloc[:, 246:246 + 41]
    gyr_Z_half = shots_wide2.iloc[:, 307:307 + 41]

    mod_halfplus_all = pickle.load(open(os.path.join(models_base, "mod_ServeNoServe_220104_halfplus_all.pkl"), 'rb'))

    shots_wide["preds_halfplus_all"] = mod_halfplus_all.predict(
        xgb.DMatrix(pd.concat([acc_X_half, acc_Y_half, acc_Z_half,
                            gyr_X_half, gyr_Y_half, gyr_Z_half], axis=1)))
    shots_wide["Serve_halfplus_all"] = np.where(shots_wide.preds_halfplus_all > 0.05, 1, 0)

    mod_halfplus_accx = pickle.load(open(os.path.join(models_base, "mod_ServeNoServe_220104_halfplus_accx.pkl"), 'rb'))

    shots_wide["preds_halfplus_accx"] = mod_halfplus_accx.predict(xgb.DMatrix(acc_X_half))
    shots_wide["Serve_halfplus_accx"] = np.where(shots_wide.preds_halfplus_accx > 0.02, 1, 0)

    # apply the new serve model - 2207 (applied 221028)
    mod_Serve_2207 = pickle.load(open(os.path.join(models_base, "mod_ServeNoServe_220714_all_2060.pkl"), 'rb'))
    
    shots_wide["preds_mod_Serve_2207"] = mod_Serve_2207.predict(xgb.DMatrix(shots_wide2[newservemodvars]))
    shots_wide["Serve2"] = np.where(shots_wide.preds_mod_Serve_2207 > 0.2, 1, 0)

    # temper the "all data" model with help from the other 2 models - in certain situations ignore the threshold set
    shots_wide["Serve"] = np.where((shots_wide.preds <= 0.4) &
                                (shots_wide.preds_halfplus_all < 0.1) &
                                (shots_wide.preds_halfplus_accx < 0.02), 0, shots_wide.Serve_halfplus_all)
    
    return shots_wide, shots_wide2

def gen_results(eval_fin, results, Name, dic):
    # results["Correct"]= np.where( results.Label3 == results.preds, 1, 0)
    eval_fin[f"{Name}_prob"] = results.max(axis=1)
    results['Max_Col'] = results.idxmax(axis=1)

    eval_fin[f"{Name}_pred"] = results.Max_Col.apply(lambda x: dic[x])
    return eval_fin

def apply_slice_mod(shots_wide2):
    models_base = gen_mod_root()
    mod_SliceRes = pickle.load(open(os.path.join(models_base, "ShotId_SliceRes_220121.pkl"), 'rb'))

    dic2 = {}
    dic2["A"] = "BH"
    dic2["B"] = "Slice"
    dic2["C"] = "FH"
    dic2["D"] = "Volley"
    dic2["E"] = "Overhead"

    # create the data so can predict on it
    slice_res_cols = ['Acc_X_00']  # manual one-off start

    for axis in ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]:
        for i in range(21, 42):  # includes 41
            col = f"{axis}_{i:02}"
            slice_res_cols.append(col)

    slice_res = shots_wide2[slice_res_cols]
    probs = mod_SliceRes.predict(xgb.DMatrix(slice_res))
    results = pd.DataFrame({'A': probs[:, 0], 'B': probs[:, 1], 'C': probs[:, 2], 'D': probs[:, 3],
                            'E': probs[:, 4]})
    return results, dic2, slice_res

def apply_BHFocus_mod(slice_res, eval_fin):
    models_base = gen_mod_root()
    mod_BH2040Focus = pickle.load(open(os.path.join(models_base, "ShotId_BH2040Focus_220126.pkl"), 'rb'))

    probs = mod_BH2040Focus.predict(xgb.DMatrix(slice_res))
    eval_fin["BH_2040Focus_prob"] = probs
    eval_fin["BH_2040Focus_pred"] = np.where(eval_fin.BH_2040Focus_prob >= 0.2, "BH", "Other")
    return eval_fin

def apply_generic_mod(shots_wide2, dic2, eval_fin):
    models_base = gen_mod_root()
    mod_ShotId2207 = pickle.load(open(os.path.join(models_base, "ShotId_220716_all_2050.pkl"), 'rb'))

    newshotid_vars  = []

    for axis in ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]:
        newshotid_vars  += [f"{axis}_{i}" for i in range(20, 50)]

    probs_new = mod_ShotId2207.predict(xgb.DMatrix(shots_wide2[newshotid_vars]))
    results = pd.DataFrame({'A': probs_new[:, 0], 'B': probs_new[:, 1], 'C': probs_new[:, 2], 'D': probs_new[:, 3],
                            'E': probs_new[:, 4]})
    eval_fin_extra = gen_results(eval_fin, results, "ShotId_2207", dic2)
    return eval_fin_extra

def combine_preds(shots_wide, eval_fin_extra, points_start_end2, points_2):
    shots_wide[["Key", "preds", "Serve_rolling", "preds_halfplus_all", "Serve_halfplus_all", "preds_halfplus_accx",
                        "Serve_halfplus_accx", "preds_mod_Serve_2207", "Serve2", "Serve"]]

    eval_fin_extra["ComboPred"] = np.where(eval_fin_extra.BH_2040Focus_pred == "BH", "BH",
                                        eval_fin_extra.Slice_Res_pred)
    eval_fin2 = pd.merge(  # eval_fin, #replacing this with the new version that has the new predictions in it
        eval_fin_extra, shots_wide[["Key", "preds", "Serve_rolling", "preds_halfplus_all", "Serve_halfplus_all",
                                    "preds_halfplus_accx", "Serve_halfplus_accx", "Serve", "Serve2",
                                    "preds_mod_Serve_2207"]], how="left", on="Key")

    eval_fin3 = pd.merge_asof(eval_fin2, points_start_end2[["minimum", "GameCount", "PointInGame"]],
                            left_on="TimeTrueStrike", right_on="minimum", direction="backward")
    eval_fin3["GamePoint"] = eval_fin3.GameCount.astype("str") + "_" + eval_fin3.PointInGame.astype("str")

    eval_fin3 = pd.merge(eval_fin3, points_2[["Seconds", "ShotInGame"]], left_on="TimeTrueStrike", right_on="Seconds",
                        how="left")
    eval_fin3["RealShot"] = np.where((eval_fin3.ShotInGame == 1) & (eval_fin3.Serve == 1), "Serve",
                                    np.where(eval_fin3.ShotId_2207_pred == "Overhead", "OH",
                                            eval_fin3.ShotId_2207_pred))
    return eval_fin3

    