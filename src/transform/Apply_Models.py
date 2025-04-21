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