import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

def melt_it(d1):
    d1_2 = pd.melt(d1, id_vars=["Label_0", "Label"], value_vars=["Frequency", "Effective_1"])
    return d1_2

def limit_eff(datei, mini, div):
    '''creates the effective score with limits defined to 100'''
    if div == 0:
        out = 0
    else:
        maxi = div - mini
        if datei < - mini:
            out = 1 / (div / 100)
        elif datei > maxi:
            out = 99
        else:
            out = (datei + mini) / (div /100)
    return out

def stat_func(df, suffix = "single"):
    lost = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Lost") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    won = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Won") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    ptsplayed = won + lost


    Won_thru_Forced = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Won_thru_Mistake = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Won_thru_Winner = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Lost_thru_Forced = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Lost_thru_Mistake = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Lost_thru_Winner = df[(df.Label_0 =="TotalPointsWonOrLost_Freq") & (df.Label == "Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    serve_eff_freq = (df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Serve") & (df.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum() / 
                    df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Serve") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum())

    ### serves ###
    #first serves total

    firstserves_freq = df[(df.Label_0 =="FirstServes_by_ServeFrom_Freq") & (df.Label == "Serve_Total") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstServe_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Serve_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstServe_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Serve_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstServe_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Serve_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    firstServe_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Serve_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    firstServe_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Serve_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstServe_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Serve_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstServe_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Serve_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    #where proportion of points as rally over 50%, add a factor to the serve cal
    if (firstServe_total_rally / firstserves_freq) > 0.5:
        firstServe_RallyFactor = firstServe_total_rally - (firstserves_freq * 0.5)
    else: 
        firstServe_RallyFactor = 0

    if df[(df.Label_0 =="Effective_Score_First") & (df.Label == "Effective_Score_First_Serve") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum() ==0:
        firstserve_eff_freq=0
    else:
        firstserve_eff_freq= ((df[(df.Label_0 =="Effective_Score_First") & (df.Label == "Effective_Score_First_Serve") & (df.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum() + firstServe_RallyFactor * -10 ) / 
                    df[(df.Label_0 =="Effective_Score_First") & (df.Label == "Effective_Score_First_Serve") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum())
 
    #first serves deuce
    deuce_firstserves_freq = df[(df.Label_0 =="FirstServes_by_ServeFrom_Freq") & (df.Label == "Serve_Deuce") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_allserves_freq = df[(df.Label_0 =="TotalServes_by_ServeFrom_Freq") & (df.Label == "Serve_Deuce") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstserves_rate = deuce_firstserves_freq / deuce_allserves_freq

    deuce_firstServe_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Serve_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstServe_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Serve_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstServe_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Serve_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    deuce_firstServe_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Serve_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    deuce_firstServe_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Serve_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstServe_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Serve_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstServe_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Serve_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    #first serves adv
    adv_firstserves_freq = df[(df.Label_0 =="FirstServes_by_ServeFrom_Freq") & (df.Label == "Serve_Adv") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_allserves_freq = df[(df.Label_0 =="TotalServes_by_ServeFrom_Freq") & (df.Label == "Serve_Adv") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstserves_rate = adv_firstserves_freq / adv_allserves_freq

    adv_firstServe_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Serve_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstServe_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Serve_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstServe_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Serve_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    adv_firstServe_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Serve_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    adv_firstServe_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Serve_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstServe_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Serve_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstServe_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Serve_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()


    #second serves total
    if df[(df.Label_0 =="Effective_Score_Second") & (df.Label == "Effective_Score_Second_Serve") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum() ==0:
        secondserve_eff_freq = 0
    else:
        secondserve_eff_freq =(df[(df.Label_0 =="Effective_Score_Second") & (df.Label == "Effective_Score_Second_Serve") & (df.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum() / 
                    df[(df.Label_0 =="Effective_Score_Second") & (df.Label == "Effective_Score_Second_Serve") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum())

    secondserves_freq = df[(df.Label_0 =="SecondServes_by_CriticalHL_Origin") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 0).sum()
    secondServe_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Serve_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondServe_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Serve_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondServe_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Serve_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    secondServe_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Serve_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    secondServe_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Serve_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondServe_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Serve_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondServe_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Serve_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()


    #second serves deuce
    Deuce_secondserves_freq = df[(df.Label_0 =="SecondServes_by_CriticalHL_Origin") & (df.Label.isin(["Serve_Critical_Deuce","Serve_NonCritical_Deuce"])) & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 0).sum()
    Deuce_secondServe_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Serve_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondServe_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Serve_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondServe_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Serve_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Deuce_secondServe_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Serve_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Deuce_secondServe_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Serve_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondServe_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Serve_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondServe_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Serve_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    #second serves deuce
    Adv_secondserves_freq = df[(df.Label_0 =="SecondServes_by_CriticalHL_Origin") & (df.Label.isin(["Serve_Critical_Adv","Serve_NonCritical_Adv"])) & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 0).sum()

    Adv_secondServe_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Serve_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondServe_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Serve_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondServe_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Serve_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Adv_secondServe_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Serve_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Adv_secondServe_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Serve_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondServe_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Serve_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondServe_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Serve_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    #outcomes
    firstserverate = firstserves_freq / (firstserves_freq+ secondserves_freq)
    dfs = df[(df.Label_0 == "Start_SplitBy_Serve_Type_Origin_Part_OutCome") & (df.Label.isin(["Serve_Second_Deuce_One_Lost_thru_Mistake", "Serve_Second_Adv_One_Lost_thru_Mistake"]))].sum(axis =1).sum()
    df_rate = dfs / secondserves_freq

    firstserve_won = (firstServe_total_wonwinner + firstServe_total_wonforced + firstServe_total_wonmistake) / firstserves_freq
    firstserve_rally = ( firstServe_total_rally) / firstserves_freq
    firstserve_lost = (firstServe_total_lostwinner + firstServe_total_lostforced + firstServe_total_lostmistake) / firstserves_freq

    secondserve_won = (secondServe_total_wonwinner + secondServe_total_wonforced + secondServe_total_wonmistake) / secondserves_freq
    secondserve_rally = ( secondServe_total_rally) / secondserves_freq
    secondserve_lost = (secondServe_total_lostwinner + secondServe_total_lostforced + secondServe_total_lostmistake) / secondserves_freq

    #### returns
    return_eff_freq = (df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Return") & (df.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum() / 
                    df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Return") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum())


    #first serves total
    if df[(df.Label_0 =="Effective_Score_First") & (df.Label == "Effective_Score_First_Return") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum() ==0:
        firstreturn_eff_freq =0
    else:
        firstreturn_eff_freq = (df[(df.Label_0 =="Effective_Score_First") & (df.Label == "Effective_Score_First_Return") & (df.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum() / 
                    df[(df.Label_0 =="Effective_Score_First") & (df.Label == "Effective_Score_First_Return") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum())


    firstreturns_freq = df[(df.Label_0 =="FirstServesReturnHit_by_ServeFrom_Freq") & (df.Label == "Return_Total") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstReturn_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Return_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstReturn_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Return_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstReturn_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Return_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    firstReturn_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Return_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    firstReturn_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Return_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstReturn_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Return_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    firstReturn_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_First") & (df.Label == "Return_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    #first serves deuce
    deuce_firstreturns_freq = df[(df.Label_0 =="FirstServesReturnHit_by_ServeFrom_Freq") & (df.Label == "Return_Deuce") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstReturn_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Return_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstReturn_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Return_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstReturn_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Return_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    deuce_firstReturn_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Return_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    deuce_firstReturn_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Return_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstReturn_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Return_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    deuce_firstReturn_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_First") & (df.Label == "Return_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    #first serves adv
    adv_firstreturns_freq = df[(df.Label_0 =="FirstServesReturnHit_by_ServeFrom_Freq") & (df.Label == "Return_Adv") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstReturn_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Return_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstReturn_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Return_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstReturn_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Return_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    adv_firstReturn_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Return_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    adv_firstReturn_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Return_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstReturn_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Return_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    adv_firstReturn_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_First") & (df.Label == "Return_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()


    #second serves total
    if df[(df.Label_0 =="Effective_Score_Second") & (df.Label == "Effective_Score_Second_Return") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum() ==0:
        secondreturn_eff_freq =0
    else:
        secondreturn_eff_freq = (df[(df.Label_0 =="Effective_Score_Second") & (df.Label == "Effective_Score_Second_Return") & (df.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum() / 
                    df[(df.Label_0 =="Effective_Score_Second") & (df.Label == "Effective_Score_Second_Return") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum())


    secondreturns_freq = df[(df.Label_0 =="SecondServesReturnHit_by_ServeFrom_Freq") & (df.Label == "Return_Total") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 0).sum()
    secondReturn_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Return_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondReturn_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Return_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondReturn_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Return_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    secondReturn_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Return_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    secondReturn_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Return_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondReturn_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Return_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    secondReturn_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Second") & (df.Label == "Return_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()


    #second serves deuce
    Deuce_secondreturns_freq = df[(df.Label_0 =="SecondServesReturnHit_by_ServeFrom_Freq") & (df.Label == "Return_Deuce") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 0).sum()
    Deuce_secondReturn_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Return_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondReturn_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Return_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondReturn_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Return_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Deuce_secondReturn_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Return_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Deuce_secondReturn_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Return_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondReturn_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Return_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Deuce_secondReturn_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Deuce_Second") & (df.Label == "Return_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    #second serves deuce
    Adv_secondreturns_freq = df[(df.Label_0 =="SecondServesReturnHit_by_ServeFrom_Freq") & (df.Label == "Return_Adv") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 0).sum()

    Adv_secondReturn_total_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Return_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondReturn_total_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Return_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondReturn_total_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Return_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Adv_secondReturn_total_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Return_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    Adv_secondReturn_total_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Return_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondReturn_total_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Return_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    Adv_secondReturn_total_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective_Adv_Second") & (df.Label == "Return_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    made_firstreturns_freq = df[(df.Label_0 =="FirstServesReturnMade_by_ServeFrom_Freq") & (df.Label == "Return_Total") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()


    made_secondreturns_freq = df[(df.Label_0 =="SecondServesReturnMade_by_ServeFrom_Freq") & (df.Label == "Return_Total") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    firstreturn_won = (firstReturn_total_wonwinner + firstReturn_total_wonforced + firstReturn_total_wonmistake) / firstreturns_freq
    firstreturn_rally = ( firstReturn_total_rally) / firstreturns_freq
    firstreturn_lost = (firstReturn_total_lostwinner + firstReturn_total_lostforced + firstReturn_total_lostmistake) / firstreturns_freq

    secondreturn_won = (secondReturn_total_wonwinner + secondReturn_total_wonforced + secondReturn_total_wonmistake) / secondreturns_freq
    secondreturn_rally = ( secondReturn_total_rally) / secondreturns_freq
    secondreturn_lost = (secondReturn_total_lostwinner + secondReturn_total_lostforced + secondReturn_total_lostmistake) / secondreturns_freq


    ###rally data###
    if df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum() ==0:
        rally_eff_freq =0
    else:    
        rally_eff_freq = (df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Rally") & (df.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).values[0] / 
                    df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).values[0])

    # rally_freq = df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).values[0]
    rally_freq = df[(df.Label_0 =="Effective_Score") & (df.Label == "Effective_Score_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    rally_lostwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective") & (df.Label == "Rally_Lost_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    rally_lostforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective") & (df.Label == "Rally_Lost_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    rally_lostmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective") & (df.Label == "Rally_Lost_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    rally_rally = df[(df.Label_0 =="Eval1_By_OutCome_Effective") & (df.Label == "Rally_Rally") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    rally_wonwinner = df[(df.Label_0 =="Eval1_By_OutCome_Effective") & (df.Label == "Rally_Won_thru_Winner") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    rally_wonforced = df[(df.Label_0 =="Eval1_By_OutCome_Effective") & (df.Label == "Rally_Won_thru_Forced") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
    rally_wonmistake = df[(df.Label_0 =="Eval1_By_OutCome_Effective") & (df.Label == "Rally_Won_thru_Mistake") & (df.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

    serve_won = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Serve_Won")].iloc[:,3:].sum(axis = 1).sum()
    serve_lost = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Serve_Lost")].iloc[:,3:].sum(axis = 1).sum()

    return_won = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Return_Won")].iloc[:,3:].sum(axis = 1).sum()
    return_lost = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Return_Lost")].iloc[:,3:].sum(axis = 1).sum()

    rally_324_won = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Rally_3to4_shots_Won")].iloc[:,3:].sum(axis = 1).sum()
    rally_324_lost = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Rally_3to4_shots_Lost")].iloc[:,3:].sum(axis = 1).sum()

    rally_5plus_won = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Rally_over4_shots_Won")].iloc[:,3:].sum(axis = 1).sum()
    rally_5plus_lost = df[(df.Label_0 == "RallyLen_Breakdown") & (df.Label == "RallyLen_Rally_over4_shots_Lost")].iloc[:,3:].sum(axis = 1).sum()

    first_serve_noncrit = (df[(df.Label_0 == "FirstServes_by_CriticalHL") & (df.Label == "Serve_NonCritical")].iloc[:,3:].sum(axis = 1).sum() /
                        df[(df.Label_0 == "TotalServes_by_CriticalHL") & (df.Label == "Serve_NonCritical")].iloc[:,3:].sum(axis = 1).sum())

    first_serve_crit = (df[(df.Label_0 == "FirstServes_by_CriticalHL") & (df.Label == "Serve_Critical")].iloc[:,3:].sum(axis = 1).sum() /
                        df[(df.Label_0 == "TotalServes_by_CriticalHL") & (df.Label == "Serve_Critical")].iloc[:,3:].sum(axis = 1).sum())

    first_serve_crit_freq = df[(df.Label_0 == "FirstServes_by_CriticalHL") & (df.Label == "Serve_Critical")].iloc[:,3:].sum(axis = 1).sum()


    ###critical vs non critical elements
    #1st serve
    noncrit_1stserve_freq = df[(df.Label_0 == "FirstServes_by_CriticalHL") & (df.Label == "Serve_NonCritical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    noncrit_totalserve_freq = df[(df.Label_0 == "TotalServes_by_CriticalHL") & (df.Label == "Serve_NonCritical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    crit_1stserve_freq = df[(df.Label_0 == "FirstServes_by_CriticalHL") & (df.Label == "Serve_Critical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    crit_totalserve_freq = df[(df.Label_0 == "TotalServes_by_CriticalHL") & (df.Label == "Serve_Critical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    if (crit_1stserve_freq / crit_totalserve_freq) >= (noncrit_1stserve_freq / noncrit_totalserve_freq):
        up_1st_serve = 1
    else:
        up_1st_serve = 0
        
    #second serves
    noncrit_2ndservenonloss_freq = df[(df.Label_0 == "SecondServes_by_CriticalHL_OutcomeGen") & (df.Label.isin(["Serve_NonCritical_Rally", "Serve_NonCritical_Won"])) & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    noncrit_2ndserve_freq = df[(df.Label_0 == "SecondServes_by_CriticalHL") & (df.Label == "Serve_NonCritical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    crit_2ndservenonloss_freq = df[(df.Label_0 == "SecondServes_by_CriticalHL_OutcomeGen") & (df.Label.isin(["Serve_Critical_Rally", "Serve_Critical_Won"])) & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    crit_secondserve_freq = df[(df.Label_0 == "SecondServes_by_CriticalHL") & (df.Label == "Serve_Critical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    if (crit_2ndservenonloss_freq / crit_secondserve_freq) >= (noncrit_2ndservenonloss_freq / noncrit_2ndserve_freq):
        up_2nd_serve = 1
    else:
        up_2nd_serve = 0

    breakball_1stserves = df[(df.Label_0 == "FirstServes_by_Critical_lv2_OutcomeGen") & (df.Label.str.contains("Serve_Crit_Breakball_"))].sum(axis=1).sum()
    breakball_allserves = df[(df.Label_0 == "TotalServes_by_Critical_lv2") & (df.Label == "Serve_Crit_Breakball")].sum(axis=1).sum()
    breakball_1stserverate = breakball_1stserves / breakball_allserves

    crit_nonbreakball_1stserves = df[(df.Label_0 == "FirstServes_by_Critical_lv2_OutcomeGen") & (df.Label.str.contains("Serve_Crit_NonBreak_"))].sum(axis=1).sum()
    crit_nonbreakball_allserves = df[(df.Label_0 == "TotalServes_by_Critical_lv2") & (df.Label == "Serve_Crit_NonBreak")].sum(axis=1).sum()
    crit_nonbreakball_1stserverate = crit_nonbreakball_1stserves / crit_nonbreakball_allserves
        
    #1st serve return
    noncrit_1streturnMADE_freq = df[(df.Label_0 == "FirstServeReturnMADE_by_CriticalHL") & (df.Label == "Return_NonCritical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    noncrit_1streturnHIT_freq = df[(df.Label_0 == "FirstServeReturn_by_CriticalHL") & (df.Label == "Return_NonCritical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    crit_1streturnMADE_freq = df[(df.Label_0 == "FirstServeReturnMADE_by_CriticalHL") & (df.Label == "Return_Critical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    crit_1streturnHIT_freq = df[(df.Label_0 == "FirstServeReturn_by_CriticalHL") & (df.Label == "Return_Critical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    if (crit_1streturnMADE_freq / crit_1streturnHIT_freq) >= (noncrit_1streturnMADE_freq / noncrit_1streturnHIT_freq):
        up_1st_return = 1
    else:
        up_1st_return = 0

    #2nd serve return
    noncrit_2ndreturnMADE_freq = df[(df.Label_0 == "SecondServeReturnMADE_by_CriticalHL") & (df.Label == "Return_NonCritical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    noncrit_2ndreturnHIT_freq = df[(df.Label_0 == "SecondServeReturn_by_CriticalHL") & (df.Label == "Return_NonCritical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    crit_2ndreturnMADE_freq = df[(df.Label_0 == "SecondServeReturnMADE_by_CriticalHL") & (df.Label == "Return_Critical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()
    crit_2ndreturnHIT_freq = df[(df.Label_0 == "SecondServeReturn_by_CriticalHL") & (df.Label == "Return_Critical") & (df.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()

    if crit_2ndreturnHIT_freq == 0:
        criticalPercentage2ndReturns = 0
    else:
        criticalPercentage2ndReturns = crit_2ndreturnMADE_freq / crit_2ndreturnHIT_freq *100

    if (crit_2ndreturnMADE_freq / crit_2ndreturnHIT_freq) >= (noncrit_2ndreturnMADE_freq / noncrit_2ndreturnHIT_freq):
        up_2nd_return = 1
    else:
        up_2nd_return = 0



    # nl = "\n"
    #note need to add in negative scales here: np where( less than 0 and 100)
    #go through json file and replace these with the values

    if firstserves_freq + secondserves_freq == 0:
        totalserve_eff = 0
    else:
        totalserve_eff = limit_eff((((firstserve_eff_freq* firstserves_freq) + (secondserve_eff_freq*secondserves_freq))/ (firstserves_freq + secondserves_freq)) , 2.5, 8.5)
    
    firstserve_eff = limit_eff(firstserve_eff_freq, 3, 9.5)
    secondserve_eff = limit_eff(secondserve_eff_freq, 4, 13)
    
    if firstreturns_freq + secondreturns_freq ==0:
        totalreturn_eff = 0
    else: 
        totalreturn_eff =limit_eff((((firstreturn_eff_freq *firstreturns_freq) + (secondreturn_eff_freq * secondreturns_freq)) / (firstreturns_freq + secondreturns_freq)) , 4, 10.5)
    firstreturn_eff = limit_eff(firstreturn_eff_freq, 4, 9)
    secondreturn_eff = limit_eff(secondreturn_eff_freq, 5, 12) 
    
    firstreturn_rate = made_firstreturns_freq / firstreturns_freq * 100
    secondreturn_rate = made_secondreturns_freq / secondreturns_freq * 100

    #rally -eff score
    rally_eff = limit_eff(rally_eff_freq, 1.5, 5.5)
    #rally_points - points ended within the rally - ie remove rally_rally
    rally_points = rally_freq - rally_rally
    # rally win loss balance - based on rally points to enhance the size
    rally_win_balance = (rally_wonwinner + rally_wonforced + rally_wonmistake - rally_lostwinner - rally_lostforced - rally_lostmistake) / rally_points
    if rally_win_balance > 0 :
        rally_win_balance2 = f"+{rally_win_balance:.0%}"
    else :
        rally_win_balance2 = f"{rally_win_balance:.0%}"
    # rally prop - based on rally_freq as its part of the effective score
    rally_prop = rally_rally / rally_freq

    rally_length = rally_freq / rally_points
    # risk balance - based on rally points to enhance the size
    rally_risk_balance = (rally_wonwinner + rally_wonforced - rally_lostmistake) / rally_points

    # if rally_risk_balance > 0:
    #     rally_risk_balance2 = f"+{rally_risk_balance:.0%}"
    # else:
    #     rally_risk_balance2 = f"{rally_risk_balance:.0%}"
        
    # rally_determine = (rally_wonwinner + rally_wonforced +  - rally_lostwinner - rally_lostforced ) / rally_points
    # if rally_determine > 0:
    #     rally_determine2 = f"+{rally_determine:.0%}"
    # else:
    #     rally_determine2 = f"{rally_determine:.0%}"

    # rally_mistakes = (rally_wonmistake - rally_lostmistake ) / rally_points
    # if rally_mistakes > 0:
    #     rally_mistakes2 = f"+{rally_mistakes:.0%}"
    # else:
    #     rally_mistakes2 = f"{rally_mistakes:.0%}"

    BH_shots = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_BH") & (df.variable == "Frequency")].sum(axis =1).sum()
    FH_shots = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_FH") & (df.variable == "Frequency")].sum(axis =1).sum()
    Slice_shots = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_Slice") & (df.variable == "Frequency")].sum(axis =1).sum()
    Volley_shots = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_Volley") & (df.variable == "Frequency")].sum(axis =1).sum()
    OH_shots = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_OH") & (df.variable == "Frequency")].sum(axis =1).sum()


    All_shots = BH_shots + FH_shots + Slice_shots + Volley_shots + OH_shots

    BH_prop = BH_shots / All_shots
    FH_prop = FH_shots / All_shots
    Slice_prop = Slice_shots / All_shots
    Volley_prop = Volley_shots / All_shots
    OH_prop = OH_shots / All_shots

    BH_effraw = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_BH") & (df.variable == "Effective_1")].sum(axis =1).sum() / BH_shots
    FH_effraw = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_FH") & (df.variable == "Effective_1")].sum(axis =1).sum()/ FH_shots
    Slice_effraw = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_Slice") & (df.variable == "Effective_1")].sum(axis =1).sum()/ Slice_shots
    if Volley_shots == 0:
        Volley_effraw = 0
    else:
        Volley_effraw = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_Volley") & (df.variable == "Effective_1")].sum(axis =1).sum()/ Volley_shots
    # OH_effraw = np.where(OH_shots == 0.0 , 0, df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_OH") & (df.variable == "Effective_1")].sum(axis =1).sum()/ OH_shots)
    if OH_shots == 0:
        OH_effraw = 0
    else:
        OH_effraw = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label == "Effective_Score_OH") & (df.variable == "Effective_1")].sum(axis =1).sum()/ OH_shots

    if OH_shots+ Volley_shots == 0:
        Net_effraw = 0
    else:
        Net_effraw = df[(df.Label_0 == "Effective_Score2_Shots") & ( df.Label.isin(["Effective_Score_OH","Effective_Score_Volley"]) ) & (df.variable == "Effective_1")].sum(axis =1).sum()/ (OH_shots+ Volley_shots)

    BH_eff = limit_eff(BH_effraw, 5, 10)
    FH_eff = limit_eff(FH_effraw, 5, 10)
    Slice_eff = limit_eff(Slice_effraw, 5, 10)

    FH_Won = df[(df.Label_0 == "ShotsEval2_By_OutComeGen_Effective") & (df.Label == "FH_Won") & (df.variable == "Frequency")].sum(axis=1).sum()
    FH_Lost = df[(df.Label_0 == "ShotsEval2_By_OutComeGen_Effective") & (df.Label == "FH_Lost") & (df.variable == "Frequency")].sum(axis=1).sum()
    FH_WinBalance = (FH_Won - FH_Lost) / FH_shots
    if FH_WinBalance > 0:
        FH_WinBalance2 = f"+ {FH_WinBalance*100:.0f}"
    else:
        FH_WinBalance2 = f"{FH_WinBalance*100:.0f}"

    BH_Won = df[(df.Label_0 == "ShotsEval2_By_OutComeGen_Effective") & (df.Label == "BH_Won") & (df.variable == "Frequency")].sum(axis=1).sum()
    BH_Lost = df[(df.Label_0 == "ShotsEval2_By_OutComeGen_Effective") & (df.Label == "BH_Lost") & (df.variable == "Frequency")].sum(axis=1).sum()
    BH_WinBalance = (BH_Won - BH_Lost) / BH_shots
    if BH_WinBalance>0:
        BH_WinBalance2 = f"+ {BH_WinBalance*100:.0f}"
    else:
        BH_WinBalance2 = f"{BH_WinBalance*100:.0f}"
    BH_Rally = df[(df.Label_0 == "ShotsEval2_By_OutComeGen_Effective") & (df.Label == "BH_Rally") & (df.variable == "Frequency")].sum(axis=1).sum()
    BH_RallyRate = BH_Rally / BH_shots

    Slice_Won = df[(df.Label_0 == "ShotsEval2_By_OutComeGen_Effective") & (df.Label == "Slice_Won") & (df.variable == "Frequency")].sum(axis=1).sum()
    Slice_Lost = df[(df.Label_0 == "ShotsEval2_By_OutComeGen_Effective") & (df.Label == "Slice_Lost") & (df.variable == "Frequency")].sum(axis=1).sum()
    Slice_LossRate= (Slice_Lost) / Slice_shots
    Slice_WinBalance = (Slice_Won - Slice_Lost) / Slice_shots
    if Slice_WinBalance>0:
        Slice_WinBalance2 = f"+ {Slice_WinBalance*100:.0f}"
    else:
        Slice_WinBalance2 = f"{Slice_WinBalance*100:.0f}"


    FH_ratio = FH_shots / (BH_shots + Slice_shots)
    BH_Slice_ratio = BH_shots / Slice_shots

    All_Time = df[(df.Label_0 == "HRTimeSpentInZone") & (df.variable == "Frequency")].sum(axis= 1).sum()
    All_Points = df[(df.Label_0 == "TotalPts_byHRZone_Freq") & (df.variable == "Frequency")].sum(axis= 1).sum()
    All_Wins = df[(df.Label_0 == "Wins_byHRZone_Freq") &  (df.variable == "Frequency")].sum(axis= 1).sum()
    All_Mistakes = df[(df.Label_0 == "Mistakes_byHRZone_Freq") & (df.variable == "Frequency")].sum(axis= 1).sum()


    Hard_Time = df[(df.Label_0 == "HRTimeSpentInZone") & (df.Label == "Hard Effort 80-90%") & (df.variable == "Frequency")].sum(axis= 1).sum()
    Hard_Points = df[(df.Label_0 == "TotalPts_byHRZone_Freq") & (df.Label == "Hard Effort 80-90%_TotalPts") & (df.variable == "Frequency")].sum(axis= 1).sum()
    Hard_Wins = df[(df.Label_0 == "Wins_byHRZone_Freq") & (df.Label == "Hard Effort 80-90%_Won") & (df.variable == "Frequency")].sum(axis= 1).sum()
    Hard_Mistakes = df[(df.Label_0 == "Mistakes_byHRZone_Freq") & (df.Label == "Hard Effort 80-90%_Lost_thru_Mistake") & (df.variable == "Frequency")].sum(axis= 1).sum()

    Hard_Time_prop = Hard_Time / All_Time
    Hard_WinRate = Hard_Wins / Hard_Points
    Hard_MistakeRate = Hard_Mistakes / Hard_Points

    Max_Time = df[(df.Label_0 == "HRTimeSpentInZone") & (df.Label == "Max Effort 90%+") & (df.variable == "Frequency")].sum(axis= 1).sum()
    Max_Points = df[(df.Label_0 == "TotalPts_byHRZone_Freq") & (df.Label == "Max Effort 90%+_TotalPts") & (df.variable == "Frequency")].sum(axis= 1).sum()
    Max_Wins = df[(df.Label_0 == "Wins_byHRZone_Freq") & (df.Label == "Max Effort 90%+_Won") & (df.variable == "Frequency")].sum(axis= 1).sum()
    Max_Mistakes = df[(df.Label_0 == "Mistakes_byHRZone_Freq") & (df.Label == "Max Effort 90%+_Lost_thru_Mistake") & (df.variable == "Frequency")].sum(axis= 1).sum()

    Max_Time_prop = Max_Time / All_Time
    Max_WinRate = Max_Wins / Max_Points
    Max_MistakeRate = Max_Mistakes / Max_Points

    # ### Averages - Define standards for comparison
    # firstserverate_average = 0.66
    # df_rate_average = 0.18
    # firstserve_eff_average = (1.8 + 2)/ (6.5/100)
    # secondserve_eff_average = (-1.6 + 3)/ (6/100)
    # firstreturn_rate_average = 0.82
    # secondreturn_rate_average = 0.83
    # firstreturn_eff_average = (-0.7 + 4) / (6/100)
    # secondreturn_eff_average = (-0.1 + 4) / (8/100)
    # TotalReturn_error_average = 248 / 2103

    # rally_eff_average = ((1208/1225)+1.5)/(5.5/100)
    # rally_win_balance_average = (569 - 547) / 1116
    # rally_risk_balance_average = (202 + 117 - 314) / 1116
    # rally_det_balance_average = (202 + 117 - 157 - 76) / 1116
    # rally_mistake_balance_average = (250 - 314) / 1116
    # rally_length_average = 1225/1116

    #
    firstserve_serveonly_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_Serve") & (df.variable == "Frequency")].sum(axis=1).sum()
    firstserve_serveFH_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_FH") & (df.variable == "Frequency")].sum(axis=1).sum()
    firstserve_serveBH_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_BH") & (df.variable == "Frequency")].sum(axis=1).sum()
    firstserve_serveSlice_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_Slice") & (df.variable == "Frequency")].sum(axis=1).sum()

    firstservesOnlyProp = (firstserve_serveonly_freq / firstserves_freq)
    serveplusplayed = firstserves_freq - firstserve_serveonly_freq
    FH_of_ServePlus1 = firstserve_serveFH_freq / serveplusplayed
    Slice_of_ServePlus1 = firstserve_serveSlice_freq / serveplusplayed
    BH_of_ServePlus1 = firstserve_serveBH_freq / serveplusplayed

    firstserve_serveonly_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_Serve") & (df.variable == "Effective_1")].sum(axis=1).sum()
    firstserve_serveFH_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_FH") & (df.variable == "Effective_1")].sum(axis=1).sum()
    firstserve_serveBH_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_BH") & (df.variable == "Effective_1")].sum(axis=1).sum()
    firstserve_serveSlice_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "FirstServe_Slice") & (df.variable == "Effective_1")].sum(axis=1).sum()

    # firstserve_serveFH_eff = ((firstserve_serveFH_effraw / firstserve_serveFH_freq) + 2) / (6.5 / 100)
    firstserve_serveonly_eff = limit_eff((firstserve_serveonly_effraw / firstserve_serveonly_freq), 3, 8.5)
    firstserve_serveFH_eff = limit_eff((firstserve_serveFH_effraw / firstserve_serveFH_freq), 3, 9.5)
    if firstserve_serveBH_effraw ==0:
        firstserve_serveBH_eff = 0
    else:
        firstserve_serveBH_eff = limit_eff((firstserve_serveBH_effraw / firstserve_serveBH_freq), 9, 8.5)

    if firstserve_serveSlice_effraw == 0:
        firstserve_serveSlice_eff = 0
    else:
        firstserve_serveSlice_eff = limit_eff((firstserve_serveSlice_effraw / firstserve_serveSlice_freq), 3, 9.5)

    secondserve_serveonly_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "SecondServe_Serve") & (df.variable == "Frequency")].sum(axis=1).sum()
    secondserve_serveFH_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "SecondServe_FH") & (df.variable == "Frequency")].sum(axis=1).sum()
    secondserve_serveBH_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "SecondServe_BH") & (df.variable == "Frequency")].sum(axis=1).sum()
    secondserve_serveSlice_freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "SecondServe_Slice") & (df.variable == "Frequency")].sum(axis=1).sum()

    secondservesOnlyProp = (secondserve_serveonly_freq / secondserves_freq)
    serveplusplayed_2nd = secondserves_freq - secondserve_serveonly_freq
    FH_of_ServePlus1_2nd = secondserve_serveFH_freq / serveplusplayed_2nd
    Slice_of_ServePlus1_2nd = secondserve_serveSlice_freq / serveplusplayed_2nd
    BH_of_ServePlus1_2nd = secondserve_serveBH_freq / serveplusplayed_2nd

    secondserve_serveonly_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "secondServe_Serve") & (df.variable == "Effective_1")].sum(axis=1).sum()
    secondserve_serveFH_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "secondServe_FH") & (df.variable == "Effective_1")].sum(axis=1).sum()

    secondserve_serveBH_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "secondServe_BH") & (df.variable == "Effective_1")].sum(axis=1).sum()
    secondserve_serveSlice_effraw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label == "secondServe_Slice") & (df.variable == "Effective_1")].sum(axis=1).sum()

    # secondserve_serveFH_eff = ((secondserve_serveFH_effraw / secondserve_serveFH_freq) + 2) / (6.5 / 100)
    secondserve_serveonly_eff = limit_eff((secondserve_serveonly_effraw / secondserve_serveonly_freq), 4, 11)
    secondserve_serveFH_eff = limit_eff((secondserve_serveFH_effraw / secondserve_serveFH_freq), 4, 11)
    if secondserve_serveBH_freq == 0:
        secondserve_serveBH_eff = 0
    else: 
        secondserve_serveBH_eff = limit_eff((secondserve_serveBH_effraw / secondserve_serveBH_freq), 4, 11)
    if secondserve_serveSlice_freq == 0:
        secondserve_serveSlice_eff = 0
    else:
        secondserve_serveSlice_eff = limit_eff((secondserve_serveSlice_effraw / secondserve_serveSlice_freq), 4, 11)


    #returns used-  first
    # Shots played on 1st - higher threshold			
    # Effectiveness of Shot on First - higher threshold required here			
    FirstReturn_FH_Eff_raw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="FirstReturn_FH") & (df.variable == "Effective_1")].sum(axis=1).sum()
    FirstReturn_BH_Eff_raw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="FirstReturn_BH") & (df.variable == "Effective_1")].sum(axis=1).sum()
    FirstReturn_Slice_Eff_raw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="FirstReturn_Slice") & (df.variable == "Effective_1")].sum(axis=1).sum()

    FirstReturn_FH_Freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="FirstReturn_FH") & (df.variable == "Frequency")].sum(axis=1).sum()
    FirstReturn_BH_Freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="FirstReturn_BH") & (df.variable == "Frequency")].sum(axis=1).sum()
    FirstReturn_Slice_Freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="FirstReturn_Slice") & (df.variable == "Frequency")].sum(axis=1).sum()

    FirstReturn_FH_prop = FirstReturn_FH_Freq / firstreturns_freq
    FirstReturn_BH_prop = FirstReturn_BH_Freq / firstreturns_freq
    FirstReturn_Slice_prop = FirstReturn_Slice_Freq / firstreturns_freq

    FirstReturn_FH_Eff = limit_eff((FirstReturn_FH_Eff_raw / FirstReturn_FH_Freq), 4, 9)
    FirstReturn_BH_Eff = limit_eff((FirstReturn_BH_Eff_raw / FirstReturn_BH_Freq), 4, 9)
    FirstReturn_Slice_Eff = limit_eff((FirstReturn_Slice_Eff_raw / FirstReturn_Slice_Freq), 4, 9)

    # second return
    SecondReturn_FH_Eff_raw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="SecondReturn_FH") & (df.variable == "Effective_1")].sum(axis=1).sum()
    SecondReturn_BH_Eff_raw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="SecondReturn_BH") & (df.variable == "Effective_1")].sum(axis=1).sum()
    SecondReturn_Slice_Eff_raw = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="SecondReturn_Slice") & (df.variable == "Effective_1")].sum(axis=1).sum()

    SecondReturn_FH_Freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="SecondReturn_FH") & (df.variable == "Frequency")].sum(axis=1).sum()
    SecondReturn_BH_Freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="SecondReturn_BH") & (df.variable == "Frequency")].sum(axis=1).sum()
    SecondReturn_Slice_Freq = df[(df.Label_0 == "ServeReturn_ShotEff") & (df.Label =="SecondReturn_Slice") & (df.variable == "Frequency")].sum(axis=1).sum()

    SecondReturn_FH_prop = SecondReturn_FH_Freq / secondreturns_freq
    SecondReturn_BH_prop = SecondReturn_BH_Freq / secondreturns_freq
    SecondReturn_Slice_prop = SecondReturn_Slice_Freq / secondreturns_freq


    SecondReturn_FH_Eff = limit_eff((SecondReturn_FH_Eff_raw / SecondReturn_FH_Freq), 5 ,12)
    if SecondReturn_BH_Freq ==0:
        SecondReturn_BH_Eff = 0
    else:
        SecondReturn_BH_Eff = limit_eff((SecondReturn_BH_Eff_raw / SecondReturn_BH_Freq), 5, 12)
    if SecondReturn_Slice_Freq == 0:
        SecondReturn_Slice_Eff = 0
    else:   
        SecondReturn_Slice_Eff = limit_eff((SecondReturn_Slice_Eff_raw / SecondReturn_Slice_Freq),5, 12)

        
    # Total Returns Shot Split - error rates			
    # do a total error rate if its above 40 returns
    BH_Return_Mistakes = df[(df.Label_0 == "Shots_By_Part_OutCome_Shot") & (df.Label == "Return_One_Lost_thru_Mistake_BH" ) & (df.variable == "Frequency")].sum(axis=1).sum()
    FH_Return_Mistakes = df[(df.Label_0 == "Shots_By_Part_OutCome_Shot") & (df.Label == "Return_One_Lost_thru_Mistake_FH" ) & (df.variable == "Frequency")].sum(axis=1).sum()
    Slice_Return_Mistakes = df[(df.Label_0 == "Shots_By_Part_OutCome_Shot") & (df.Label == "Return_One_Lost_thru_Mistake_Slice" ) & (df.variable == "Frequency")].sum(axis=1).sum()
    Total_Return_Mistakes = BH_Return_Mistakes + FH_Return_Mistakes + Slice_Return_Mistakes

    BH_Returns = df[(df.Label_0 == "ReturnShot") & (df.Label == "Count_BH") & (df.variable == "Frequency")].sum(axis=1).sum()
    FH_Returns = df[(df.Label_0 == "ReturnShot") & (df.Label == "Count_FH") & (df.variable == "Frequency")].sum(axis=1).sum()
    Slice_Returns = df[(df.Label_0 == "ReturnShot") & (df.Label == "Count_Slice") & (df.variable == "Frequency")].sum(axis=1).sum()
    Total_Returns = BH_Returns + FH_Returns + Slice_Returns

    FH_Return_prop = FH_Returns / Total_Returns
    BH_Return_prop = BH_Returns / Total_Returns
    Slice_Return_prop = Slice_Returns / Total_Returns

    TotalReturn_ErrorRate = (Total_Return_Mistakes/Total_Returns)

        # --- End of stat_func: build and return suffixed dictionary ---

    stat_keys = [
        "serve_won", "serve_lost", "return_won", "return_lost", "won", "lost",
        "rally_324_won", "rally_324_lost", "rally_5plus_won", "rally_5plus_lost", "totalserve_eff",
        "return_eff_freq", "rally_eff_freq", "firstserverate", "firstserve_eff", "secondserve_eff",
        "df_rate", "secondserves_freq", "crit_totalserve_freq", "crit_1stserve_freq", "noncrit_1stserve_freq",
        "noncrit_totalserve_freq", "breakball_allserves", "breakball_1stserverate", "deuce_allserves_freq",
        "adv_allserves_freq", "firstserves_freq", "deuce_firstserves_rate", "adv_firstserves_rate",
        "firstserve_serveonly_freq", "firstserve_serveFH_freq", "firstserve_serveBH_freq", "serveplusplayed",
        "FH_of_ServePlus1", "firstserve_serveFH_eff", "BH_of_ServePlus1", "firstserve_serveBH_eff",
        "Slice_of_ServePlus1", "firstserve_serveSlice_eff", "secondserve_serveonly_freq",
        "secondserve_serveFH_freq", "secondserve_serveBH_freq", "FH_of_ServePlus1_2nd", "secondserve_serveFH_eff",
        "BH_of_ServePlus1_2nd", "secondserve_serveBH_eff", "Slice_of_ServePlus1_2nd", "secondserve_serveSlice_eff",
        "firstreturn_rate", "secondreturn_rate", "secondreturns_freq", "firstreturns_freq", "firstreturn_eff",
        "secondreturn_eff", "Total_Returns", "FH_Return_prop", "BH_Return_prop", "Slice_Return_prop",
        "FirstReturn_FH_prop", "FirstReturn_BH_prop", "FirstReturn_Slice_prop", "FirstReturn_FH_Eff",
        "FirstReturn_BH_Eff", "FirstReturn_Slice_Eff", "SecondReturn_FH_prop", "SecondReturn_BH_prop",
        "SecondReturn_Slice_prop", "SecondReturn_FH_Eff", "SecondReturn_BH_Eff", "SecondReturn_Slice_Eff",
        "rally_points", "rally_eff", "rally_win_balance", "rally_win_balance2", "rally_length", "All_shots",
        "FH_prop", "BH_prop", "Slice_prop", "Volley_prop", "OH_prop", "FH_eff", "FH_shots", "FH_WinBalance2",
        "FH_WinBalance", "BH_eff", "BH_shots", "BH_WinBalance2", "BH_RallyRate", "BH_WinBalance", "Slice_eff",
        "Slice_shots", "Slice_LossRate", "Slice_WinBalance2", "firstserve_lost", "firstserve_rally",
        "firstserve_won", "firstservesOnlyProp", "dfs", "secondServe_total_lostwinner",
        "secondServe_total_lostforced", "secondServe_total_lostmistake", "secondserve_lost",
        "crit_nonbreakball_allserves", "crit_nonbreakball_1stserverate", "firstreturn_lost", "firstreturn_won",
        "secondreturn_lost", "secondreturn_won", "secondreturn_rally", "crit_1streturnHIT_freq",
        "noncrit_1streturnHIT_freq", "rally_prop", "FH_ratio", "BH_Slice_ratio", "ptsplayed",
        "Won_thru_Winner", "Lost_thru_Mistake", "firstServe_total_lostmistake", "firstServe_total_rally",
        "firstServe_total_wonwinner", "deuce_firstServe_total_lostmistake", "deuce_firstServe_total_rally",
        "deuce_firstServe_total_wonwinner", "deuce_firstserves_freq", "adv_firstServe_total_lostmistake",
        "adv_firstServe_total_rally", "adv_firstServe_total_wonwinner", "adv_firstserves_freq",
        "secondServe_total_rally", "secondServe_total_wonwinner", "Deuce_secondServe_total_lostmistake",
        "Deuce_secondServe_total_rally", "Deuce_secondServe_total_wonwinner", "Deuce_secondserves_freq",
        "Adv_secondServe_total_lostmistake", "Adv_secondServe_total_rally", "Adv_secondServe_total_wonwinner",
        "Adv_secondserves_freq", "secondserve_rally", "secondserve_won", "totalreturn_eff",
        "firstReturn_total_lostmistake", "firstReturn_total_rally", "firstReturn_total_wonwinner",
        "deuce_firstReturn_total_lostmistake", "deuce_firstReturn_total_rally",
        "deuce_firstReturn_total_wonwinner", "deuce_firstreturns_freq", "adv_firstReturn_total_lostmistake",
        "adv_firstReturn_total_rally", "adv_firstReturn_total_wonwinner", "adv_firstreturns_freq",
        "secondReturn_total_lostmistake", "secondReturn_total_rally", "secondReturn_total_wonwinner",
        "Deuce_secondReturn_total_lostmistake", "Deuce_secondReturn_total_rally",
        "Deuce_secondReturn_total_wonwinner", "Deuce_secondreturns_freq", "Adv_secondReturn_total_lostmistake",
        "Adv_secondReturn_total_rally", "Adv_secondReturn_total_wonwinner", "Adv_secondreturns_freq",
        "firstreturn_rally", "rally_lostmistake", "rally_rally", "rally_wonwinner", "rally_freq", "up_1st_serve",
        "crit_1streturnMADE_freq", "noncrit_1streturnMADE_freq", "up_1st_return", "crit_secondserve_freq",
        "noncrit_2ndserve_freq", "crit_2ndservenonloss_freq", "noncrit_2ndservenonloss_freq", "up_2nd_serve",
        "crit_2ndreturnHIT_freq", "noncrit_2ndreturnHIT_freq", "criticalPercentage2ndReturns",
        "noncrit_2ndreturnMADE_freq", "up_2nd_return", "FirstReturn_FH_Freq", "FirstReturn_BH_Freq",
        "FirstReturn_Slice_Freq", "SecondReturn_FH_Freq", "SecondReturn_BH_Freq", "SecondReturn_Slice_Freq"
    ]

    local_vars = locals()
    stats = {f"{key}_{suffix}": local_vars[key] for key in stat_keys}
    return df, stats

