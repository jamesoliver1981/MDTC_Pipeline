import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from .Create_KPIs import limit_eff

def compare_data(allD):

    """this function generates comparisons of multiple matches 
    and generates those outputs needed to flow into multi.
    This can be run on multiple cuts of data, training vs comp, better vs worse to give alternate insights"""
    import os
    base = os.path.dirname(os.path.abspath(__file__))    
    # allD.to_csv(f"{base}/servedata_allD.csv", index = False)
    #create first serve rate consistency
    #number of games
    numGames = allD.shape[1] - 4
    serveRates =[]
    for i in range(numGames):
        g = allD[["Label_0", "Label", "variable", f"Game{i+1}"]]
        firstserves = g[(g.Label_0 =="FirstServes_by_ServeFrom_Freq") & (g.Label == "Serve_Total") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        totalserves = g[(g.Label_0 == "TotalServes_by_ServeFrom_Freq") & (g.Label == "Serve_Total") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        
        serveRates.append((f"Game{i+1}",firstserves / totalserves ))
    serveRates = pd.DataFrame(serveRates, columns = ["Game", "Rate"])
    overallFirstServe = (allD[(allD.Label_0 =="FirstServes_by_ServeFrom_Freq") & (allD.Label == "Serve_Total") & (allD.variable == "Frequency")][["Total"]].sum(axis = 1).sum() / 
                            allD[(allD.Label_0 == "TotalServes_by_ServeFrom_Freq") & (allD.Label == "Serve_Total") & (allD.variable == "Frequency")][["Total"]].sum(axis = 1).sum())
    
    serveRates["consistent"] = np.where( ((overallFirstServe - serveRates.Rate) < 0.1) & ((overallFirstServe - serveRates.Rate) > -0.1), 1, 0 )
    
    consistentServes = len(serveRates[serveRates.consistent == 1])
    if (consistentServes / numGames) > 0.65:
        firstserve_consistency = "High"
    elif (consistentServes / numGames) > 0.50:
        firstserve_consistency = "Medium"
    else :
        firstserve_consistency = "Low"

    # what are the levels of first serve effectiveness & are they consistent?

    firstServeEff_Games =[]
    for i in range(numGames):
        g = allD[["Label_0", "Label", "variable", f"Game{i+1}"]]
        firstserves_freq = g[(g.Label_0 =="FirstServes_by_ServeFrom_Freq") & (g.Label == "Serve_Total") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        firstServe_total_rally = g[(g.Label_0 =="Eval1_By_OutCome_Effective_First") & (g.Label == "Serve_Rally") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

        if (firstServe_total_rally / firstserves_freq) > 0.5:
            firstServe_RallyFactor = firstServe_total_rally - (firstserves_freq * 0.5)
        else: 
            firstServe_RallyFactor = 0

        firstserve_eff_freq = ((g[(g.Label_0 =="Effective_Score_First") & (g.Label == "Effective_Score_First_Serve") & (g.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum() + firstServe_RallyFactor * -10 ) / 
                        g[(g.Label_0 =="Effective_Score_First") & (g.Label == "Effective_Score_First_Serve") & (g.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum())
        firstserve_eff = limit_eff(firstserve_eff_freq, 3, 9.5)
        firstServeEff_Games.append((f"Game{i+1}", firstserve_eff))

    firstServeEff_Games = pd.DataFrame(firstServeEff_Games, columns = ["Game", "Rate"])
    overallFirstServe_Raw = (allD[(allD.Label_0 =="Effective_Score_First") & (allD.Label == "Effective_Score_First_Serve") & (allD.variable == "Effective_1")][["Total"]].sum(axis = 1).sum() / 
                            allD[(allD.Label_0 == "Effective_Score_First") & (allD.Label == "Effective_Score_First_Serve") & (allD.variable == "Frequency")][["Total"]].sum(axis = 1).sum())
    overallFirstServe_Effectiveness = limit_eff(overallFirstServe_Raw, 3, 8.5)
    firstServeEff_Games["consistent"] = np.where( ((overallFirstServe_Effectiveness - firstServeEff_Games.Rate) < 0.1) & ((overallFirstServe_Effectiveness - firstServeEff_Games.Rate) > -0.1), 1, 0 )
    
    consistent_1stEff = len(firstServeEff_Games[firstServeEff_Games.consistent == 1])
    if (consistent_1stEff / numGames) > 0.65:
        firstserve_eff_consistency = "High"
    elif (consistent_1stEff / numGames) > 0.50:
        firstserve_eff_consistency = "Medium"
    else :
        firstserve_eff_consistency = "Low"

    #building a calculation that gives the insights into serve performance
    #for loop like above, but rather than just one stat & eval, there are multi stats - outcomes & serve impact
    firstServe_Outcomes =[]
    for i in range(numGames):
        g = allD[["Label_0", "Label", "variable", f"Game{i+1}"]]
        #total points
        firstserves_freq = g[(g.Label_0 =="FirstServes_by_ServeFrom_Freq") & (g.Label == "Serve_Total") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        #won
        firstServe_won = g[(g.Label_0 =="Eval1_By_OutCome_Effective_First") & (g.Label == "Serve_Won_thru_Winner") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        #rally
        firstServe_rally = g[(g.Label_0 =="Eval1_By_OutCome_Effective_First") & (g.Label == "Serve_Rally") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        #lost
        firstServe_lost = g[(g.Label_0 =="Eval1_By_OutCome_Effective_First") & (g.Label == "Serve_Lost_thru_Mistake") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

        # how many points are won directly from first serve
        firstServe_wondirect = g[(g.Label_0 =="Start_SplitBy_Serve_Type_Part_OutCome_Shot") & (g.Label == "Serve_First_One_Won_thru_Winner_Serve") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        #how many points lost directly from first serve
        firstServe_lostdirect = g[(g.Label_0 =="Start_SplitBy_Serve_Type_Part_OutCome_Shot") & (g.Label == "Serve_First_One_Lost_thru_Mistake_Serve") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()

        # forehand used #
        firstServe_FHs = g[(g.Label_0 =="ServeReturn_ShotEff") & (g.Label == "FirstServe_FH") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        firstServe_FHsraw = g[(g.Label_0 =="ServeReturn_ShotEff") & (g.Label == "FirstServe_FH") & (g.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum()
        #forehand effectiveness
        firstserve_FHeff = limit_eff((firstServe_FHsraw / firstServe_FHs), 3, 8.5)
        firstServe_Outcomes.append((f"Game{i+1}", firstserves_freq, firstServe_won, firstServe_rally, firstServe_lost, firstServe_wondirect, firstServe_lostdirect, firstServe_FHs   ,firstserve_FHeff))
    firstServe_Outcomes = pd.DataFrame(firstServe_Outcomes, columns = ["Game", "TotalPoints", "Won", "Rally", "Lost", "WonWServe", "LostWServe", "FHsPlus1_Freq", "FHPlus1_Effective"])
    #total FH effectiveness
    firstServe_FHs_total = allD[(allD.Label_0 =="ServeReturn_ShotEff") & (allD.Label == "FirstServe_FH") & (allD.variable == "Frequency")][["Total"]].sum(axis = 1).sum()
    firstServe_FHsraw_total = allD[(allD.Label_0 =="ServeReturn_ShotEff") & (allD.Label == "FirstServe_FH") & (allD.variable == "Effective_1")][["Total"]].sum(axis = 1).sum()
    firstserve_FHeff_total = limit_eff((firstServe_FHsraw_total / firstServe_FHs_total), 3, 8.5)
    #have the df now - 
    firstServe_Outcomes["WonWServe_Percent"] = firstServe_Outcomes.WonWServe / firstServe_Outcomes.TotalPoints
    #calc serve +1 balance
    firstServe_Outcomes["serveplusWon"] = firstServe_Outcomes.Won - firstServe_Outcomes.WonWServe
    firstServe_Outcomes["serveplusLost"] = firstServe_Outcomes.Lost - firstServe_Outcomes.LostWServe
    # firstServe_Outcomes["serveplusWonLossBalance"] = firstServe_Outcomes.serveplusWon + firstServe_Outcomes.serveplusLost

    # calc serve + 1 points
    firstServe_Outcomes["ServePlusPoints"] = firstServe_Outcomes.serveplusWon + firstServe_Outcomes.serveplusLost + firstServe_Outcomes.Rally

    #calc % FH played on serve +1
    firstServe_Outcomes["FHPlayed_percent"] = firstServe_Outcomes.FHsPlus1_Freq / firstServe_Outcomes.ServePlusPoints

    #what metrics do I want here?
    #first up what do I want here at all - I want to know how strong is the serve & how strong is the follow up
        # in terms of feeding it into the insights & recs, there will be nuances but start with insights and when need more build more
        # start high level - % of points winning directly from serve  (% of total played)
    First_WonWServe = firstServe_Outcomes.WonWServe.sum() / firstServe_Outcomes.TotalPoints.sum()
            # and is this consistent
    firstServe_Outcomes["WinwServeConsistent"] = np.where(((firstServe_Outcomes.WonWServe_Percent - First_WonWServe )< 0.1) & ((firstServe_Outcomes.WonWServe_Percent - First_WonWServe )> -0.1),1,0 )
    if (firstServe_Outcomes.WinwServeConsistent.sum() / numGames) > 0.65:
        FirstWinwServeConsistent = "High"
    elif (firstServe_Outcomes.WinwServeConsistent.sum() / numGames) > 0.5:
        FirstWinwServeConsistent = "Medium"
    else:
        FirstWinwServeConsistent = "Low"
        # then this represents % points won
    First_ServeWinAllWin = firstServe_Outcomes.WonWServe.sum() / firstServe_Outcomes.Won.sum()

        # then on the remaining serve plus1, what happens
            # am I consistently able to play FH?
    # FirstFHPlayed_percent = firstServe_Outcomes.FHsPlus1_Freq.sum() / firstServe_Outcomes.ServePlusPoints.sum()

    firstServe_Outcomes["FirstFHPlayed_consistent"] = np.where(firstServe_Outcomes.FHPlayed_percent > 0.7,1,0 )
    if (firstServe_Outcomes["FirstFHPlayed_consistent"].sum() / numGames) >0.65:
        FirstFHPlayed_percent = "High"
    elif (firstServe_Outcomes["FirstFHPlayed_consistent"].sum() / numGames) >0.5:
        FirstFHPlayed_percent = "Medium"
    else:
        FirstFHPlayed_percent = "Low"
            #need to figure something out with regards to how interpret / deliver info on serve +1 outcomes... balance is nice but cannot use as percentage
            # won loss differential on serve plus 1 points
    serveplusWonpercent = firstServe_Outcomes["serveplusWon"].sum() / firstServe_Outcomes["ServePlusPoints"].sum()
    serveplusLostpercent = firstServe_Outcomes["serveplusLost"].sum() / firstServe_Outcomes["ServePlusPoints"].sum()
    serveplusWonLossBalance = serveplusWonpercent - serveplusLostpercent
                # is it something to do with the FH effectiveness score?
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    #adding total to serveRates
    # serveRates.loc[len(serveRates)] = ["Total", overallFirstServe, 1]
    
    #can create total and then just add that to the bottom
    totals = pd.DataFrame([{"Game": "Total", "FirstServeRate": overallFirstServe*100 , "1stEffective": overallFirstServe_Effectiveness, "FHPlus1_Effective": firstserve_FHeff_total}])
    
    firstServeEff_Games.rename(columns = {"Rate": "1stEffective"}, inplace = True)
    firstServeOut = pd.concat([firstServe_Outcomes,firstServeEff_Games[["1stEffective"]],serveRates["Rate"] ], axis =1)
    firstServeOut = pd.concat([firstServeOut, totals], axis = 0)
    # import os
    # base = os.path.dirname(os.path.abspath(__file__))    
    # firstServeOut.to_csv(f"{base}/servedata.csv", index = False)

    return firstserve_consistency, firstserve_eff_consistency, First_WonWServe, FirstWinwServeConsistent,First_ServeWinAllWin, FirstFHPlayed_percent, serveplusWonLossBalance
#firstserve_consistency (high, medium, low), firstserve_eff_consistency(high, medium, low), First_WonWServe (%), FirstWinwServeConsistent (high, medium, low),First_ServeWinAllWin (%), FirstFHPlayed_percent (high, medium, low), serveplusWonLossBalance (%)


def compare_returns(allD):
    #first returns calc
    
    #consistency of effectiveness
    numGames = allD.shape[1] - 4
    returns = []
    for i in range(numGames):
        g = allD[["Label_0", "Label", "variable", f"Game{i+1}"]]
        firsteffraw = g[(g.Label_0 =="Effective_Score_First") & (g.Label == "Effective_Score_First_Return") & (g.variable == "Effective_1")].iloc[:,3:].sum(axis = 1).sum()
        firstfreq = g[(g.Label_0 =="Effective_Score_First") & (g.Label == "Effective_Score_First_Return") & (g.variable == "Frequency")].iloc[:,3:].sum(axis =1).sum()          
        firstreturn_eff = limit_eff(firsteffraw/firstfreq, 4, 9)

        # how often is 40% reached - % of numGames
            #points lost
        firstReturn_lost = g[(g.Label_0 =="Eval1_By_OutCome_Effective_First") & (g.Label.isin(["Return_Lost_thru_Winner", "Return_Lost_thru_Forced", "Return_Lost_thru_Mistake"])) & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        firstReturnMade = g[(g.Label_0 =="FirstServesReturnMade_by_ServeFrom_Freq") & (g.Label == "Return_Total") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 1).sum()
        #how many second serve points
        secondreturnsFREQ = g[(g.Label_0 =="SecondServesReturnHit_by_ServeFrom_Freq") & (g.Label == "Return_Total") & (g.variable == "Frequency")].iloc[:,3:].sum(axis = 0).sum()

        returns.append((f"Game{i+1}", firsteffraw, firstreturn_eff, firstfreq, firstReturn_lost, firstReturnMade, secondreturnsFREQ))
    returns = pd.DataFrame(returns, columns = ["Game", "firsteffraw", "firstreturn_eff", "firstfreq", "firstReturn_lost", "firstReturnMade", "secondreturnsFREQ"])    
    
    #calc total effectiveness & then add in consistency and calc
    total_firsteff = limit_eff((returns.firsteffraw.sum()/ returns.firstfreq.sum()), 4,9)
    returns["FirstEff_Consistent"] = np.where(((total_firsteff - returns.firstreturn_eff)< 0.1) & ((total_firsteff - returns.firstreturn_eff)> -0.1), 1, 0)

    if (returns.FirstEff_Consistent.sum() / numGames) > 0.65:
        First_eff_cons = "High"
    elif (returns.FirstEff_Consistent.sum() / numGames) > 0.5:
        First_eff_cons = "Medium"
    else:
        First_eff_cons = "Low"

    returns["FirstCompletionRate"] = returns.firstReturnMade / returns.firstfreq

    #average is 75% - more than 10% off of that - highlight
    returns["LessThanAverage"] = np.where(returns.FirstCompletionRate < 0.67, 1, 0)
    if (returns.LessThanAverage.sum() / numGames) >= 0.6:
        FirstCompletionRate = "Low" # here say work on return
    elif (returns.LessThanAverage.sum() / numGames) >= 0.4:
        FirstCompletionRate = "Medium" # here say work on both
    else:
        FirstCompletionRate = "High" # here say work on 2nd shot quality

    #calc lost percentage
    returns["LostPer"] = returns.firstReturn_lost / returns.firstfreq
    # how often is 40% reached - % of numGames
    returns["LostPer40"] = np.where(returns.LostPer >=0.4, 1, 0)
    LostOver40Rate = returns.LostPer40.sum() / numGames

    #when above 40%, what is completion rate vs below: group by
    over40 = returns[returns.LostPer40 == 1]
    numGamesover40 = over40.shape[1]
    #need to add in consistency of above completion rate when above 40%
    #average is 75% - more than 10% off of that - highlight
    # over40["LessThanAverage"] = np.where(over40.FirstCompletionRate < 0.67, 1, 0)
    if (over40.LessThanAverage.sum() / numGamesover40) >= 0.6:
        FirstCompletionRate_Over40 = "Low" # here say work on return
    elif (over40.LessThanAverage.sum() / numGamesover40) >= 0.4:
        FirstCompletionRate_Over40 = "Medium" # here say work on both
    else:
        FirstCompletionRate_Over40 = "High" # here say work on 2nd shot quality

    #2nd return analytics
    #median number of points
    median2ndReturnPts = returns.secondreturnsFREQ.median()


    #effective return scores are overalls at present
    #total lost % include in other
    return First_eff_cons, LostOver40Rate, FirstCompletionRate, numGamesover40, FirstCompletionRate_Over40, median2ndReturnPts

def gen_out(df, alldata, type, **stats):
    suffix = f"_{type}"
    unpacked_keys = []

    for key, value in stats.items():
        if key.endswith(suffix):
            clean_key = key[:-len(suffix)]
            globals()[clean_key] = value
            unpacked_keys.append(clean_key)

    # --- IDE/Linter Hints: Predeclare variables for runtime unpacking ---
    # use for debugging - otherwise commented out
    
    # serve_won = serve_lost = return_won = return_lost = won = lost = None
    # rally_324_won = rally_324_lost = rally_5plus_won = rally_5plus_lost = None
    # totalserve_eff = return_eff_freq = rally_eff_freq = firstserverate = None
    # firstserve_eff = secondserve_eff = df_rate = secondserves_freq = None
    # crit_totalserve_freq = crit_1stserve_freq = noncrit_1stserve_freq = None
    # noncrit_totalserve_freq = breakball_allserves = breakball_1stserverate = None
    # deuce_allserves_freq = adv_allserves_freq = firstserves_freq = None
    # deuce_firstserves_rate = adv_firstserves_rate = None
    # firstserve_serveonly_freq = firstserve_serveFH_freq = firstserve_serveBH_freq = None
    # serveplusplayed = FH_of_ServePlus1 = firstserve_serveFH_eff = BH_of_ServePlus1 = None
    # firstserve_serveBH_eff = Slice_of_ServePlus1 = firstserve_serveSlice_eff = None
    # secondserve_serveonly_freq = secondserve_serveFH_freq = secondserve_serveBH_freq = None
    # FH_of_ServePlus1_2nd = secondserve_serveFH_eff = BH_of_ServePlus1_2nd = None
    # secondserve_serveBH_eff = Slice_of_ServePlus1_2nd = secondserve_serveSlice_eff = None
    # firstreturn_rate = secondreturn_rate = secondreturns_freq = firstreturns_freq = None
    # firstreturn_eff = secondreturn_eff = Total_Returns = FH_Return_prop = BH_Return_prop = None
    # Slice_Return_prop = FirstReturn_FH_prop = FirstReturn_BH_prop = FirstReturn_Slice_prop = None
    # FirstReturn_FH_Eff = FirstReturn_BH_Eff = FirstReturn_Slice_Eff = None
    # SecondReturn_FH_prop = SecondReturn_BH_prop = SecondReturn_Slice_prop = None
    # SecondReturn_FH_Eff = SecondReturn_BH_Eff = SecondReturn_Slice_Eff = None
    # rally_points = rally_eff = rally_win_balance = rally_win_balance2 = rally_length = All_shots = None
    # FH_prop = BH_prop = Slice_prop = Volley_prop = OH_prop = None
    # FH_eff = FH_shots = FH_WinBalance2 = FH_WinBalance = None
    # BH_eff = BH_shots = BH_WinBalance2 = BH_RallyRate = BH_WinBalance = None
    # Slice_eff = Slice_shots = Slice_LossRate = Slice_WinBalance2 = None
    # firstserve_lost = firstserve_rally = firstserve_won = firstservesOnlyProp = dfs = None
    # secondServe_total_lostwinner = secondServe_total_lostforced = secondServe_total_lostmistake = None
    # secondserve_lost = crit_nonbreakball_allserves = crit_nonbreakball_1stserverate = None
    # firstreturn_lost = firstreturn_won = secondreturn_lost = secondreturn_won = secondreturn_rally = None
    # crit_1streturnHIT_freq = noncrit_1streturnHIT_freq = rally_prop = FH_ratio = BH_Slice_ratio = None
    # ptsplayed = Won_thru_Winner = Lost_thru_Mistake = None
    # firstServe_total_lostmistake = firstServe_total_rally = firstServe_total_wonwinner = None
    # deuce_firstServe_total_lostmistake = deuce_firstServe_total_rally = deuce_firstServe_total_wonwinner = None
    # deuce_firstserves_freq = adv_firstServe_total_lostmistake = adv_firstServe_total_rally = None
    # adv_firstServe_total_wonwinner = adv_firstserves_freq = secondServe_total_rally = secondServe_total_wonwinner = None
    # Deuce_secondServe_total_lostmistake = Deuce_secondServe_total_rally = Deuce_secondServe_total_wonwinner = None
    # Deuce_secondserves_freq = Adv_secondServe_total_lostmistake = Adv_secondServe_total_rally = None
    # Adv_secondServe_total_wonwinner = Adv_secondserves_freq = secondserve_rally = secondserve_won = None
    # totalreturn_eff = firstReturn_total_lostmistake = firstReturn_total_rally = firstReturn_total_wonwinner = None
    # deuce_firstReturn_total_lostmistake = deuce_firstReturn_total_rally = deuce_firstReturn_total_wonwinner = None
    # deuce_firstreturns_freq = adv_firstReturn_total_lostmistake = adv_firstReturn_total_rally = None
    # adv_firstReturn_total_wonwinner = adv_firstreturns_freq = secondReturn_total_lostmistake = None
    # secondReturn_total_rally = secondReturn_total_wonwinner = None
    # Deuce_secondReturn_total_lostmistake = Deuce_secondReturn_total_rally = Deuce_secondReturn_total_wonwinner = None
    # Deuce_secondreturns_freq = Adv_secondReturn_total_lostmistake = Adv_secondReturn_total_rally = None
    # Adv_secondReturn_total_wonwinner = Adv_secondreturns_freq = firstreturn_rally = None
    # rally_lostmistake = rally_rally = rally_wonwinner = rally_freq = None
    # up_1st_serve = crit_1streturnMADE_freq = noncrit_1streturnMADE_freq = up_1st_return = None
    # crit_secondserve_freq = noncrit_2ndserve_freq = crit_2ndservenonloss_freq = noncrit_2ndservenonloss_freq = None
    # up_2nd_serve = crit_2ndreturnHIT_freq = noncrit_2ndreturnHIT_freq = criticalPercentage2ndReturns = None
    # noncrit_2ndreturnMADE_freq = up_2nd_return = None
    # FirstReturn_FH_Freq = FirstReturn_BH_Freq = FirstReturn_Slice_Freq = None
    # SecondReturn_FH_Freq = SecondReturn_BH_Freq = SecondReturn_Slice_Freq = None
    # -------------------------------------------------------------------
    #### INSIGHTS & RECOMMENDATIONS CODE
    nl = "\n"
    if type == "multi":
        allfirstserve_consistency,firstserve_eff_consistency, First_WonWServe, FirstWinwServeConsistent,First_ServeWinAllWin, FirstFHPlayed_percent, serveplusWonLossBalance = compare_data(alldata)
        
        First_eff_cons, LostOver40Rate, FirstCompletionRate, numGamesover40, FirstCompletionRate_Over40, median2ndReturnPts = compare_returns(alldata)
    #First_eff_cons (High medium low), LostOver40Rate (%), FirstCompletionRate (kinda like consistency low, medium, high), numGamesover40 (count), FirstCompletionRate_Over40 (kinda like consistency low, medium, high)

    #firstserve_consistency (high, medium, low), firstserve_eff_consistency(high, medium, low), First_WonWServe (%), FirstWinwServeConsistent (high, medium, low),First_ServeWinAllWin (%), FirstFHPlayed_percent (high, medium, low), serveplusWonLossBalance (%)


    allnumGames = alldata.shape[1] - 4

    ### Averages - Define standards for comparison
    firstserverate_average = 0.66
    df_rate_average = 0.18
    firstserve_eff_average = (1.8 + 2)/ (6.5/100)
    secondserve_eff_average = (-1.6 + 3)/ (6/100)
    firstreturn_rate_average = 0.75
    secondreturn_rate_average = 0.64
    firstreturn_eff_average = (-0.7 + 4) / (6/100)
    secondreturn_eff_average = (-0.1 + 4) / (8/100)
    TotalReturn_error_average = 248 / 2103

    rally_eff_average = ((1208/1225)+1.5)/(5.5/100)
    rally_win_balance_average = (569 - 547) / 1116
    rally_risk_balance_average = (202 + 117 - 314) / 1116
    rally_det_balance_average = (202 + 117 - 157 - 76) / 1116
    rally_mistake_balance_average = (250 - 314) / 1116
    rally_length_average = 1225/1116
    ###Home Screen INSIGHTS ####
    # Proportion of points under 4 shots long: { enter value} of which won { enter value}
    under4 = serve_won + serve_lost + return_won + return_lost
    under4_won_per = (return_won + serve_won) / under4
    under4_per = under4 / (won + lost)

    rally_324_per = (rally_324_won + rally_324_lost) / (won + lost)
    rally_5plus_per = (rally_5plus_won + rally_5plus_lost) / (won+lost)

    home_insights1 = f"Proportion of points 4 shots or less:{under4_per: .0%} of which won{under4_won_per: .0%}"
    # [if proportion of points 5-8 or 9+ less than proportion under 4 shots, “ Serve & Return parts of the game are the most critical areas”]
    if rally_324_per > under4_per:
        home_insights2 = ""
    elif rally_5plus_per > under4_per:
        home_insights2 = ""
    else:
        home_insights2 = "4 shots compromising the Serve & Return are most critical area of game as represent largest section of game"
    # Serve Effectiveness is [ high / medium / low ] - based on {enter value}
    if totalserve_eff < 38:
        serve_eff_level = "Low" # on average not winning as much as losing - or reliant on mistakes and having winners hit against you
        serve_eff_level_exp = "Score suggests on average losing more than winning, or you are reliant on mistakes and having winners hit against you"
    elif totalserve_eff < 51:
        serve_eff_level = "Medium - but on the lower end" # between parity 0 & 1 - ie getting to rally
        serve_eff_level_exp = "Score suggests the outcomes of your serves are fairly equal.  Ideally you want to be winning more"
    elif totalserve_eff < 63:
        serve_eff_level = "Medium - but on the higher end" # getting to rally on average
        serve_eff_level_exp = "Score suggests you are winning your serve points on average.  There is however room for improvement"
    else:
        serve_eff_level = "High"
        serve_eff_level_exp = "You are winning a good majority of the points on your serve - its hurting your opponent"

    home_insights3 = f"Serve Effectiveness is {serve_eff_level}"
    home_insights3a = f"{serve_eff_level_exp} - see the Serve Section for more"
    # Return Effectiveness is [ high / medium / low ] - based on {enter value}

    if totalreturn_eff < 33: # on average losing points at a high rate
        return_eff_level = "Low"
        return_eff_level_exp = "On average, losing return points quickly with server dominant"
    elif totalreturn_eff < 50:
        return_eff_level = "Medium - but on the lower end" # on average losing points but not extremely
        return_eff_level_exp = "On average, losing but server is not as dominant and getting towards parity"
    elif totalreturn_eff < 66:
        return_eff_level = "Medium - but on the higher end" # on average balancing out points and getting towards rally
        return_eff_level_exp = "On average, achieving parity in points and averaging getting towards rally"
    else: 
        return_eff_level = "High" # on average getting to rally and winning some
        return_eff_level_exp = "On average, getting point to rally and able to get a winning percentage in the return game"

    home_insights4 = f"Return Effectiveness is {return_eff_level}"
    home_insights4a = f"{return_eff_level_exp} - see the Return Section for more"

    # Rally Effectiveness is [ high / medium / low ] - based on {enter value}
    if rally_eff < 33: # means losing and not even getting to rally
        rally_eff_level = "Low"
        rally_eff_level_exp = "When getting to rally, tending to lose and quickly"
    elif rally_eff < 45: # means beginning to rally but still losing too many
        rally_eff_level = "Medium - but on the lower end"
        rally_eff_level_exp = "Have a losing tendancy here once getting beyond serve & returns"
    elif rally_eff < 66: # means rallying and starting to win majority
        rally_eff_level = "Medium - but on the higher end"
        rally_eff_level_exp = "Able to build the point to a rally and have a positive outcome of rallies"
    else: 
        rally_eff_level = "High" # win when it gets to rally
        rally_eff_level_exp = "Outcomes are favourable in rally - winning majority of points once getting to rally"

    home_insights5 = f"Rally Effectiveness is {rally_eff_level}"
    home_insights5a = f"{rally_eff_level_exp} - see the Rally Section for more"
    # Average Rally Length overall is {enter value} of which you win {enter value}

    # When serving, the Average Rally Length overall is {enter value} of which you win {enter value}

    # When returning, the Average Rally Length overall is {enter value} of which you win {enter value}


    ### Serve Insights Code ###

    #first serve rate of  - which is what vs average
    firstserverate_diff = firstserverate - firstserverate_average
    if firstserverate_diff >= 0:
        if ((type == "single") | (allnumGames == 1)):
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is{firstserverate_diff * 100: .1f}pts better than average"
        elif allfirstserve_consistency == "High":
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is{firstserverate_diff * 100: .1f}pts better than average, and is consistently so."
        elif allfirstserve_consistency == "Medium":
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is on generally {firstserverate_diff * 100: .1f}pts better than average.  However, it is not always around this level."
        else:
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is on generally {firstserverate_diff * 100: .1f}pts better than average.  However, the consistency is low.  This inconsistency impacts results."
    else:
        if ((type == "single") | (allnumGames == 1)):
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is{firstserverate_diff * -100: .1f}pts worse than average"
        elif allfirstserve_consistency == "High":
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is{firstserverate_diff * 100: .1f}pts worse than average, and is consistently so."
        elif allfirstserve_consistency == "Medium":
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is on generally {firstserverate_diff * 100: .1f}pts worse than average.  However, the level varies.  Work on this as it will make a difference to results."
        else:
            firstserverate_insight = f"At{firstserverate: .1%}, 1st Serve Rate is on generally {firstserverate_diff * 100: .1f}pts worse than average.  However, the consistency is low.  Keep working on this as it will impact results."
    #with an effectiveness of  - which is what vs average
    firstserve_eff_diff = firstserve_eff - firstserve_eff_average
    if firstserve_eff_diff >= 0:
        if type == "multi":
            if firstserve_eff_consistency == "High":
                firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff : .1f}pts better than average, and its highly consistent."
            elif firstserve_eff_consistency == "Medium":
                firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff : .1f}pts better than average overall, though its not always at this level."
            else: 
                firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff : .1f}pts better than average overall, but there are signficant differences in performances."    
        else:
            firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff : .1f}pts better than average"
    else:
        if type == "multi":
            if firstserve_eff_consistency == "High":
                firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff : .1f}pts worse than average, and consistently so."
            elif firstserve_eff_consistency == "Medium":
                firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff : .1f}pts worse than average overall, though its not always at this level."
            else: 
                firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff : .1f}pts worse than average overall, but there are signficant differences in performances."    
        else:
            firstserve_eff_insight = f"1st Serve Effectiveness is{firstserve_eff_diff *-1: .1f}pts worse than average"
        
    #second serve effectiveness of - which is what vs average
    secondserve_eff_diff = secondserve_eff - secondserve_eff_average
    if secondserve_eff_diff >= 0:
        secondserve_eff_insight = f"2nd Serve Effectiveness is{secondserve_eff_diff : .1f}pts better than average"
    else:
        secondserve_eff_insight = f"2nd Serve Effectiveness is{secondserve_eff_diff *-1: .1f}pts worse than average"

    if secondserve_lost > secondserve_won :
        secondserve_eff_insight2 = f"2nd Serve Effectiveness suffers because you lose more points than you win behind your second serve."
    else:
        secondserve_eff_insight2 = "You win more than you lose behind your 2nd Serve.  This is good basis."
    #df rate - could add a if less than 30 second serves
    df_rate_diff = df_rate - df_rate_average
    # if secondserves_freq < 30:
    if type == "single":
        dfrate_insight = ""
    else:
        if df_rate_diff >= 0:
            dfrate_insight = f"At {df_rate:.1%}, the points lost directly from the serve is{df_rate_diff*100: .1f}pts higher than average - based on{secondserves_freq: .0f} second serves"
        else:
            dfrate_insight = f"At {df_rate:.1%}, the points lost directly from the serve is{df_rate_diff*100: .1f}pts lower than average - based on{secondserves_freq: .0f} second serves"
    #difference in effectiveness is driven by outcomes
    #on 1st serve win x%, x% goes to rally, & lose x% - # this is the same info as is in the chart - where is the value add?
    #on 2nd serve win x%, x% goes to rally, & lose x% # should these instead be ratios of win to loss? but ignores rally which is more the goal on 2nd

    #how winning on 1st - only if more than 30 - # include note saying need to play more to get insight

    #on critical points, first serve is x vs non critical
    # if crit_totalserve_freq < 30:
    if crit_totalserve_freq < 30:
        critical_1stserve_insights = "Insufficient data to generate insights on Critical 1st Serve Rate.  Play more to get these insights"
    else:
        critical_1stserve_diff = (crit_1stserve_freq / crit_totalserve_freq) - (noncrit_1stserve_freq / noncrit_totalserve_freq)
        if critical_1stserve_diff > 0:
            critical_1stserve_insights = f"On critical points, 1st serve is{critical_1stserve_diff *100: .1f}pts better than on non-critical points, based on{crit_totalserve_freq: .0f} critcal points"
        else:
            critical_1stserve_insights = f"On critical points, 1st serve is{critical_1stserve_diff *-100: .1f}pts worse than on non-critical points, based on{crit_totalserve_freq: .0f} critcal points"
                                        
        #can go deeper and look at break balls - if have enough data
    if breakball_allserves < 30:
        break_1strate_insights = "Insufficient data to generate insights on Breakpoints 1st Serve Rate.  Play more to get these insights"
    else:
        break_1strate_diff = breakball_1stserverate - (noncrit_1stserve_freq / noncrit_totalserve_freq)
        if break_1strate_diff > 0:
            break_1strate_insights = f"On break points, 1st serve is{break_1strate_diff *100: .1f}pts better than on non-critical points, based on{breakball_allserves: .0f} break points"
        else:
            break_1strate_insights = f"On break points, 1st serve is{break_1strate_diff *-100: .1f}pts worse than on non-critical points, based on{breakball_allserves: .0f} break points"

    #add com
        
    #first serve rate on deuce & on adv - #update  combined this together
    if deuce_allserves_freq + adv_allserves_freq < 55:
        deuce_1stserve_insight = "Insufficient data in this match to split 1st Serve Rate to into Deuce & Advantage.  An overview is available in the Personal Summary."
    else:
        deuce_1stserve_insight = f"First Serve Rate on Deuce side is{deuce_firstserves_rate: .1%}, vs {adv_firstserves_rate: .1%} on the Ad side.  This is based on{deuce_allserves_freq + adv_allserves_freq: .0f} First Serve points"
        
    # if adv_allserves_freq < 30:
    #     adv_1stserve_insight = "Insufficient data to generate insights on Advantage 1st Serve Rate.  Play more to get these insights"
    # else:
    #     adv_1stserve_insight = f"First Serve Rate on Advantage side is{adv_firstserves_rate: .1%}, based on{adv_allserves_freq: .0f} Advantage Serve points"

    #add deuce effectiveness on 1st serve




    #add in shots post first - & effectiveness too
        #where number of 1st serve points less than 30, don't show prop
        #firstserve_consistency (high, medium, low), firstserve_eff_consistency(high, medium, low), First_WonWServe (%), FirstWinwServeConsistent (high, medium, low),First_ServeWinAllWin (%), FirstFHPlayed_percent (high, medium, low), serveplusWonLossBalance (%)

    if type == "multi":
        shot_from_firstserve_warn = ""
        if FirstWinwServeConsistent == "High":
            shot_from_firstserve_insight = f"You win {First_WonWServe:.0%} points directly with your 1st Serve, and it is consistently around this level.  This represents {First_ServeWinAllWin:.0%} of your 1st Serve won points."
        elif FirstWinwServeConsistent == "Medium":
            shot_from_firstserve_insight = f"You win {First_WonWServe:.0%} points directly with your 1st Serve, though the consistency could be better. This average represents {First_ServeWinAllWin:.0%} of your 1st Serve won points."
        else:
            shot_from_firstserve_insight = f"On average, you win {First_WonWServe:.0%} points directly with your 1st Serve.  But sometimes it is signficantly worse or better than this. This average represents {First_ServeWinAllWin:.0%} of your 1st Serve won points."
        
        if serveplusWonLossBalance > 0:
            if FirstFHPlayed_percent == "High":
                shot_eff_from_firstserve_insight = f"On the serve+1, you win {serveplusWonLossBalance:.0%} points than you lose. You consistently play a high percentage of Forehands on your +1 shot. This suggests effective use of the Forehand as the sword."
            elif FirstFHPlayed_percent == "Medium":
                shot_eff_from_firstserve_insight = f"On the serve+1, you win {serveplusWonLossBalance:.0%} points than you lose. You generally play a high percentage of Forehands on your +1 shot, but its not always consistently so."
            else:
                shot_eff_from_firstserve_insight = f"On the serve+1, you win {serveplusWonLossBalance:.0%} points than you lose. You don't play a consistently high percentage of forehands on your +1 shot."
        else:
            if FirstFHPlayed_percent == "High":
                shot_eff_from_firstserve_insight = f"On the serve+1, you lose {serveplusWonLossBalance:.0%} points than you win. You do consistently play a high percentage of Forehands on your +1 shot, suggesting you need to work on forehand effectiveness."
            elif FirstFHPlayed_percent == "Medium":
                shot_eff_from_firstserve_insight = f"On the serve+1, you lose {serveplusWonLossBalance:.0%} points than you win. You generally play a high percentage of Forehands on your +1 shot, but its not always consistently so."
            else:
                shot_eff_from_firstserve_insight = f"On the serve+1, you lose {serveplusWonLossBalance:.0%} points than you win. You don't play a consistently high percentage of forehands on your +1 shot. This suggests your serve needs work so you can bring in your forehand more."
        
    elif firstserves_freq < 30:
        shot_from_firstserve_warn = f"Note: only {firstserves_freq:.0f} First Serve points played.  Insights could be volatile where proportions are low."
        shot_from_firstserve_insight = "No shot distributions are shown to avoid overinterpretation."
        shot_eff_from_firstserve_insight = "Details over multiple matches are available in the Personal Summary."
        shot_eff_from_firstserve_insight2 = f"Backhand was used {BH_of_ServePlus1:.0%} of the time, with an effectiveness score of {firstserve_serveBH_eff:.0f}, and Slice was used {Slice_of_ServePlus1:.0%} of the time, with an effectiveness score of {firstserve_serveSlice_eff:.0f}."
    else:
        shot_from_firstserve_warn = ""
        shot_from_firstserve_insight = f"Of 1st Serve points,{(firstserve_serveonly_freq/firstserves_freq): .0%} were Serve only,{firstserve_serveFH_freq/firstserves_freq: .0%} were a Serve FH combination, and {firstserve_serveBH_freq/firstserves_freq:.0%} were a Serve BH combo"
        shot_eff_from_firstserve_insight = f"Of {serveplusplayed:.0f} First Serve points where a Serve+1 shot was played, Forehand was used {FH_of_ServePlus1:.0%} of the time, with an effectiveness score of {firstserve_serveFH_eff:.0f}."
        shot_eff_from_firstserve_insight2 = f"Backhand was used {BH_of_ServePlus1:.0%} of the time, with an effectiveness score of {firstserve_serveBH_eff:.0f}, and Slice was used {Slice_of_ServePlus1:.0%} of the time, with an effectiveness score of {firstserve_serveSlice_eff:.0f}."
        

        #where specific shot is less than 30, don't show effectiveness - should be on same scale as 1st serves
    #add in shots post second - & effectiveness too
    if secondserves_freq < 30:
        shot_from_secondserve_insight = "Insufficient data to generate insights on a shot breakdown from 2nd Serve.  A view on this performance is available in your Personal Summary."
    else:
        shot_from_secondserve_insight = f"Of 2nd Serve points,{(secondserve_serveonly_freq/secondserves_freq): .0%} were Serve only,{secondserve_serveFH_freq/secondserves_freq: .0%} were a Serve FH combination, and {secondserve_serveBH_freq/firstserves_freq:.0%} were a Serve BH combo"
    if secondserve_serveonly_freq < 30:
        shot_eff_from_secondserve_insight = "Insufficient data to generate insights on effectiveness of shots from 2nd Serve.  Play more to get these insights"
        shot_eff_from_secondserve_insight2 = ""
    else:
        shot_eff_from_secondserve_insight = f"Of {serveplusplayed:.0f} Second Serve points where a Serve+1 shot was played, Forehand was used {FH_of_ServePlus1_2nd:.0%} of the time, with an effectiveness score of {secondserve_serveFH_eff:.0f}."
        shot_eff_from_secondserve_insight2 = f"Backhand was used {BH_of_ServePlus1_2nd:.0%} of the time, with an effectiveness score of {secondserve_serveBH_eff:.0f}, and Slice was used {Slice_of_ServePlus1_2nd:.0%} of the time, with an effectiveness score of {secondserve_serveSlice_eff:.0f}."



    ### RETURN INSIGHTS ####

    # Average return rate: 75 %
    firstreturn_rate_diff = firstreturn_rate - firstreturn_rate_average*100
    if firstreturn_rate_diff >=0 :
        firstreturn_rate_ins = f"1st Return Completion Rate of {firstreturn_rate: .0f}% is {firstreturn_rate_diff*1:.0f}pts better than average."
    else:
        firstreturn_rate_ins = f"1st Return Completion Rate of {firstreturn_rate: .0f}% is {firstreturn_rate_diff*-1:.0f}pts worse than average."
        
    

    # Played against a 1st serve rate of : vs average			
    firstserverate_faced = firstreturns_freq / (firstreturns_freq + secondreturns_freq)
    firstserverate_faced_diff = firstserverate_faced - firstserverate_average
    if firstserverate_faced_diff >= 0:
        firstserverate_faced_ins = f"You faced a 1st Serve Rate of{firstserverate_faced: .0%}, which is{firstserverate_faced_diff*100: .0f}pts higher than average."
    else:
        firstserverate_faced_ins = f"You faced a 1st Serve Rate of{firstserverate_faced: .0%}, which is{firstserverate_faced_diff*-100: .0f}pts worse than average."
    
        # Effectiveness of 1st - vs average			
    firstreturn_eff_diff = firstreturn_eff - firstreturn_eff_average
    if type == "multi":
        if firstreturn_eff_diff >=0 :
            firstreturn_eff_ins = f"Across {allnumGames:.0f} matches, you faced {firstreturns_freq:.0f} 1st Serves, and had an effectiveness of {firstreturn_eff:.0f}, which is{firstreturn_eff_diff*1: .0f}pts better than average."
        else:
            firstreturn_eff_ins = f"Across {allnumGames:.0f} matches, you faced {firstreturns_freq:.0f} 1st Serves, and had an effectiveness of {firstreturn_eff:.0f}, which is{firstreturn_eff_diff*-1: .0f}pts worse than average."
    else:
        if firstreturn_eff_diff >=0 :
            firstreturn_eff_ins = f"At{firstreturn_eff: .0f}, 1st Return Effectiveness is{firstreturn_eff_diff*1: .0f}pts better than average."
        else:
            firstreturn_eff_ins = f"At{firstreturn_eff: .0f}, 1st Return Effectiveness is{firstreturn_eff_diff*-1: .0f}pts worse than average."
    


    # if Total_Returns < 40:
    #     #TotalReturn_error_ins = "Insufficient data."
    #     ReturnShot_ins = "Insufficient data."
    #     ReturnShot_ins2 = ""
    # else:
    #     #TotalReturn_error_ins = f"Of the{Total_Returns: .0f} Return points played, {TotalReturn_ErrorRate: .0%} were lost to mistakes in the Return shot.  Average is{TotalReturn_error_average: .0%} "
    #     ReturnShot_ins = f"Of the{Total_Returns: .0f} Return points played, Forehands were{FH_Return_prop: .0%}, Backhands were{BH_Return_prop: .0%}, and Slice were{Slice_Return_prop: .0%}"
        # ReturnShot_ins2 = "This is useful to know what Return Shot to practise."
        
        #can add in FH and BH and Slice error mistakes - 

    #These are possible additions but leave now - they are very mistake focused
        
    # Where is pt lost - of lost, 1st or second			

    # 	split by deuce / ad? - #caution dependent		
       
    #First_eff_cons (High medium low), LostOver40Rate (%), FirstCompletionRate (kinda like consistency low, medium, high), numGamesover40 (count), FirstCompletionRate_Over40 (kinda like consistency low, medium, high)         

    if type == "multi":
        FirstReturn_ShotEff_ins = f"{FirstReturn_FH_prop:.0%} of 1stReturns were Forehands, {FirstReturn_BH_prop:.0%} were Backhands, & {FirstReturn_Slice_prop:.0%} were Slice."
        FirstReturn_ShotEff_ins2 = f"Forehand had an Effectiveness of {FirstReturn_FH_Eff:.0f}, Backhand:{FirstReturn_BH_Eff: .0f}, and Slice:{FirstReturn_Slice_Eff: .0f}"
    elif firstreturns_freq < 35:
        FirstReturn_ShotEff_ins = "There are low number of 1st return points. Consider the below, but for adjustments, look to the information provided in your Personal Summary which looks at performance over multiple games."
        FirstReturn_ShotEff_ins2 = f"Of the {firstreturns_freq:.0f} First Return points played, {FirstReturn_FH_prop:.0%} were Forehands, {FirstReturn_BH_prop:.0%} were Backhands, & {FirstReturn_Slice_prop:.0%} were Slice."
        # FirstReturn_ShotEff_ins3 = ""
    else:
        FirstReturn_ShotEff_ins = f"Of the {firstreturns_freq:.0f} First Return points played, {FirstReturn_FH_prop:.0%} were Forehands, {FirstReturn_BH_prop:.0%} were Backhands, & {FirstReturn_Slice_prop:.0%} were Slice."
        if (FirstReturn_FH_Freq >= 15) & ( FirstReturn_BH_Freq>= 15) & (FirstReturn_Slice_Freq >= 15):
            FirstReturn_ShotEff_ins2 = f"On First Return, Forehand had an Effectiveness Score of {FirstReturn_FH_Eff:.0f}, Backhand:{FirstReturn_BH_Eff: .0f}, and Slice:{FirstReturn_Slice_Eff: .0f}"
        elif (FirstReturn_FH_Freq >= 15) & ( FirstReturn_BH_Freq>= 15):
            FirstReturn_ShotEff_ins2 = f"On First Return, Forehand had an Effectiveness Score of {FirstReturn_FH_Eff:.0f}, and Backhand:{FirstReturn_BH_Eff: .0f}.  There are insufficient slice returns to calculate an effectiveness score."
        elif (FirstReturn_FH_Freq >= 15) & ( FirstReturn_Slice_Freq>= 15):
            FirstReturn_ShotEff_ins2 = f"On First Return, Forehand had an Effectiveness Score of {FirstReturn_FH_Eff:.0f}, and Backhand:{FirstReturn_Slice_Eff: .0f}.  There are insufficient backhand returns to calculate an effectiveness score."
        else:
            FirstReturn_ShotEff_ins2 = "There are insufficient variations in the return shots to generate effectiveness scores."
        # if firstreturns_freq < 60:
        #     FirstReturn_ShotEff_ins3 = "Note that Effective scores can be volatile where occurances are low."
        # else:
        #     FirstReturn_ShotEff_ins3 = ""
        # Repeat for 2nd return

    #2nd return completion rate vs average
    secondreturn_rate_diff = secondreturn_rate - secondreturn_rate_average*100
    # Effectiveness of 2nd - vs average			
    secondreturn_eff_diff = secondreturn_eff - secondreturn_eff_average
    # if secondreturns_freq < 30:
    #     secondreturn_rate_ins = "Insufficient data."
    # else:
    #     if secondreturn_rate_diff >=0 :
    #         secondreturn_rate_ins = f"2nd Return Completion Rate of {secondreturn_rate: .0f}% is {secondreturn_rate_diff*1:.0f}pts better than average."
    #     else:
    #         secondreturn_rate_ins = f"2nd Return Completion Rate of {secondreturn_rate: .0f}% is {secondreturn_rate_diff*-1:.0f}pts worse than average."
    
    # if secondreturn_eff_diff >=0 :
    #     secondreturn_eff_ins = f"At{secondreturn_eff: .0f}, 2nd Return Effectiveness is{secondreturn_eff_diff*1: .0f}pts better than average."
    # else:
    #     secondreturn_eff_ins = f"At{secondreturn_eff: .0f}, 2nd Return Effectiveness is{secondreturn_eff_diff*-1: .0f}pts worse than average."
    # if secondreturns_freq < 35:
    #     SecondReturn_ShotEff_ins = "There are low number of 2nd return points. Consider the below, but for adjustments, look to the information provided in your Personal Summary which looks at performance over multiple games."
    #     SecondReturn_ShotEff_ins2 = f"Of the {secondreturns_freq:.0f} Second Return points played, {SecondReturn_FH_prop:.0%} were Forehands, {SecondReturn_BH_prop:.0%} were Backhands, & {SecondReturn_Slice_prop:.0%} were Slice."
    #     SecondReturn_ShotEff_ins3 = ""
    # else:
    #     SecondReturn_ShotEff_ins = f"Of the {secondreturns_freq:.0f} Second Return points played, {SecondReturn_FH_prop:.0%} were Forehands, {SecondReturn_BH_prop:.0%} were Backhands, & {SecondReturn_Slice_prop:.0%} were Slice."
    #     if (SecondReturn_FH_Freq >= 15) & ( SecondReturn_BH_Freq>= 15) & (SecondReturn_Slice_Freq >= 15):
    #         SecondReturn_ShotEff_ins2 = f"On 2nd Return, Forehand had an Effectiveness Score of {SecondReturn_FH_Eff:.0f}, Backhand:{SecondReturn_BH_Eff: .0f}, and Slice:{SecondReturn_Slice_Eff: .0f}"
    #     elif (SecondReturn_FH_Freq >= 15) & ( SecondReturn_BH_Freq>= 15):
    #         SecondReturn_ShotEff_ins2 = f"On 2nd Return, Forehand had an Effectiveness Score of {SecondReturn_FH_Eff:.0f}, and Backhand:{SecondReturn_BH_Eff: .0f}.  There are insufficient slice returns to calculate an effectiveness score."
    #     elif (SecondReturn_FH_Freq >= 15) & ( SecondReturn_Slice_Freq>= 15):
    #         SecondReturn_ShotEff_ins2 = f"On 2nd Return, Forehand had an Effectiveness Score of {SecondReturn_FH_Eff:.0f}, and Backhand:{SecondReturn_Slice_Eff: .0f}.  There are insufficient backhand returns to calculate an effectiveness score."
    #     else:
    #         SecondReturn_ShotEff_ins2 = "There are insufficient variations in the 2nd return shots to generate effectiveness scores."
    if secondreturns_freq < 30:
        if secondreturn_eff_diff >=0 :
            secondreturn_eff_ins = f"At{secondreturn_eff: .0f}, 2nd Return Effectiveness is{secondreturn_eff_diff*1: .0f}pts better than average."
        else:
            secondreturn_eff_ins = f"At{secondreturn_eff: .0f}, 2nd Return Effectiveness is{secondreturn_eff_diff*-1: .0f}pts worse than average."
        secondreturn_rate_ins = "Insufficient data."
        SecondReturn_ShotEff_ins = f"There are only {secondreturns_freq:.0f} 2nd Serve Return points in total.  Insights are limited as the data is volatile."
        SecondReturn_ShotEff_ins2 = f""
    else:
        if secondreturn_rate_diff >=0 :
            secondreturn_rate_ins = f"2nd Return Completion Rate of {secondreturn_rate: .0f}% is {secondreturn_rate_diff*1:.0f}pts better than average."
        else:
            secondreturn_rate_ins = f"2nd Return Completion Rate of {secondreturn_rate: .0f}% is {secondreturn_rate_diff*-1:.0f}pts worse than average."
    
        if secondreturn_eff_diff >=0 :
            secondreturn_eff_ins = f"At{secondreturn_eff: .0f}, 2nd Return Effectiveness is{secondreturn_eff_diff*1: .0f}pts better than average."
        else:
            secondreturn_eff_ins = f"At{secondreturn_eff: .0f}, 2nd Return Effectiveness is{secondreturn_eff_diff*-1: .0f}pts worse than average."
        
        SecondReturn_ShotEff_ins = f"Of the {secondreturns_freq:.0f} Second Return points played, {SecondReturn_FH_prop:.0%} were Forehands, {SecondReturn_BH_prop:.0%} were Backhands, & {SecondReturn_Slice_prop:.0%} were Slice."
        if (SecondReturn_FH_Freq >= 15) & ( SecondReturn_BH_Freq>= 15) & (SecondReturn_Slice_Freq >= 15):
            SecondReturn_ShotEff_ins2 = f"On 2nd Return, Forehand had an Effectiveness Score of {SecondReturn_FH_Eff:.0f}, Backhand:{SecondReturn_BH_Eff: .0f}, and Slice:{SecondReturn_Slice_Eff: .0f}"
        elif (SecondReturn_FH_Freq >= 15) & ( SecondReturn_BH_Freq>= 15):
            SecondReturn_ShotEff_ins2 = f"On 2nd Return, Forehand had an Effectiveness Score of {SecondReturn_FH_Eff:.0f}, and Backhand:{SecondReturn_BH_Eff: .0f}.  There are insufficient slice returns to calculate an effectiveness score."
        elif (SecondReturn_FH_Freq >= 15) & ( SecondReturn_Slice_Freq>= 15):
            SecondReturn_ShotEff_ins2 = f"On 2nd Return, Forehand had an Effectiveness Score of {SecondReturn_FH_Eff:.0f}, and Backhand:{SecondReturn_Slice_Eff: .0f}.  There are insufficient backhand returns to calculate an effectiveness score."
        else:
            SecondReturn_ShotEff_ins2 = "There are insufficient variations in the 2nd return shots to generate effectiveness scores."


    #### Rally Insights####
    #rally caution if under 30
    if rally_points <30:
        rally_ins_caution = f"There are only {rally_points:.0f} rally points.  Rally insights could be volatile."
    else:
        rally_ins_caution = ""
    #define terms # 
    #effective rally - vs average
    rally_eff_diff = rally_eff - rally_eff_average
    if rally_eff_diff > 0:
        rally_eff_ins = f"Your Rally Effectiveness Score is {rally_eff:.0f}.  This is {rally_eff_diff:.0f}pts better than average."
    else :
        rally_eff_ins = f"Your Rally Effectiveness Score is {rally_eff:.0f}.  This is {rally_eff_diff *-1:.0f}pts worse than average."

    rally_eff_ins2 = "Effectiveness is calculated on outcomes per shot. If the shot continues to rally, the score is given an extra +1."
    #win - balance
    rally_winbalance_diff = rally_win_balance - rally_win_balance_average
    if rally_winbalance_diff > 0 :
        rally_winbalance_ins = f"Your Win Balance is {rally_win_balance2}.  This is {rally_winbalance_diff*100:.0f}pts better than average."
    else:
        rally_winbalance_ins = f"Your Win Balance is {rally_win_balance2}.  This is {rally_winbalance_diff*-100:.0f}pts worse than average."
    rally_winbalance_ins2 = "Win Balance compares the points played in rally and subtracts the lost points from the won points and divides by Rally points played."

    # #risk balance
    # rally_riskbalance_diff = rally_risk_balance  - rally_risk_balance_average
    # if rally_riskbalance_diff > 0:
    #     rally_riskbalance_ins = f"Your Risk Balance is {rally_risk_balance2}.  This is {rally_riskbalance_diff*100:.0f}pts better than average."
    # else:
    #     rally_riskbalance_ins = f"Your Risk Balance is {rally_risk_balance2}.  This is {rally_riskbalance_diff*-100:.0f}pts worse than average."
    # rally_riskbalance_ins2 = "Risk Balance are the points you determine (win via winner or a forced error) less the mistakes you make, and divided by Rally points played."

    # # determination balance
    # rally_det_balance_diff = rally_determine - rally_det_balance_average
    # if rally_det_balance_diff>0:
    #     rally_det_ins = f"Your Determined Balance is {rally_determine2}.  This is {rally_det_balance_diff*100:.0f}pts better than average."
    # else:
    #     rally_det_ins = f"Your Determined Balance is {rally_determine2}.  This is {rally_det_balance_diff*-100:.0f}pts worse than average."
    # rally_det_ins2 = "Determined Balance is the sum of the points you won via winners and forced errors, less the points the opponent won with winners and forced errors.  This is then divided by the Rally points played."

    # # unforced error balance
    # rally_mistake_balance_diff = rally_mistakes - rally_mistake_balance_average
    # if rally_mistake_balance_diff > 0:
    #     rally_mistake_ins = f"Your Unforced Error Balance is {rally_mistakes2}.  This is {rally_mistake_balance_diff*100:.0f}pts better than average."
    # else:
    #     rally_mistake_ins = f"Your Unforced Error Balance is {rally_mistakes2}.  This is {rally_mistake_balance_diff*-100:.0f}pts worse than average."
    # rally_mistake_ins2 = "Unforced Error Balance is the difference between points you won due to unforced errors, and the points you lost due to unforced errors.  This is divided by number of Rally points played."

    #length of rallies on  serve & return - vs averages - these values don't make sense ignore
    # serve_rally_len = (df[(df.Label_0 == "AverageRally_Length") & (df.Label == "CountandSum_Serve") & (df.variable == "Effective_1")].sum(axis= 1).sum()/
    #                    df[(df.Label_0 == "AverageRally_Length") & (df.Label == "CountandSum_Serve") & (df.variable == "Frequency")].sum(axis= 1).sum())

    # return_rally_len = (df[(df.Label_0 == "AverageRally_Length") & (df.Label == "CountandSum_Return") & (df.variable == "Effective_1")].sum(axis= 1).sum()/
    #                    df[(df.Label_0 == "AverageRally_Length") & (df.Label == "CountandSum_Return") & (df.variable == "Frequency")].sum(axis= 1).sum())

    #when get to rally
    rally_length_diff  = rally_length - rally_length_average
    if rally_length_diff > 0:
        rally_length_ins = f"When the point progresses to rally, the point is on average {rally_length*2 +4:.1f} shots long.  This is {rally_length_diff*2:.1f} shots longer than average. This includes serve & return."
    else:
        rally_length_ins = f"When the point progresses to rally, the point is on average {rally_length*2 +4:.1f} shots long.  This is {rally_length_diff*2:.1f} shots shorter than average. This includes serve & return."

    rally_length_totalav = df[(df.Label_0 == "AverageRally_Length") & (df.variable == "Effective_1")].sum(axis=1).sum() / df[(df.Label_0 == "AverageRally_Length") & (df.variable == "Frequency")].sum(axis=1).sum()
    rally_length_totalav_ins = f"Average Rally length is {rally_length_totalav*2:.1f} shots long."


    #shot distribution - excludes return shot
    shot_advise_ins = "Note shot distributions and shot effectiveness are for ALL shots, NOT just the rallies.  The only shots excluded are the Serve & Return.  The +1 shots are included in the below."
    rally_shot_dist_ins = f"Of the {All_shots:.0f} shots played, {FH_prop:.0%} were Forehands, {BH_prop:.0%} were Backhands, {Slice_prop:.0%} were Slice, {Volley_prop:.0%} were Volleys, and {OH_prop:.0%} were Overheads."
    #effectiveness
    eff_caution_ins = "Effectiveness for each shot is defined differently as each shot typically has different aims.  Also be careful about over-interpretating the data if the number of shots is small."
    # based on x shots - this is the effectiveness and what it means
    if FH_eff < 38:
        rally_FHeff_ins = f"Based on {FH_shots:.0f} shots, the Forehand Effectiveness Score is {FH_eff:.0f}, which is Low.  This is because you lose more than you win when you play your Forehand. Win Balance is {FH_WinBalance2}pts."
    elif FH_eff < 46:
        if FH_WinBalance < 0.05:
            rally_FHeff_ins = f"Based on {FH_shots:.0f} shots, the Forehand Effectiveness Score is {FH_eff:.0f}, which is Medium but on the Low end.  This is because you are only winning marginally more points than you lose on your Forehand."
        else:
            rally_FHeff_ins = f"Based on {FH_shots:.0f} shots, the Forehand Effectiveness Score is {FH_eff:.0f}, which is Medium but on the Low end.  Whilst you win a higher proportion than you lose, the lost points represent too large a share of points."     
    elif FH_eff < 64:
        rally_FHeff_ins = f"Based on {FH_shots:.0f} shots, the Forehand Effectiveness Score is {FH_eff:.0f}, which is Medium and on the High end.  You have a clear Win Balance of {FH_WinBalance2}pts."     
    else:
        rally_FHeff_ins = f"Based on {FH_shots:.0f} shots, the Forehand Effectiveness Score is {FH_eff:.0f}, which is High.  You have a clear Win Balance of {FH_WinBalance2}pts."     

    if BH_eff < 38:
        rally_BHeff_ins = f"Based on {BH_shots:.0f} shots, the Backhand Effectiveness Score is {BH_eff:.0f}, which is Low.  This is because you loss rate is high and the Win Balance is significantly negative at {BH_WinBalance2}pts."
    elif BH_eff < 56:
        rally_BHeff_ins = f"Based on {BH_shots:.0f} shots, the Backhand Effectiveness Score is {BH_eff:.0f}, which is Medium but on the Low end.  The Win Balance is moderately negative at {BH_WinBalance2}pts and Rally Rate is {BH_RallyRate:.0%}."     
    elif BH_eff < 70:
        if BH_WinBalance > 0:
            rally_BHeff_ins = f"Based on {BH_shots:.0f} shots, the Backhand Effectiveness Score is {BH_eff:.0f}, which is Medium and on the High end.  You have a positive Win Balance of {BH_WinBalance2}pts."     
        else:
            rally_BHeff_ins = f"Based on {BH_shots:.0f} shots, the Backhand Effectiveness Score is {BH_eff:.0f}, which is Medium and on the High end.  The Win Balance is marginally negative at {BH_WinBalance2}pts and Rally Rate is high at {BH_RallyRate:.0%}.."     
    else:
        rally_BHeff_ins = f"Based on {BH_shots:.0f} shots, the Backhand Effectiveness Score is {BH_eff:.0f}, which is High.  You have a clear Win Balance of {BH_WinBalance2}pts."     

    if Slice_eff < 34:
        rally_Sliceeff_ins = f"Based on {Slice_shots:.0f} shots, the Slice Effectiveness Score is {Slice_eff:.0f}, which is Low.  This is because on average when you play a slice you tend to lose the point."
    elif Slice_eff < 50:
        rally_Sliceeff_ins = f"Based on {Slice_shots:.0f} shots, the Slice Effectiveness Score is {Slice_eff:.0f}, which is Medium but on the Low end.  You are getting to point parity but your loss rate is too high at {Slice_LossRate:.0%}."     
    elif Slice_eff < 67:
        rally_Sliceeff_ins = f"Based on {Slice_shots:.0f} shots, the Slice Effectiveness Score is {Slice_eff:.0f}, which is Medium and on the High end.  Slice keeps you in the rally and your Win Balance is {Slice_WinBalance2}pts."     
    else:
        rally_Sliceeff_ins = f"Based on {Slice_shots:.0f} shots, the Slice Effectiveness Score is {Slice_eff:.0f}, which is High.  Your Slice is a wepoan and have a clear Win Balance of {Slice_WinBalance2}pts."     


        # only include if have 30 in each shot

    # #HR rates & error & win rates - cannot create error rates so needs recoding
    # HR_def_ins = "Maximum Heart Rate is specific based on your age & gender. Hard is 80 - 90% of Maximum HR, and Maximum is above 90%."
    # HR_Hard_ins = f"You spent {Hard_Time_prop:.0%} of your time in the Hard HR Zone.  Your Win Rate here was {Hard_WinRate:.0%} & Error Rate was {Hard_MistakeRate:.0%}"
    # if Max_Time == 0:
    #     HR_Max_Ins = "You spent 0 time in the Max heart rate zone. Either the game was not challenging at all or you are exceptionally fit."
    # else:
    #     HR_Max_Ins = f"You spent {Max_Time_prop:.0%} of your time in the Hard HR Zone.  Your Win Rate here was {Max_WinRate:.0%} & Error Rate was {Max_MistakeRate:.0%}"


    rally_insights_data = [
            ["Rally Caution Insight", rally_ins_caution, 1 ],
        ["Rally Effectiveness", rally_eff_ins, 2 ],
        ["Rally Effectiveness def",rally_eff_ins2,3 ],
        ["Rally Win Balance",rally_winbalance_ins,4],
        ["Rally Win Balance def", rally_winbalance_ins2,5],  
        #["Rally Risk Balance", rally_riskbalance_ins, 6],
        #["Rally Risk Balance def", rally_riskbalance_ins2, 7],
        #["Rally Determined Balance", rally_det_ins ,8],
        #["Rally Determined Balance def", rally_det_ins2, 9],
        #["Rally Unforced Error Balance",rally_mistake_ins, 10],
        #["Rally Unforced Error Balance def", rally_mistake_ins2, 11],
    #     ["Rally Length", rally_length_ins, 12],
        
        ["Shot Props", rally_shot_dist_ins, 13],
        ["Shot Props def", shot_advise_ins, 14],
        ["Effective Shots Caution", eff_caution_ins, 15],
        ["Effective FH", rally_FHeff_ins, 16],
        ["Effective BH", rally_BHeff_ins, 17],
        ["Effective Slice", rally_Sliceeff_ins, 18],
        
    #     ["HR Hard Zone", HR_Hard_ins, 19],
    #     ["HR Max Zone", HR_Max_Ins, 20],
    #     ["HR def", HR_def_ins, 21],
    #     ["Second Return Effectiveness By Shot", SecondReturn_ShotEff_ins2, 13],
    #     ["Second Return Effectiveness By Shot",SecondReturn_ShotEff_ins3, 14 ],
    ]
    rally_insights = pd.DataFrame(rally_insights_data, columns=["Label", "Insight", "InherentRanking"])

    rally_insights_show = rally_insights[~rally_insights.Insight.str.contains("Insufficient")].sort_values("InherentRanking")
    rally_insights_show = rally_insights[rally_insights.Insight != ""]


    rally_insights_show_ls = list(rally_insights_show.Insight)
    rally_label_show_ls = list(rally_insights_show.Label)
    rally_ins_out = ""
    nl = "\n"
    for i in range(len(rally_insights_show_ls)):
        if i == len(rally_insights_show_ls)-1:
            rally_ins_out = rally_ins_out + rally_insights_show_ls[i] + nl + nl
        elif "def" in rally_label_show_ls[i+1]:
            rally_ins_out = rally_ins_out + rally_insights_show_ls[i] + nl
        else:
            rally_ins_out = rally_ins_out + rally_insights_show_ls[i] + nl + nl


    return_insights_data = [
        ["First Return Rate",firstreturn_rate_ins,3 ],
        ["Second Return Rate",secondreturn_rate_ins,4],
        ["First Serve Faced", firstserverate_faced_ins,5],
        ["First Serve Effectiveness", firstreturn_eff_ins, 1 ],
        ["Second Serve Effectiveness", secondreturn_eff_ins, 2 ],
        #["Total Return Error Rates", TotalReturn_error_ins ,8],
        # ["Total Return Shots Proportion", ReturnShot_ins, 6],
        # ["Total Return Shots Proportion", ReturnShot_ins2, 7],
        ["First Return Shots Proportion", FirstReturn_ShotEff_ins, 9],
        ["First Return Effectiveness By Shot",FirstReturn_ShotEff_ins2, 10],
        # ["First Return Effectiveness By Shot", FirstReturn_ShotEff_ins3, 11],
        ["Second Return Shots Proportion", SecondReturn_ShotEff_ins, 12],
        ["Second Return Effectiveness By Shot", SecondReturn_ShotEff_ins2, 13],
        # ["Second Return Effectiveness By Shot",SecondReturn_ShotEff_ins3, 14 ],
    ]
    return_insights = pd.DataFrame(return_insights_data, columns=["Label", "Insight", "InherentRanking"])


    return_insights_show = return_insights[~return_insights.Insight.str.contains("Insufficient")].sort_values("InherentRanking")
    return_insights_show = return_insights_show[return_insights_show.Insight != ""]

    return_insights_missing = return_insights[return_insights.Insight.str.contains("Insufficient")].sort_values("InherentRanking")
    return_insights_missing

    return_insights_show_ls = list(return_insights_show.Insight)


    return_ins_out = ""
    nl = "\n"
    for i in range(len(return_insights_show_ls)):
        return_ins_out = return_ins_out + return_insights_show_ls[i] + nl + nl



    return_insights_missing_ls = list(return_insights_missing.Label)
    if len(return_insights_missing_ls) == 0:
        return_insights_missing_out = ""
    else:
        return_insights_missing_out = "Play more to get insights on these areas: " + nl
        for i in range(len(return_insights_missing_ls)):
            return_insights_missing_out = return_insights_missing_out + return_insights_missing_ls[i] + nl
        


    return_insights_out_fin = return_ins_out + return_insights_missing_out



    # take firsts, create where have, where missing goes to play more
    return_insights_firsts = return_insights[return_insights.InherentRanking.isin([1,3,5,9,10,11])].sort_values("InherentRanking")
    return_insights_firsts = return_insights_firsts[return_insights_firsts.Insight != ""]
    return_insights_firsts_data = list(return_insights_firsts[~return_insights_firsts.Insight.str.contains("Insufficient")].Insight)
    # return_insights_firsts_data = return_insights_firsts_data[return_insights_firsts_data.notnull()].Insight
    return_insights_firsts_out = "First Return"
    for i in range(len(return_insights_firsts_data)):
        return_insights_firsts_out = return_insights_firsts_out + nl + return_insights_firsts_data[i] 
        
    return_insights_seconds = return_insights[return_insights.InherentRanking.isin([4,2,12,13,14])].sort_values("InherentRanking")
    return_insights_seconds = return_insights_seconds[return_insights_seconds.Insight != ""]
    return_insights_seconds_data = list(return_insights_seconds[~return_insights_seconds.Insight.str.contains("Insufficient")].Insight)
    return_insights_seconds_out = "Second Return"
    for i in range(len(return_insights_seconds_data)):
        return_insights_seconds_out = return_insights_seconds_out + nl + return_insights_seconds_data[i] 
        
    return_insights_gen = return_insights[return_insights.InherentRanking.isin([8,6,7])].sort_values("InherentRanking")
    return_insights_gen = return_insights_gen[return_insights_gen.Insight != ""]
    return_insights_gen_data = list(return_insights_gen[~return_insights_gen.Insight.str.contains("Insufficient")].Insight)
    # return_insights_firsts_data = return_insights_firsts_data[return_insights_firsts_data.notnull()].Insight
    return_insights_gen_out = "Generic Return Insights"
    if len(return_insights_gen_data) == 0:
        return_insights_gen_out = return_insights_gen_out + nl + "Insufficient data to provide generic insights."
    for i in range(len(return_insights_gen_data)):
        return_insights_gen_out = return_insights_gen_out + nl + return_insights_gen_data[i] 
    return_insights_out_fin = return_insights_firsts_out + nl + nl + return_insights_seconds_out + nl + nl + return_insights_gen_out + nl +nl+ return_insights_missing_out

    serve_insights_data = [
        ["First Serve Rate",firstserverate_insight , 2 ],
        
        ["First Serve Effectiveness", firstserve_eff_insight, 1],
            ["Warning Shots Played behind 1st Serve",shot_from_firstserve_warn, 3 ],
        ["Proportion Shots Played behind 1st Serve",shot_from_firstserve_insight, 4 ],
        ["Effectiveness of Shots Played Behind 1st", shot_eff_from_firstserve_insight, 5],
        ["Second Serve Effectiveness", secondserve_eff_insight, 6 ],
        ["Second Serve Effectiveness Parity", secondserve_eff_insight2,7 ],
        ["Double Fault Rate", dfrate_insight ,8],
        
        ["Critical Points 1st Serve Rate", critical_1stserve_insights, 9],
        ["Break points 1st Serve Rate", break_1strate_insights, 10],
        
        ["Deuce 1st Serve Rate", deuce_1stserve_insight, 11],
        # ["Ad 1st Serve Rate", adv_1stserve_insight,12],
        

        ["Proportion Shots Played behind 2nd Serve", shot_from_secondserve_insight, 13],
        ["Effectiveness of Shots Played behind 2nd",shot_eff_from_secondserve_insight, 14],
        ["Effectiveness of Shots Played behind 2nd",shot_eff_from_secondserve_insight2, 15]
    ]
    serve_insights = pd.DataFrame(serve_insights_data, columns=["Label", "Insight", "InherentRanking"])



    serve_insights_show = serve_insights[~serve_insights.Insight.str.contains("Insufficient")].sort_values("InherentRanking")
    serve_insights_show_ls = list(serve_insights_show.Insight)
    serve_ins_out = ""
    for i in range(len(serve_insights_show_ls)):
        serve_ins_out = serve_ins_out + serve_insights_show_ls[i] + nl + nl

    # serve_insights_missing = serve_insights[serve_insights.Insight.str.contains("Insufficient")].sort_values("InherentRanking")
    # serve_insights_missing_ls = list(serve_insights_missing.Label)
    # serve_ins_missing_out = "Play more to get insights on these areas: " + nl
    # for i in range(len(serve_insights_missing_ls)):
    #     serve_ins_missing_out = serve_ins_missing_out + serve_insights_missing_ls[i] + nl
        
    # serve_ins_out_fin = serve_ins_out + serve_ins_missing_out
    serve_ins_out_fin = serve_ins_out #+ serve_ins_missing_out


    # take firsts, create where have, where missing goes to play more
    serve_insights_firsts = serve_insights[serve_insights.InherentRanking.isin([1,2,3,4,5,10,11])].sort_values("InherentRanking")
    serve_insights_firsts_data = list(serve_insights_firsts[~serve_insights_firsts.Insight.str.contains("Insufficient")].Insight)
    serve_insights_firsts_out = "First Serve"
    for i in range(len(serve_insights_firsts_data)):
        serve_insights_firsts_out = serve_insights_firsts_out + nl + serve_insights_firsts_data[i] 
        
    serve_insights_seconds = serve_insights[serve_insights.InherentRanking.isin([6,7,8,13,14])].sort_values("InherentRanking")
    serve_insights_seconds_data = list(serve_insights_seconds[~serve_insights_seconds.Insight.str.contains("Insufficient")].Insight)
    serve_insights_seconds_out = "Second Serve"
    for i in range(len(serve_insights_seconds_data)):
        serve_insights_seconds_out = serve_insights_seconds_out + nl + serve_insights_seconds_data[i] 
        
    serve_insights_critical = serve_insights[serve_insights.InherentRanking.isin([9,10])].sort_values("InherentRanking")
    serve_insights_critical_data = list(serve_insights_critical[~serve_insights_critical.Insight.str.contains("Insufficient")].Insight)
    serve_insights_critical_out = "Critical Points"
    if len(serve_insights_critical_data) == 0:
        serve_insights_critical_out = serve_insights_critical_out + nl + "Not enough data for insights on critical points."
    else:
        for i in range(len(serve_insights_critical_data)):
            serve_insights_critical_out = serve_insights_critical_out + nl + serve_insights_critical_data[i] 
    serve_ins_out_fin = serve_insights_firsts_out + nl + nl + serve_insights_seconds_out + nl + nl + serve_insights_critical_out #+ nl + nl + serve_ins_missing_out


    ### Serve RECOMMENDATIONS Code ###
    # lots of ifs and projected impact
    # create as df - then sort by impact - largest top & then remove non impactful
        # list of recommendations & code below creates the nl string
    # already know that comparing win & loss brings most value - how build that in
        # can do generic
        # add in a recommendation from win loss

    ## start creating the data & the projected impact
    #first serve rate - compare to 2nd win rate - impact is if hit average rate
        #calc two numbers - how many win as is - how many win if serve rate were higher
        #as is is just first serve_won + second serve won + first serve rally won + second serve rally won

        #projected is total serves * first aver rate * win rate + remaining second serves * won rate + first serve new * rally win + second serve new * rally win
    totalserves = df[(df.Label_0 == "Shots_By_OutComeGen") & (df.Label.str.contains("Serve_")) & (df.variable == "Frequency")].sum(axis = 1).sum()
    firstserve_winrate = (df[(df.Label_0 == "Shots_By_OutComeGen_First") & (df.Label.str.contains("Serve_Won")) & (df.variable == "Frequency")].sum(axis = 1).sum()/
                        df[(df.Label_0 == "Shots_By_OutComeGen_First") & (df.Label.str.contains("Serve_")) & (df.variable == "Frequency")].sum(axis = 1).sum())
    secondserve_winrate = (df[(df.Label_0 == "Shots_By_OutComeGen_Second") & (df.Label.str.contains("Serve_Won")) & (df.variable == "Frequency")].sum(axis = 1).sum()/
                        df[(df.Label_0 == "Shots_By_OutComeGen_Second") & (df.Label.str.contains("Serve_")) & (df.variable == "Frequency")].sum(axis = 1).sum())

    firstserve_rallyrate = (df[(df.Label_0 == "Shots_By_OutComeGen_First") & (df.Label.str.contains("Serve_Rally")) & (df.variable == "Frequency")].sum(axis = 1).sum()/
                        df[(df.Label_0 == "Shots_By_OutComeGen_First") & (df.Label.str.contains("Serve_")) & (df.variable == "Frequency")].sum(axis = 1).sum())
    secondserve_rallyrate = (df[(df.Label_0 == "Shots_By_OutComeGen_Second") & (df.Label.str.contains("Serve_Rally")) & (df.variable == "Frequency")].sum(axis = 1).sum()/
                        df[(df.Label_0 == "Shots_By_OutComeGen_Second") & (df.Label.str.contains("Serve_")) & (df.variable == "Frequency")].sum(axis = 1).sum())


    firstserverally_winrate = (df[(df.Label_0 == "RallyOutcomes_FromHowStart") & (df.Label.str.contains("First_Serve_Won")) & (df.variable == "Frequency")].sum(axis = 1).sum()/
                        df[(df.Label_0 == "RallyOutcomes_FromHowStart") & (df.Label.isin(["First_Serve_Won", "First_Serve_Lost"])) & (df.variable == "Frequency")].sum(axis = 1).sum())

    secondserverally_winrate = (df[(df.Label_0 == "RallyOutcomes_FromHowStart") & (df.Label.str.contains("Second_Serve_Won")) & (df.variable == "Frequency")].sum(axis = 1).sum()/
                        df[(df.Label_0 == "RallyOutcomes_FromHowStart") & (df.Label.isin(["Second_Serve_Won", "Second_Serve_Lost"])) & (df.variable == "Frequency")].sum(axis = 1).sum())

    won_serve_actual = ((totalserves * firstserverate * firstserve_winrate) + (totalserves * firstserverate * firstserve_rallyrate * firstserverally_winrate ) +
                    (totalserves * (1-firstserverate) * secondserve_winrate) + (totalserves * (1-firstserverate) * secondserve_rallyrate * secondserverally_winrate ) )
    winrate_1stserve = ((totalserves * firstserverate * firstserve_winrate) + (totalserves * firstserverate * firstserve_rallyrate * firstserverally_winrate )) / firstserves_freq

    winrate_2ndserve = ((totalserves * (1-firstserverate) * secondserve_winrate) + (totalserves * (1-firstserverate) * secondserve_rallyrate * secondserverally_winrate ) )/secondserves_freq

    if firstserverate <= firstserverate_average:
    #     won_serve_projected = ((totalserves * firstserverate_average * firstserve_winrate) + (totalserves * firstserverate_average * firstserve_rallyrate * firstserverally_winrate ) +
    #                    (totalserves * (1-firstserverate_average) * secondserve_winrate) + (totalserves * (1-firstserverate_average) * secondserve_rallyrate * secondserverally_winrate ) )
    #    points_with_better1stserverate = (won_serve_projected / won_serve_actual)-1
        if (type == "single") | ((allnumGames == 1)):
            won_serve_projected_comment = f"1st Serve rate is less than average."
            firstserve_eff_home5 = "1st Serve Rate is lower than average. Look to increase this for better outcomes.  See the Serve section for more details."
        elif allfirstserve_consistency == "High":
            won_serve_projected_comment = "1st Serve rate is consistently less than average."
            firstserve_eff_home5 = "1st Serve Rate is consistently lower than average. Look to increase this for better outcomes.  See the Serve section for more details."
        else:
            won_serve_projected_comment = "1st Serve rate is less than average over multiple matches, but its not always so low.  Keep working to increase this rate."
            firstserve_eff_home5 = "1st Serve Rate is overall lower than average. Look to replicate good matches and increase this rate for better outcomes.  See the Serve section for more details."
        if winrate_1stserve - winrate_2ndserve > 0.5:
            won_serve_projected_comment2 = f"You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve. Look to increase this rate for better outcomes."
        else:
            won_serve_projected_comment2 = "It is typically expected that you win more points behind your 1st serve. Look to increase this rate for better outcomes."
        won_serve_projected_rating = "WorkOn"
        
    elif (firstserverate > 0.75) & (firstserve_eff < 60):
        if (type == "single") | ((allnumGames == 1)):
            won_serve_projected_comment = "1st Serve rate is strong, but effectiveness is not."
            won_serve_projected_comment2 = "Consider sacrificing serve rate to force more points directly."
            won_serve_projected_rating = "OK"
            firstserve_eff_home5 = ""
        elif allfirstserve_consistency == "High":
            won_serve_projected_comment = "1st Serve rate is consistently high, but effectiveness is not."
            won_serve_projected_comment2 = "You can win more points by being more aggressive on 1st serve to win points directly."
            won_serve_projected_rating = "OK"
            firstserve_eff_home5 = ""
        else:
            won_serve_projected_comment = "1st Serve rate is high on average but inconsistent, but you don't win a enough points from it."
            won_serve_projected_comment2 = "Work on the quality and consistency of your 1st serve to create pressure and win more points."
            won_serve_projected_rating = "OK"
            firstserve_eff_home5 = ""
    elif (firstserve_eff < 60):
        if (type == "single") | ((allnumGames == 1)):
            won_serve_projected_comment = "1st Serve rate is above average, but effectiveness is not."
            won_serve_projected_comment2 = "  Consider sacrificing serve rate to force more points directly."
            won_serve_projected_rating = "OK"
            firstserve_eff_home5 = ""
        elif allfirstserve_consistency == "High":
            won_serve_projected_comment = "1st Serve rate is consistently above average, but effectiveness is not."
            won_serve_projected_comment2 = "You have a strong basis.  You can try being more aggressive on 1st serve to win points directly."
            won_serve_projected_rating = "OK"
            firstserve_eff_home5 = ""
        else:
            won_serve_projected_comment = "1st Serve rate is above average across matches but inconsistent, and you don't win a enough points from it."
            won_serve_projected_comment2 = "Work on the quality and consistency of your 1st serve to create pressure and win more points."
            won_serve_projected_rating = "OK"
            firstserve_eff_home5 = ""
    else: # can I put in a if above 80% - & effectiveness not amazing, maybe try to be more aggressive with it - lower rate with more effectiveness is more pts
        won_serve_projected = ((totalserves * firstserverate *1.1 * firstserve_winrate) + (totalserves * firstserverate *1.1 * firstserve_rallyrate * firstserverally_winrate ) +
                    (totalserves * (1-firstserverate *1.1) * secondserve_winrate) + (totalserves * (1-firstserverate *1.1) * secondserve_rallyrate * secondserverally_winrate ) )
        points_with_better1stserverate = (won_serve_projected / won_serve_actual)-1
    #     won_serve_projected_comment = "Improving 1st Serve rate by 10% results in"
        won_serve_projected_comment = f"1st Serve Rate is above average and is effective."
        won_serve_projected_comment2 = "Keep it up."
        won_serve_projected_rating = "OK"
        firstserve_eff_home5 = "Keep it up."




    # 1st serve effectiveness recommendations
    if type == "multi":
        #firstserve_eff_consistency(high, medium, low), First_WonWServe (%), FirstWinwServeConsistent (high, medium, low),First_ServeWinAllWin (%), FirstFHPlayed_percent (high, medium, low), serveplusWonLossBalance (%)
        if First_WonWServe > 0.2:
            firstserve_eff_rec = f"You are winning {First_WonWServe:.0%} of points directly from your 1st Serve.  This is a strength and should be maintained."
            if serveplusWonLossBalance > 0.1:
                firstserve_eff_rec2 = f"You also winning {serveplusWonLossBalance:.0%} more points than losing when you need to use your serve+1."
                firstserve_eff_rec3 = "This is a really strong basis.  Maintain these strengths."
                firstserve_eff_rec4 = ""
            elif serveplusWonLossBalance > 0:
                firstserve_eff_rec2 = f"You are only winning {serveplusWonLossBalance:.0%} more points than losing when you need to use your serve+1."
            else:
                firstserve_eff_rec2 = f"You are only actually losing {serveplusWonLossBalance*-1:.0%} more points than winning when you need to use your serve+1."
            firstserve_eff_rec3 = "Work on your serve+1 with your coach. Or play points but only allow a maximum of 6 shots.  If the point is still live, you lost the point."
            firstserve_eff_rec4 = ""
        elif First_WonWServe > 0.15:
            firstserve_eff_rec = f"You are winning {First_WonWServe:.0%} of points directly from your 1st Serve.  This is a OK but higher performers get more free points."
            firstserve_eff_rec2 = "Look to work on your serve.  Is it variable enough, or is it predictable?"
            if serveplusWonLossBalance > 0.1:
                firstserve_eff_rec3 = f"On the plus side, you are winning {serveplusWonLossBalance:.0%} more points than losing when you need to use your serve+1."
                firstserve_eff_rec4 = "This is a strong result, given the lack of free points.  Improve the serve & you'll win a lot more."
            if serveplusWonLossBalance > 0:
                firstserve_eff_rec3 = f"You only win marginally more points when you need to use your serve+1."
            else:
                firstserve_eff_rec3 = f"You actually {serveplusWonLossBalance*-1:.0%} lose more points than you win when you need to use you serve+1."
            if FirstFHPlayed_percent == "High":
                firstserve_eff_rec4 = "You use your forehand a lot but its not having impact.  Look to work on the quality of your serve+1 forehand."
            else:
                firstserve_eff_rec4 = "You aren't able to consistently play your Forehand post serve and its not hurting your opponent.  Focus on improving your serve first and see if this rate improves."
        else:
            firstserve_eff_rec = "You aren't getting enough free points from your serve."
            firstserve_eff_rec2 = f"You only win {First_WonWServe:.0%} points directly from your serve.  This is a low level vs your peers."
            firstserve_eff_rec3 = "This means you have to work much harder to win points."
            firstserve_eff_rec4 = "Work on the placement, power and positioning of your 1st serve as a priority."

        if firstserve_eff <= 50:
            firstserve_eff_home1 = "Your First Serve is not effective enough.  The goal is to win directly or generate a return where you create pressure.  Based on outcomes, this is NOT happening enough.  Go to the Serve section for more."
        elif firstserve_eff < 66:
            firstserve_eff_home1 = "Your First Serve Effectiveness could be improved. You have a winning percentage, but loss rate and or rally rate is hurting you.  Go to the Serve section for more details."
        else:
            firstserve_eff_home1 = "Keep it up."

    # <38	low not winning enough	
    elif firstserve_eff <= 33:
        firstserve_eff_rec = "First serve effectiveness is Low."
        firstserve_eff_rec2 = "Desired outcome from a 1st serve is either to win directly, or to generate a ball to attack either open court or opponents weaker side."
        if firstserve_lost - firstserve_won > 0:
            firstserve_eff_rec3 = "However, in this instance more points are being lost than won."
        else: 
            firstserve_eff_rec3 = "However, in this instance, you are not winning many more than you lose."
        firstserve_eff_rec4 = "Look to both improve the 1st serve, and use the +1 to create pressure, with minimal risk."
        # firstserve_eff_rec3 = "Review the video to see what is missing here and how you can achieve the desired outcomes."
        # firstserve_eff_rec4 = "Do you need to strengthen your serve, vary it, use the 'Serve + 1' shot to create more pressure?"
        firstserve_eff_home1 = "Your First Serve is not effective enough.  The goal is to win directly or generate a return where you create pressure.  Based on outcomes, this is NOT happening enough.  Go to the Serve section for more."
    # 		review video: want serve to deliver short ball can attack to either open space or weaker side.  What is missing here?
    # 38-51	mid low positive - not a healthy win	
    # 		either loss rate is too high - > =0.25 & not being compensated for by won points - focus on 
    elif firstserve_eff <50:
            firstserve_eff_rec = "First serve effectiveness is Medium - but on the lower end"
            if (firstserve_lost >= 0.25) :
                firstserve_eff_rec2 = "The loss rate is too high and not being compensated for by the won points enough to create an effective score."
                firstserve_eff_rec3 = "Either you are being attacked directly from your serve, in which case you need to work on your 1st serve."
                firstserve_eff_rec4 = "Or you are being too aggressive on the +1.  Look to create pressure via attacking space, or going to the weaker side, but with low risk."
                # firstserve_eff_rec3 = "Review video to determine, are you making too many mistakes? If so, how can you reduce this?"
                # firstserve_eff_rec4 = "Or if you are being attacked, look to if you should vary your serves, attack space, or go to the weaker side more."
                firstserve_eff_home1 = "Your First Serve is not effective enough.  The goal is to win directly or generate a return where you create pressure.  Based on outcomes, this is NOT happening enough.  Go to the Serve section for more."
    # 		or rally rate is too high - 37% - need to look at how to win more of the points
            else: #firstserve_rally > 0.37:
                firstserve_eff_rec2 = "The percentage of points going to rally is too high."
    #             firstserve_eff_rec3 = "Review the video to see where you played the 'Serve + 1' shot.  How can you change this to create more pressure?"
    #             firstserve_eff_rec4 = ""
                if ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff < 50) ) :
                    firstserve_eff_rec3 = f"You aren't getting free points from your serve, and the majority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is not good enough."
                    firstserve_eff_rec4 = "Look to use the Forehand to create pressure via attacking space, or the weak side.  This pressure will result in more points."
                    # firstserve_eff_rec4 = "Review the video to see how you can use your Forehand to create more pressure."
                elif ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff > 50) ) :
                    firstserve_eff_rec3 = f"You aren't getting free points from your serve.  However the majority of the points where a Serve + 1 is needed employs a Forehand, and its effective."
                    firstserve_eff_rec4 = "Work on your 1st Serve to get more free points."
                    # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more."
                elif ((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff > 50) ) :
                    firstserve_eff_rec3 = f"You aren't getting free points from your serve. Also only a minority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is good."
                    firstserve_eff_rec4 = "Look to improve your 1st Serve to get more free points, and actively try to play the forehand more as it hurts your opponent."
                    # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more. Do you need to vary or improve your serve?"
                else:  #((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff < 50) ) :
                    firstserve_eff_rec3 = f"You aren't getting free points from your serve. A minority of the points where a Serve + 1 is needed employs a Forehand, and the effectiveness of this shot is not good."
                    firstserve_eff_rec4 = "You likely need to improve your serve to be able to get more attackable shots, and you need to use your Forehand to create more pressure."#  Review the video to confirm."
                
                firstserve_eff_home1 = "Your First Serve is not effective enough.  The goal is to win directly or generate a return where you create pressure.  Based on outcomes, this is NOT happening enough.  Go to the Serve section for more."

    elif firstserve_eff < 66 :
            firstserve_eff_rec = "First serve effectiveness is Medium - and on the higher end."
            if firstserve_won >= 0.4:
                firstserve_eff_rec2 = "Win rate is strong here, but the loss or rally rate reduces the effectiveness score"
    #             firstserve_eff_rec3 = "Review the video and compare the points you win from 'Serve + 1'  to those that go to rally or you lose"
    #             firstserve_eff_rec4 ="Ask how you can reduce the number of rally or lost points whilst maintaining win rate"
                if firstservesOnlyProp > 0.2:
                    if ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Whilst the majority of the points where a Serve + 1 is needed employs a Forehand, the effectiveness of this shot is not good enough."
                        firstserve_eff_rec4 = "Your serve and footwork is clearly good.  Set targets to hit to create pressure with your forehand on the serve +1.  This pressure will result in more points."
                        # firstserve_eff_rec4 = "Review the video to see how you can use your Forehand to create more pressure."
                    elif ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve. Also the majority of the points where a Serve + 1 is needed employs a Forehand, and the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "This is highly effective strategy and great execution.  Keep it up."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more."
                    elif ((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  However only a minority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "Your serve is clearly good, but you need to more actively bring your forehand into play to create pressure."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more. Do you need to vary or improve your serve?"
                    else :#((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Unfortunately a minority of the points where a Serve + 1 is needed employs a Forehand, and the effectiveness of this shot is not good."
                        firstserve_eff_rec4 = "You likely need to improve your serve to be able to get more attackable shots, and you need to use your Forehand to create more pressure."#  Review the video to confirm."
                else:
                    if ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Whilst the majority of the points where a Serve + 1 is needed employs a Forehand, the effectiveness of this shot is not good enough."
                        firstserve_eff_rec4 = "Work on your serve (power, placement, variation) to get more points directly.  Also aim for targets with your serve +1 forehand to create pressure."
                        # firstserve_eff_rec4 = "Review the video to see how you can use your Forehand to create more pressure, & work on serve (placement / variation / power) to increase the free points."
                    elif ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Majority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "Your follow up is good but the number of free points is low.  Work on improving placement, power and add variation to your 1st serve."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more, & work on serve (placement / variation / power) to increase the free points."
                    elif ((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  A minority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "You need to work on your 1st serve, and the footwork post the serve so you can employ your forehand more."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more. Do you need to vary or improve your serve?"
                    else:# ((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  A minority of the points where a Serve + 1 is needed employs a Forehand, and the effectiveness of this shot is not good."
                        firstserve_eff_rec4 = "You likely need to improve your serve to be able to get more attackable shots, and you need to use your Forehand to create more pressure."#  Review the video to confirm."

                firstserve_eff_home1 = "Your First Serve Effectiveness could be improved. You have a winning percentage, but loss rate and or rally rate is hurting you.  Go to the Serve section for more details."
            else:
                firstserve_eff_rec2 = "Win rate is lower than 40% which is highly correlated to winning"
    #             firstserve_eff_rec3 = "Review video and look at the won points in comparison to rally & lost points.  How can you replicate more won points?"
    #             firstserve_eff_rec4 = "Can you vary the serve more via type or placement? Can you attack space better? Can you create more variation in your play"
                if firstservesOnlyProp > 0.2:
                    if ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Whilst the majority of the points where a Serve + 1 is needed employs a Forehand, the effectiveness of this shot is not good enough."
                        firstserve_eff_rec4 = "Your serve and footwork is clearly good.  Set targets to hit to create pressure with your forehand on the serve +1.  This pressure will result in more points."
                        # firstserve_eff_rec4 = "Review the video to see how you can use your Forehand to create more pressure."
                    elif ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve. Also the majority of the points where a Serve + 1 is needed employs a Forehand, and the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "This is highly effective strategy and great execution.  Keep it up."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more."
                    elif ((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  However only a minority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "Your serve is clearly good, but you need to more actively bring your forehand into play to create pressure."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more. Do you need to vary or improve your serve?"
                    else :#((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Unfortunately a minority of the points where a Serve + 1 is needed employs a Forehand, and the effectiveness of this shot is not good."
                        firstserve_eff_rec4 = "You likely need to improve your serve to be able to get more attackable shots, and you need to use your Forehand to create more pressure."#  Review the video to confirm."
                else:
                    if ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Whilst the majority of the points where a Serve + 1 is needed employs a Forehand, the effectiveness of this shot is not good enough."
                        firstserve_eff_rec4 = "Work on your serve (power, placement, variation) to get more points directly.  Also aim for targets with your serve +1 forehand to create pressure."
                        # firstserve_eff_rec4 = "Review the video to see how you can use your Forehand to create more pressure, & work on serve (placement / variation / power) to increase the free points."
                    elif ((FH_of_ServePlus1 > 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  Majority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "Your follow up is good but the number of free points is low.  Work on improving placement, power and add variation to your 1st serve."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more, & work on serve (placement / variation / power) to increase the free points."
                    elif ((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff > 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  A minority of the points where a Serve + 1 is needed employs a Forehand, but the effectiveness of this shot is good."
                        firstserve_eff_rec4 = "You need to work on your 1st serve, and the footwork post the serve so you can employ your forehand more."
                        # firstserve_eff_rec4 = "Review the video to see how you can bring in your Forehand more. Do you need to vary or improve your serve?"
                    else:# ((FH_of_ServePlus1 < 0.5) & (firstserve_serveFH_eff < 50) ) :
                        firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  A minority of the points where a Serve + 1 is needed employs a Forehand, and the effectiveness of this shot is not good."
                        firstserve_eff_rec4 = "You likely need to improve your serve to be able to get more attackable shots, and you need to use your Forehand to create more pressure."#  Review the video to confirm."
                firstserve_eff_home1 = "Your First Serve Effectiveness could be improved. You have a winning percentage, but its not as high as desired.  Go to the Serve section to see more details."
    else:
        firstserve_eff_rec = "First serve effectiveness is High."
        if firstserve_won >= 0.4:
            firstserve_eff_rec2 = "Keep it up. You win over 40% of points on First Serve which is highly correlated to winning."
            firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  {FH_of_ServePlus1:.0%} of the points where a Serve + 1 is needed employs a Forehand, and its effectiveness score is {firstserve_serveFH_eff:.0f}."
            firstserve_eff_rec4 = "This is an excellent result and should be maintained, whilst working on other areas of your game."
            # firstserve_eff_rec4 = "You can watch the video to see why this is so effective so you can maintain & replicate this behaviour."
        else:
            firstserve_eff_rec2 = "Score is good but you don't win over 40% of points on First Serve which is highly correlated to winning."
            firstserve_eff_rec3 = f"You get {firstservesOnlyProp:.0%} points directly from 1st Serve.  {FH_of_ServePlus1:.0%} of the points where a Serve + 1 is needed employs a Forehand, and its effectiveness score is {firstserve_serveFH_eff:.0f}."
            firstserve_eff_rec4 = "This is an excellent result and should be maintained, whilst working on other areas of your game."
            # firstserve_eff_rec4 = "Review the video to see where you can win more via either improving the serve or the serve + 1."
        firstserve_eff_home1 = "Keep it up."

    if firstserve_eff < 66:
        firstserve_eff_rating = "WorkOn"
    else:
        firstserve_eff_rating = "OK"
            

    # 2nd serve effectiveness recommendations
    dfsOfLost = (dfs / (secondServe_total_lostwinner + secondServe_total_lostforced + secondServe_total_lostmistake))

    if (type == "single") & (secondserves_freq < 20):
        secondserve_eff_rec = f"Note: There are only{secondserves_freq: .0f} 2nd Serve points.  Recommendations could be volatile on such little data."
        secondserve_eff_rec2 = "Therefore no deepdive analysis is performed here."
        secondserve_eff_rec3 = "However aggregated insights on 2nd Serve Performance are generated in your Personal Summary."
        secondserve_eff_rec4 = ""
        secondserve_eff_rec5 = ""
        secondserve_eff_home1 = "There aren't enough points to assess 2nd Serve points for this match.  Check out the insights in your Personal Summary."
    else:
        # 33	very bad					
        # 	Opponent sees this as opprtunity to attack.  You are losing a high proportion of points.					
        # elif secondserve_eff < 33:
        if secondserve_eff < 33:
            secondserve_eff_rec = "Second Serve: Opponent sees this as an opportunity to attack.  You are losing a high proportion of points."
            if dfsOfLost >= 0.4:
                secondserve_eff_rec2 = f"Points lost immediately from the 2nd Serve account for{dfsOfLost: .0%} of LOST points on Second Serve."
                if df_rate >= 0.2:
                    secondserve_eff_rec3 = f"This translates to {df_rate:.0%} of ALL points started with your 2nd serve.  Look to address this."
                else:
                    secondserve_eff_rec3 = ""
            else: 
                secondserve_eff_rec2 = ""
                secondserve_eff_rec3 = ""
        # 		if df account for more than 40% of lost points, point it out 				
        # 			where 2nd pts played is over 30, if df rate above 20%, look to address			
        # 			else df rate less than 20% is OK			
        # 		where 2nd pts played is over 30, if df rate above 20%, look to address				
        # 	Review video & ask how can you reduce their ability to attack? Does you serve need more power, spin, or better placement?					
            secondserve_eff_rec4 = "Improve your 2nd serve to prevent your opponent from attacking you directly."
            # secondserve_eff_rec4 = "Review 2nd serve video & ask how can you reduce the opponents ability to attack you? Do you need more power, spin, better and more varied placement"
        # 	Review video: review your foot preparation for the "Serve +1" - are you setting up to absorb pressure?		
            secondserve_eff_rec5 = "Work on your footwork post your serve.  Ensure you are setting up to absorb pressure."
            # secondserve_eff_rec5 = "Review 2nd serve video: is your footwork for 'Serve + 1' setting up to absorb pressure?"
            secondserve_eff_home1 = "Your Second Serves are not ending in your favour.  Go to the Serve section for recommendations how to resolve this."
        # 33-50	still losing on average					
        elif secondserve_eff < 50:
            if secondserve_won > secondserve_lost:
                secondserve_eff_rec = "You having a winning balance on 2nd serve, but your score is hurt by losing points directly from your serve."
            else:
                secondserve_eff_rec = "Still losing pts on Second Serve points on average."
        # 	if loss rate higher than 25%, loss rate is still high vs Rally and Won points.					
            if (secondserve_lost ) >= 0.25:
        # 		Note that dfs account for x% of lost points				
        # 			where 2nd pts played is over 30, if df rate above 20%, look to address			
                secondserve_eff_rec2 = "Loss rate of 2nd Serve points is still high."
                secondserve_eff_rec3 = "Your opponent will naturally see this as an opportunity to attack.  Your aim is to nullify this and get to a rally and neutral situation."
                if dfsOfLost > 0.5:
                    secondserve_eff_rec4 = f"Points lost immediately from the 2nd Serve account for{dfsOfLost: .0%} of lost points on Second Serve."
                    secondserve_eff_rec5 = "Work on your Second Serve so there is greater consistency and it cannot be attacked so easily."
                else: 
                    secondserve_eff_rec4 = "Losses are primarily coming from the Serve +1 shot."
                    secondserve_eff_rec5 = "Work on your footwork to absorb pressure and progress to a rally situation."
        # 		Review video and see how can reduce loss points? Does your serve need improvement so you are less under pressure?				
                # secondserve_eff_rec5 = "Review 2nd serve video.  How can you reduce lost points? Does serve need improvement to reduce pressure? Are you playing too aggressively behind your 2nd serve"
        # 		Review to see if you are playing too aggressively immediately behind 2nd serve.  Can you build the point more?				

        # 	if less than 25%, loss rate is under control.  Review points, can you see how you could create more pressure?					
            else:
                secondserve_eff_rec2 = "Loss rate is under control."
                # secondserve_eff_rec3 = "Review 2nd serve video: can you see where you can create more pressure to win more of the points rather than rally?"
                secondserve_eff_rec3 = "Can you see where you can create more pressure to win more of the points rather than rally?"
                secondserve_eff_rec4 = "How often do you get your preferred shot in play post 2nd serve? Can you increase this?"
                secondserve_eff_rec5 = ""
            secondserve_eff_home1 = "Your Second Serves are not ending in your favour.  Go to the Serve section for recommendations how to resolve this."
        # 50 - 66	parity and to rally					
        elif secondserve_eff < 66:
            secondserve_eff_rec = "Getting to at least parity on 2nd serve and moving towards rally on average."
        # 	if loss rate greater than 30%, winning a good proportion but losing high % too.					
            if secondserve_lost >= 0.3:
                secondserve_eff_rec2 = f"Winning a good proportion of points, but a loss rate of{(secondserve_lost ): .0%} is still high."
                if dfsOfLost > 0.5:
                    secondserve_eff_rec3 = f"Points lost immediately from the 2nd Serve account for{dfsOfLost: .0%} of lost points on Second Serve."
                    secondserve_eff_rec4 = "Work on your Second Serve so there is greater consistency and it cannot be attacked so easily."
                else: 
                    secondserve_eff_rec3 = "Losses are primarily coming from the Serve +1 shot."
                    secondserve_eff_rec4 = "Work on your footwork to absorb pressure and progress to a rally situation."
                # secondserve_eff_rec3 = "Review video to see how you can reduce lost points whilst not sacrificing win rate"
                # if df_rate >= 0.2:
                #     secondserve_eff_rec4 = f"You have a Double Fault rate of{df_rate:.0%}.  However given 2nd serve is effective, this may not be an area of concern."
                # else: 
                #     secondserve_eff_rec4 = ""
                secondserve_eff_rec5 = ""
        # 		Review video to see if how reduce errors				
        # 	else, loss rate is under control and winning more, can look to get more points rather than go to rally					
            else: 
                secondserve_eff_rec2 = "Loss rate is under control on 2nd serve."
                secondserve_eff_rec3 = "To level up your 2nd score, look where you can win more points rather than going to rally."
                # secondserve_eff_rec3 = "Review video to see where you can win more points rather than going to rally."
        # 	where 2nd pts played is over 30, if df rate above 20%, look to address - though less pressing given effectiveness of 2nd serve.  Perhaps its worth it.					
                if df_rate >= 0.2:
                    secondserve_eff_rec4 = f"You lose {df_rate:.0%} of points immediately after your 2nd serve.  However given 2nd serve is effective, this may not be an area of concern."
                else: 
                    secondserve_eff_rec4 = ""
                secondserve_eff_rec5 = ""
            secondserve_eff_home1 = "Whilst you Second Serves are on average in your favour, its very marginal.  Go to the Serve section for recommendations how to resolve this."
        # 66	winning on average					
        else:
            if (secondserve_won - secondserve_lost) > 0.15:
                secondserve_eff_rec = "You are winning points behind your 2nd serve on average."
                secondserve_eff_rec2 = "You can see if there are more opportunities to win points."
                secondserve_eff_rec3 = "However, given the strong score, this is a low priority."
            else:
                secondserve_eff_rec = "You don't win signficantly more points than you lose on 2nd serve."
                secondserve_eff_rec2 = f"However {secondserve_serveonly_freq/(secondserve_won*secondserves_freq):.0%} of your 2nd serve won points come directly from your serve."
                secondserve_eff_rec3 = "This boosts your effective score.  Look to progress to rally when the you need to play the serve +1."
        # 	on average, getting to rally from 2nd serve and winning.  Review to see if there are opportunities to win more here.					
        # 	where 2nd pts played is over 30, if df rate above 20%, look to address - though less pressing given effectiveness of 2nd serve.  Perhaps its worth it.					
                                
            if df_rate >= 0.2:
                secondserve_eff_rec4 = f"You lose {df_rate:.0%} of points immediately after your 2nd serve.  However given 2nd serve is effective, this may not be an area of concern."
                secondserve_eff_rec5 = ""
            else: 
                secondserve_eff_rec4 = ""
                secondserve_eff_rec5 = ""
            secondserve_eff_home1 = "Keep it up."

    # if (secondserve_eff < 66) & ( secondserves_freq >= 30):
    # removed as want to show this now... have the warning so its OK
    if (secondserve_eff < 66) :
        secondserve_eff_rating = "WorkOn"
    else :
        secondserve_eff_rating = "OK"



    #other elements to build a recommendation on

    #  perhaps have 1st serves as a structure - some elements fall away, some are always there
        #have first serves, second serves - includes all elements, those elements might be blanks - if blank, not selected for the nl
            #second serves filtered and add an extra nl before paste the line
                #how to combine to total recommendations - need to look at but this is a good start

    # first serve loss rate - if too high - look to reduce this - strengthen first serve or play more solid ball
        # is this view dependent on how lose - direct from serve or as next point?  is this 2 recommendations - can be but group it to be me more generic
            #needs additional comments about why though
            # what is the impact of this however - how calculate

    #first serve how loss - if mistakes are high - raise flag, if determination, raise flag to strengthn point to prevent this  - restict to #points
            
    # first serve rally rate - if too high -& first serve win rate not very high look to be more aggressive - dictate point earlier
        # move rally rate & win rate to average levels
        #if  win rate is ok, point out can save energy
        
    # first serve - how win - if mistakes to determination below x - highlight this as potential concern - flag it - add a column to include - figure order later

    #critical & break points vs non critical - show if break is worse than other - show chance of winning -
        # change ranges for recommendation - if highly above, keep it up, try to bump it - as have more winning rate
            #do you factor in the 1st serve rate here - if its already high?
            
            
    # winrate_1stserve = ((totalserves * firstserverate * firstserve_winrate) + (totalserves * firstserverate * firstserve_rallyrate * firstserverally_winrate )) / firstserves_freq
    # winrate_2ndserve = ((totalserves * (1-firstserverate) * secondserve_winrate) + (totalserves * (1-firstserverate) * secondserve_rallyrate * secondserverally_winrate )) / secondserves_freq



    if (crit_totalserve_freq < 30) | (breakball_allserves >=30) | (crit_nonbreakball_allserves >= 30):
        critical_1stserve_rec = ""
        critical_1stserve_rating = ""
        firstserve_eff_home2 = ""
        #if serve rate already high and not lower on critical, keep it up
    else:
        if ((noncrit_1stserve_freq / noncrit_totalserve_freq)  > 0.7) & ((crit_1stserve_freq / crit_totalserve_freq) > (noncrit_1stserve_freq / noncrit_totalserve_freq) ):
            critical_1stserve_rec = "1st Serve rate is good and is better on critical points - Keep it up"
            critical_1stserve_rating = "OK"
            firstserve_eff_home2 = "Keep it up."
            #if rate high, and lower on critical, highlight  improved chance of winning
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  > 0.7):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                critical_1stserve_rec = f"1st Serve rate is less on critical points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for critical points"
                critical_1stserve_rating = "WorkOn"
                firstserve_eff_home2 = "On critical points, where there is more pressure on the outcome, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
            else:
                critical_1stserve_rec = f"1st Serve rate is less on critical points. On average, 16% more points are won behind 1st serve.  Try to increase the 1st serve rate for critical points"
                critical_1stserve_rating = "WorkOn"
                firstserve_eff_home2 = "On critical points, where there is more pressure on the outcome, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
                #if rate not high, and rate on critical less than 5 pts more, highlight improved chance of winning
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  <= 0.7) & ((crit_1stserve_freq / crit_totalserve_freq) < ((noncrit_1stserve_freq / noncrit_totalserve_freq)) ):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                critical_1stserve_rec = f"1st Serve rate is worse on critical points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for critical points"
                critical_1stserve_rating = "WorkOn"
                firstserve_eff_home2 = "On critical points, where there is more pressure on the outcome, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
            else:
                critical_1stserve_rec = f"1st Serve rate is less on critical points. On average, 16% more points are won behind 1st serve.  Try to increase the 1st serve rate for critical points"
                critical_1stserve_rating = "WorkOn"
                firstserve_eff_home2 = "On critical points, where there is more pressure on the outcome, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  <= 0.7) & ((crit_1stserve_freq / crit_totalserve_freq) < ((noncrit_1stserve_freq / noncrit_totalserve_freq) + 0.05) ):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                critical_1stserve_rec = f"1st Serve rate is not significantly better on critical points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for critical points"
                critical_1stserve_rating = "WorkOn"
                firstserve_eff_home2 = "On critical points, where there is more pressure on the outcome, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
            else:
                critical_1stserve_rec = f"1st Serve rate is less on critical points. On average, 16% more points are won behind 1st serve.  Try to increase the 1st serve rate for critical points"
                critical_1stserve_rating = "WorkOn"
                firstserve_eff_home2 = "On critical points, where there is more pressure on the outcome, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
        else:
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                critical_1stserve_rec = f"1st serve rate is significantly better on critical points than non critical points.  Keep it up.  You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to generally increase your 1st serve however"   
                critical_1stserve_rating = "OK"
                firstserve_eff_home2 = "Keep it up."
            else:
                critical_1stserve_rec = f"1st serve rate is significantly better on critical points than non critical points.  Keep it up.  20% more points are won behind 1st serve.  Try to generally increase your 1st serve however" 
                critical_1stserve_rating = "OK"
                firstserve_eff_home2 = "Keep it up."
                # if rate not high, and rate already 5 pts - keep it up but try to generally increase 1st serve rate and maintain this behaviour

        # critical first - bigger- 1st serve rate & outcome
        # break points - where have enough - what is that number
    if breakball_allserves < 30:
        break_1strate_rec = ""
        break_1strate_ranking = ""
        firstserve_eff_home3 = ""
    else:
        break_1strate_diff = breakball_1stserverate - (noncrit_1stserve_freq / noncrit_totalserve_freq)
        if ((noncrit_1stserve_freq / noncrit_totalserve_freq)  > 0.7) & (breakball_1stserverate > (noncrit_1stserve_freq / noncrit_totalserve_freq) ):
            break_1strate_rec = "1st Serve rate is good and is better on break points - Keep it up"
            break_1strate_ranking = "OK"
            firstserve_eff_home3 = "Keep it up."
                #if rate high, and lower on critical, highlight  improved chance of winning
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  > 0.7):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                break_1strate_rec = f"1st Serve rate is less on break points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for break points"
                break_1strate_ranking = "WorkOn"
                firstserve_eff_home3 = "On break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on break points for better outcomes. More details are in the Serve section."
            else:
                break_1strate_rec = f"1st Serve rate is less on break points. On average, 20% more points are won behind 1st serve.  Try to increase the 1st serve rate for break points"
                break_1strate_ranking = "WorkOn"
                firstserve_eff_home3 = "On break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on break points for better outcomes. More details are in the Serve section."
                #if rate not high, and rate on critical less than 5 pts more, highlight improved chance of winning
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  <= 0.7) & (breakball_1stserverate < ((noncrit_1stserve_freq / noncrit_totalserve_freq)) ):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                break_1strate_rec = f"1st Serve rate is worse on break points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for break points"
                break_1strate_ranking = "WorkOn"
                firstserve_eff_home3 = "On break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on break points for better outcomes. More details are in the Serve section."
            else:
                break_1strate_rec = f"1st Serve rate is less on break points. On average, 20% more points are won behind 1st serve.  Try to increase the 1st serve rate for break points"
                break_1strate_ranking = "WorkOn"
                firstserve_eff_home3 = "On break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on break points for better outcomes. More details are in the Serve section."
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  <= 0.7) & (breakball_1stserverate < ((noncrit_1stserve_freq / noncrit_totalserve_freq) + 0.05) ):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                break_1strate_rec = f"1st Serve rate is not significantly better on break points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for break points"
                break_1strate_ranking = "WorkOn"
                firstserve_eff_home3 = "On break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on break points for better outcomes. More details are in the Serve section."
            else:
                break_1strate_rec = f"1st Serve rate is less on break points. On average, 20% more points are won behind 1st serve.  Try to increase the 1st serve rate for critical points"
                break_1strate_ranking = "WorkOn"
                firstserve_eff_home3 = "On break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on break points for better outcomes. More details are in the Serve section."
        else:
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                break_1strate_rec = f"1st serve rate is significantly better on break points than non critical points.  Keep it up.  You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to generally increase your 1st serve however"   
                break_1strate_ranking = "OK"
                firstserve_eff_home3 = "Keep it up."
            else:
                break_1strate_rec = f"1st serve rate is significantly better on break points than non critical points.  Keep it up.  20% more points are won behind 1st serve.  Try to generally increase your 1st serve however" 
                break_1strate_ranking = "OK"
                firstserve_eff_home3 = "Keep it up."

    #for crit non break
    if crit_nonbreakball_allserves < 30:
        crit_nonbreak_1strate_rec = ""
        crit_nonbreak_1strate_rating =""
        firstserve_eff_home4 = ""
    else:
        crit_nonbreak_1strate_diff = crit_nonbreakball_1stserverate - (noncrit_1stserve_freq / noncrit_totalserve_freq)
        if ((noncrit_1stserve_freq / noncrit_totalserve_freq)  > 0.7) & (crit_nonbreakball_1stserverate > (noncrit_1stserve_freq / noncrit_totalserve_freq) ):
            crit_nonbreak_1strate_rec = "1st Serve rate is good and is better on pre-break points - Keep it up"
            crit_nonbreak_1strate_rating = "OK"
            firstserve_eff_home4 = "Keep it up."
                #if rate high, and lower on critical, highlight  improved chance of winning
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  > 0.7):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                crit_nonbreak_1strate_rec = f"1st Serve rate is less on pre-break points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for pre-break points"
                crit_nonbreak_1strate_rating = "WorkOn"
                firstserve_eff_home4 = "On critical but not break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
            else:
                crit_nonbreak_1strate_rec = f"1st Serve rate is less on break points. On average, 20% more points are won behind 1st serve.  Try to increase the 1st serve rate for pre-break points"
                crit_nonbreak_1strate_rating = "WorkOn"
                firstserve_eff_home4 = "On critical but not break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
                #if rate not high, and rate on critical less than 5 pts more, highlight improved chance of winning
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  <= 0.7) & (breakball_1stserverate < ((noncrit_1stserve_freq / noncrit_totalserve_freq)) ):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                crit_nonbreak_1strate_rec = f"1st Serve rate is worse on critical but non break points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for pre-break points"
                crit_nonbreak_1strate_rating = "WorkOn"
                firstserve_eff_home4 = "On critical but not break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
            else:
                crit_nonbreak_1strate_rec = f"1st Serve rate is less on pre-break points. On average, 20% more points are won behind 1st serve.  Try to increase the 1st serve rate for pre-break points"
                crit_nonbreak_1strate_rating = "WorkOn"
                firstserve_eff_home4 = "On critical but not break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
        elif ((noncrit_1stserve_freq / noncrit_totalserve_freq)  <= 0.7) & (crit_nonbreakball_1stserverate < ((noncrit_1stserve_freq / noncrit_totalserve_freq) + 0.05) ):
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                crit_nonbreak_1strate_rec = f"1st Serve rate is not significantly better on pre-break points. You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to increase the 1st serve rate for pre-break points"
                crit_nonbreak_1strate_rating = "WorkOn"
                firstserve_eff_home4 = "On critical but not break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
            else:
                crit_nonbreak_1strate_rec = f"1st Serve rate is less on pre-break points. On average, 20% more points are won behind 1st serve.  Try to increase the 1st serve rate for critical points"
                crit_nonbreak_1strate_rating = "WorkOn"
                firstserve_eff_home4 = "On critical but not break points, your 1st serve rate is lower than non critical points.  Look to maximise your 1st serve rate on these points for better outcomes. More details are in the Serve section."
        else:
            if winrate_1stserve > (winrate_2ndserve + 0.05):
                crit_nonbreak_1strate_rec = f"1st serve rate is significantly better on pre-break points than non critical points.  Keep it up.  You win{(winrate_1stserve - winrate_2ndserve): .0%} more points behind 1st serve.  Try to generally increase your 1st serve however"   
                crit_nonbreak_1strate_rating = "OK"
                firstserve_eff_home4 = "Keep it up."
            else:
                crit_nonbreak_1strate_rec = f"1st serve rate is significantly better on pre-break points than non critical points.  Keep it up.  20% more points are won behind 1st serve.  Try to generally increase your 1st serve however" 
                crit_nonbreak_1strate_rating = "OK"
                firstserve_eff_home4 = "Keep it up."

    # define what critical points are - where have critical, explain the difference, but both are included here - 
    if critical_1stserve_rec != "":
        define_criticalpts_rec = "Critical points are points which could result in a break in serve (30-40), or are points that could lead to a breakpoint if not won (30-30).  There is not enough data to split them out at present"
        define_criticalpts_rating = critical_1stserve_rating
    #when have only pre break, explain diff but not enough data for break
    elif crit_nonbreak_1strate_rec != "": 
        define_criticalpts_rec = "Critical points are points which could result in a break in serve (30-40), or are points that could lead to a breakpoint if not won (30-30).  There is not enough data show break points, but pre break is shown"
        define_criticalpts_rating = crit_nonbreak_1strate_rating
    #where have both explain them - note behaviour may vary
    elif (crit_nonbreak_1strate_rec != "") & (break_1strate_rec != ""):   
        define_criticalpts_rec = "Critical points are points which could result in a break in serve (30-40), or are points that could lead to a breakpoint if not won (30-30).  Recommendations for each are shown below"
        if (break_1strate_ranking == "WorkOn" | crit_nonbreak_1strate_rating == "WorkOn"):
            define_criticalpts_rating = "WorkOn"
        else:
            define_criticalpts_rating = "OK"
    else:
        define_criticalpts_rec = ""
        define_criticalpts_rating = ""
                #deuce & adv views
    #deuce first serve rate - if more than 7 pts different to average - call it out
        #same for ad

    #if deuce 1st effectiveness 10pts worse than average - call it out - look at how winning & losing here - depends on numbers - 30
        # same for ad



    #### RETURN 1st Serves Recommendations ###
    # how factor in low frequency # caution flag?				
                    
    firstreturn_rate_diff = (firstreturn_rate_average - firstreturn_rate)
    firstreturn_lost_freq = firstreturn_lost * firstreturns_freq
    # firstreturn_lossmistake_rate = (firstReturn_total_lostmistake / firstreturn_lost_freq)


    if firstreturns_freq < 30:
            firstreturn_eff_rec_caution = f"These 1st Return insights are based{firstreturns_freq: .0f} points. The data is volatile at this level.  Consider it but note it can change easily with more data. A more robust overview is available in your Personal Summary as it considers performance over multiple matches."
    else:
        firstreturn_eff_rec_caution = ""
    #First_eff_cons (High medium low), LostOver40Rate (%), FirstCompletionRate (kinda like consistency low, medium, high), numGamesover40 (count), FirstCompletionRate_Over40 (kinda like consistency low, medium, high)
    if type == "multi":
        if firstreturn_eff < 50:
            first_return_home1 = "The low 1st Return Effectiveness is score is driven by a high loss rate.  See more details in the Return section."
        else:
            first_return_home1 = "Keep it up"
        if First_eff_cons == "High":
            firstreturn_eff_rec = f"The 1st Return Effectiveness score is {firstreturn_eff:.0f} & consistent."
        elif First_eff_cons == "Medium":
            firstreturn_eff_rec = f"The 1st Return Effectiveness score is {firstreturn_eff:.0f} though not always consistent."
        else:
            firstreturn_eff_rec = f"The 1st Return Effectiveness score is {firstreturn_eff:.0f} though there is a lot of variability, meaning there are areas to improve."
        if LostOver40Rate < 0.3:
            firstreturn_eff_rec2 = f"In only {LostOver40Rate:.0%} of matches do you lose more than 40% of points in the 1stReturn situation.  This is great as winning 40% on 1st serve is highly correlated with winning."
        else:
            firstreturn_eff_rec2 = f"In {LostOver40Rate:.0%} of matches, you lose more than 40% of points in the 1stReturn situation.  Our aim here is to reduce this percentage as its highly correlated with winning."
        if numGamesover40 <= 2:
            if FirstCompletionRate == "High":
                firstreturn_eff_rec3 = f"You often get to play your Return+1, and have overall lose rate in control.  This is great keep it up."
                firstreturn_eff_rec4 = f"You could potentially challenge yourself to be more aggressive on return to win more, but this is a low priority."
                #where not enough slice returns
                if (( FirstReturn_FH_Freq < 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    firstreturn_eff_rec5 = "" # Shot analytics here - nothing because frequencies are too low
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_BH_Eff ):
                        firstreturn_eff_rec5 = "Your Backhand return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Backhand.  Look to train this."
                    #where not enough backhand returns
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the slice is actually more effective. Are you protecting your slice? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Slice.  Look to train this."
                #all shots can be assessed
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff ) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_BH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Backhand drive.  Look to train this or play more backhands."
                    elif (FirstReturn_BH_Eff < FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is more effective than your Backhand drive.  Look to train this or play more slice returns."
                    else: #more variations I can add here but its good already
                        firstreturn_eff_rec5 = ""
                else: 
                    firstreturn_eff_rec5 = ""

            elif FirstCompletionRate == "Medium":
                firstreturn_eff_rec3 = f"Sometimes you complete a lot of returns, and other times not."
                firstreturn_eff_rec4 = f"This suggests train on both the quality of your return, the recovery for the Return +1, where your aim should be to make the server uncomfortable."
                #where not enough slice returns
                if (( FirstReturn_FH_Freq < 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    firstreturn_eff_rec5 = "" # Shot analytics here - nothing because frequencies are too low
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_BH_Eff ):
                        firstreturn_eff_rec5 = "Your Backhand return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Backhand.  Look to train this."
                    #where not enough backhand returns
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the slice is actually more effective. Are you protecting your slice? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Slice.  Look to train this."
                #all shots can be assessed
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff ) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_BH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Backhand drive.  Look to train this or play more backhands."
                    elif (FirstReturn_BH_Eff < FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is more effective than your Backhand drive.  Look to train this or play more slice returns."
                    else: #more variations I can add here but its good already
                        firstreturn_eff_rec5 = ""
                else: 
                    firstreturn_eff_rec5 = ""
            else:
                firstreturn_eff_rec3 = f"Whilst your loss rate is under control, the server still wins points quickly."
                firstreturn_eff_rec4 = f"You are either too aggressive or too passive. Adjust your returns so the opponent cannot win points so quickly."
                #where not enough slice returns
                if (( FirstReturn_FH_Freq < 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    firstreturn_eff_rec5 = "" # Shot analytics here - nothing because frequencies are too low
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_BH_Eff ):
                        firstreturn_eff_rec5 = "Your Backhand return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Backhand.  Look to train this."
                    #where not enough backhand returns
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the slice is actually more effective. Are you protecting your slice? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Slice.  Look to train this."
                #all shots can be assessed
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff ) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_BH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Backhand drive.  Look to train this or play more backhands."
                    elif (FirstReturn_BH_Eff < FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is more effective than your Backhand drive.  Look to train this or play more slice returns."
                    else: #more variations I can add here but its good already
                        firstreturn_eff_rec5 = ""
                else: 
                    firstreturn_eff_rec5 = ""
        else:
            if FirstCompletionRate_Over40 == "High":
                firstreturn_eff_rec3 = f"Where you lose over 40% of points to 1stServe, you very often progress to the Return+1 shot."
                firstreturn_eff_rec4 = f"So your work on is to improve the recovery to the Return+1, and to play a shot that makes the server uncomfortable."
                #where not enough slice returns
                if (( FirstReturn_FH_Freq < 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    firstreturn_eff_rec5 = "" # Shot analytics here - nothing because frequencies are too low
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_BH_Eff ):
                        firstreturn_eff_rec5 = "Your Backhand return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Backhand.  Look to train this."
                    #where not enough backhand returns
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the slice is actually more effective. Are you protecting your slice? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Slice.  Look to train this."
                #all shots can be assessed
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff ) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_BH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Backhand drive.  Look to train this or play more backhands."
                    elif (FirstReturn_BH_Eff < FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is more effective than your Backhand drive.  Look to train this or play more slice returns."
                    else: #more variations I can add here but its good already
                        firstreturn_eff_rec5 = ""
                else: 
                    firstreturn_eff_rec5 = ""
            elif FirstCompletionRate_Over40 == "Medium":
                firstreturn_eff_rec3 = f"Where you exceed that 40% threshold, sometimes you complete a lot of returns, and other times not."
                firstreturn_eff_rec4 = f"This suggests you need to train both the quality of your return, & the recovery for the Return +1, where your aim should be to make the server uncomfortable."
                #where not enough slice returns
                if (( FirstReturn_FH_Freq < 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    firstreturn_eff_rec5 = "" # Shot analytics here - nothing because frequencies are too low
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_BH_Eff ):
                        firstreturn_eff_rec5 = "Your Backhand return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Backhand.  Look to train this."
                    #where not enough backhand returns
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the slice is actually more effective. Are you protecting your slice? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Slice.  Look to train this."
                #all shots can be assessed
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff ) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_BH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Backhand drive.  Look to train this or play more backhands."
                    elif (FirstReturn_BH_Eff < FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is more effective than your Backhand drive.  Look to train this or play more slice returns."
                    else: #more variations I can add here but its good already
                        firstreturn_eff_rec5 = ""
                else: 
                    firstreturn_eff_rec5 = ""
            else:
                firstreturn_eff_rec3 = f"The server wins points quickly when you lose over 40% of points."
                firstreturn_eff_rec4 = f"Work on returning 1st serves in a way that prevents your opponent from hurting you."
                #where not enough slice returns
                if (( FirstReturn_FH_Freq < 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    firstreturn_eff_rec5 = "" # Shot analytics here - nothing because frequencies are too low
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq < 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_BH_Eff ):
                        firstreturn_eff_rec5 = "Your Backhand return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Backhand.  Look to train this."
                    #where not enough backhand returns
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq < 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the slice is actually more effective. Are you protecting your slice? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_FH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Forehand.  Look to train this."
                    else: 
                        firstreturn_eff_rec5 = "Your Forehand return is less effective than your Slice.  Look to train this."
                #all shots can be assessed
                elif (( FirstReturn_FH_Freq > 30) & ( FirstReturn_BH_Freq > 30) & ( FirstReturn_Slice_Freq > 30)  ):
                    if ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < FirstReturn_BH_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, but the backhand is actually more effective. Are you protecting your backhand? You don't need to."
                    elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff > FirstReturn_BH_Eff ) & (FirstReturn_FH_Eff > FirstReturn_Slice_Eff )):
                        firstreturn_eff_rec5 = "You play a high percentage of forehand returns, and it is more effective. Don't forget to train the backhand in case a server can target it."
                    elif (FirstReturn_BH_Eff > FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is less effective than your Backhand drive.  Look to train this or play more backhands."
                    elif (FirstReturn_BH_Eff < FirstReturn_Slice_Eff ):
                        firstreturn_eff_rec5 = "Your Slice return is more effective than your Backhand drive.  Look to train this or play more slice returns."
                    else: #more variations I can add here but its good already
                        firstreturn_eff_rec5 = ""
                else: 
                    firstreturn_eff_rec5 = ""

    elif firstreturn_eff <= 33:
    # 		 where loss rate is above 40% - reduce this - 1st serve 40% highly correlated with win		
    # <33	Low - losing on average and within return +1			
    # -2				
        first_return_home1 = "The low 1st Return Effectiveness is score is driven by a high loss rate.  See more details in the Return section."
        if firstreturn_lost >= 0.4:
            firstreturn_eff_rec = f"Loss rate on First Returns is{firstreturn_lost: .0%}.  You should aim to reduce this as a 1st Serve win rate of 40% is highly correlated with winning. Evaluate how you can neutralise the serve & serve +1 and get to rally."
    # 			if return rate < 10% off, flag	
                
            # else there is a significant difference between won & lost points
        else:
            firstreturn_eff_rec = f"There is a significant difference between won & lost points on Return.  Need to look to reduce this difference."
        if firstreturn_rate_diff > 0.1:
            firstreturn_eff_rec2 = f"Returning against First Serve, you only get to play the return+1 in {firstreturn_rate*100:.0%} of points: This is {firstreturn_rate_diff*100: .0f}pts worse than average. Look to improve the quality of your return so you get further into the game."
        else:
    #         firstreturn_eff_rec2 = "" # could reference the mistake rate & which shots working better here
            if (( FirstReturn_FH_Eff < 50) & ( FirstReturn_BH_Eff < 50) & ( FirstReturn_Slice_Eff < 50)  ):
                firstreturn_eff_rec2 = f"All return shots are ineffective.  Consider improving your footwork or your starting position so you can neutralise the serve and get the point to a rally."
            elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < 50)):
                firstreturn_eff_rec2 = f"Forehand makes up {FirstReturn_FH_prop:.0%} of First Serve Returns but is not effective"
            elif ((FirstReturn_BH_Eff < 50) & ((FirstReturn_Slice_Eff -FirstReturn_BH_Eff )> 15 )  & (FirstReturn_BH_prop > 0.2) ):
                firstreturn_eff_rec2 = "Backhand return is ineffective. Consider using slice more in matches and working on your Backhand return in training."
            elif ((FirstReturn_BH_Eff < 50) & (FirstReturn_BH_prop > 0.3 ) ):
                firstreturn_eff_rec2 = "Backhand return has a low effectiveness score."
            else:
                firstreturn_eff_rec2 = ""
    # 		Aim complete returns & prevent attack.		
        firstreturn_eff_rec3 = "Given the score, your aim here should be to have a high return rate, and prevent immediate attack."
            #caution if less than 30
    # 		Review vid - are returns too short & attackable? How can you neutralise the 1st serve to get to rally?		
        firstreturn_eff_rec4 = "What shot gives your the best chance to recover to the middle and not be immediately under pressure?"
        # firstreturn_eff_rec4 = "Review the 1st Return Video and see if your returns are too short & attackable."
        firstreturn_eff_rec5 = "Set yourself a target zone in line with your strategy.  Middle beyond the Service Line is good."
    # 			define a target zone	
    # 33-50	Medium - low - losing and tendancy for opponent to win within return +1			
    elif firstreturn_eff < 50:
        firstreturn_eff_rec = "First Return Score is Medium but on the Low end."
    # -1				
    # 		if loss rate is above 40% - reduce this - 1st serve 40% highly correlated with win		
        if firstreturn_lost >= 0.4:
            firstreturn_eff_rec2 = f"Loss rate on First Returns is{firstreturn_lost: .0%}.  Reduce this. 1st Serve win rate of 40% is highly correlated with winning."
    # 			if return rate < 10% off, flag	
            if firstreturn_rate_diff > 0.1:
                firstreturn_eff_rec3 = f"Returning against First Serve, you only get to play the return+1 in {firstreturn_rate*100:.0%} of points: This is {firstreturn_rate_diff*100: .0f}pts worse than average. Look to improve the quality of your return so you get further into the game."
            elif firstreturn_rate_diff < -0.1:
                firstreturn_eff_rec3 = f"Returning against First Serve, you manage to play the return+1 in {firstreturn_rate*100:.0%} of points. This is {firstreturn_rate_diff*-100: .0f}pts better than average. You are doing a great job of neutralising the serve."
            else:
                firstreturn_eff_rec3 = f"When facing the 1st Serve, you play the return+1 in {firstreturn_rate: .0%} of points, which is in line with average."
    # 			Review video - how can you reduce the points you lose? Is the return too short, how you can stop him hurting you?	
    #         firstreturn_eff_rec4 = "Review the First Return Video.  How can you reduce the points you lose? How can you stop him hurting you?"
    #         firstreturn_eff_rec5 = ""# add in & flip the piece about the shots & mistake rate
            
            if (( FirstReturn_FH_Eff < 50) & ( FirstReturn_BH_Eff < 50) & ( FirstReturn_Slice_Eff < 50)  ):
                firstreturn_eff_rec4 = f"All return shots are ineffective.  Consider improving your footwork or your starting position."
            elif ((FirstReturn_FH_prop > 0.6) & (FirstReturn_FH_Eff < 50)):
                firstreturn_eff_rec4 = f"Forehand makes up {FirstReturn_FH_prop:.0%} of First Serve Returns but its effectiveness isn't good."
            elif ((FirstReturn_BH_Eff < 50) & ((FirstReturn_Slice_Eff -FirstReturn_BH_Eff) > 15 )  & (FirstReturn_BH_prop > 0.2) ):
                firstreturn_eff_rec4 = "Backhand return is ineffective. Consider using slice more in matches and working on your Backhand return in training."
            elif ((FirstReturn_BH_Eff < 50) & (FirstReturn_BH_prop > 0.3) ):
                firstreturn_eff_rec4 = "Backhand return has a low effectiveness score."
            else:
                firstreturn_eff_rec4 = ""
            firstreturn_eff_rec5 = "Your overall aim when returning 1st serve is to neutralise the serve, and get to a rally situation. Use this mindest when facing 1st serves to reduce number of points lost."
            # firstreturn_eff_rec5 = "Review the First Return Video.  How can you reduce the points you lose? How can you stop him hurting you?"# add in & flip the piece about the shots & mistake rate
    # 		else loss rate is under control,		
            first_return_home1 = "The low 1st Return Effectiveness is score is driven by a high loss rate.  See more details in the Return section."
        else:
            firstreturn_eff_rec2 = "Loss rate is under control."
    # 			if return rate < 10% off, can enhance	
            if firstreturn_rate_diff > 0.1:
                firstreturn_eff_rec3 = f"Returning against First Serve, you only get to play the return+1 in {firstreturn_rate*100:.0%} of points: This is {firstreturn_rate_diff*100: .0f}pts worse than average. Look to improve the quality of your return so you get further into the game."
            elif firstreturn_rate_diff < -0.1:
                firstreturn_eff_rec3 = f"Returning against First Serve, you manage to play the return+1 in {firstreturn_rate*100:.0%} of points. This is {firstreturn_rate_diff*-100: .0f}pts better than average. You are doing a great job of neutralising the serve."
            else:
                firstreturn_eff_rec3 = f"When facing the 1st Serve, you play the return+1 in {firstreturn_rate: .0%} of points, which is in line with average."
    # 			if mistakes of loss >50%, flag but with cavaet of number of lost points	
            #if firstreturn_lossmistake_rate >= 0.5:
                # firstreturn_eff_rec4 = f"Mistakes make up{firstreturn_lossmistake_rate: .0%} of lost points. Reduce this error rate.  Note this is based on{firstreturn_lost_freq: .0f} lost points."
                # #add to this line where can put in about the shot effectiveness.
                # first_return_home1 = "The low 1st Return Effectiveness is score is driven by a high mistake rate.  See more details in the Return section."
            #else:
            #jjo review 240720
            firstreturn_eff_rec4 = f"There is however a significant difference between won & lost points on Return.  Need to look to reduce this difference."
                #add to this line where can put in shot effectiveness.
            first_return_home1 = "The low 1st Return Effectiveness is score is driven by a high difference between lost & won points.  See more details in the Return section."
                # 			Review rally & win rate and see how can grow these values	
            firstreturn_eff_rec5 = "Returning 1st serves is hard.  But your aim is get beyond the return+1.  Make your opponent work for their points on their 1st serve.  Make it uncomfortable for them and this will give you more points."
            # firstreturn_eff_rec5 = "Review First Return video and see how can replicate more rally and win points."
    #         first_return_home1 = "The low 1st Return Effectiveness is score is driven by a high mistake rate.  See more details in the Return section."


                    
    # 50-67	Medium - higher - losing but its close.  Opponent only has a slight advantage here.			
    elif firstreturn_eff < 67:
        first_return_home1 = "Keep it up."
        if (firstreturn_lost / firstreturns_freq) - (firstreturn_won /firstreturns_freq) > 0.1:
            firstreturn_eff_rec = "First Return effectiveness is Medium - but on a higher end. Losing but you win or get to rally on a high proportion of points."
        elif firstreturn_won >= firstreturn_lost:
            firstreturn_eff_rec = "First Return effectiveness is Medium - but on a higher end.  You are winning slightly more than the server."
        else:
            firstreturn_eff_rec = "First Return effectiveness is Medium - but on a higher end.  Losing on average but its close."
    # 0		should I factor in total points for & against		
    # 		if loss greater than win, getting towards parity but still positive for opponent - 		
        if firstreturn_lost >= 0.4:
            firstreturn_eff_rec2 = f"Loss rate on First Returns is{firstreturn_lost: .0%}.  Reduce this. 1st Serve win rate of 40% is highly correlated with winning."
            firstreturn_eff_rec3 = "Aim for now is to progress to Rally from First Return."
            firstreturn_eff_rec4 = "Ask yourself; How can you reduce your losses and progress more to Rally? From what shot can your opponent not hurt you? Can you play more of these?"
            # firstreturn_eff_rec4 = "Review First Return Video.  How can you reduce your losses and progress more to Rally?"
            firstreturn_eff_rec5 = ""
    # 			if loss rate >40%, high probability - try to get more to rally	
    # 				mistake / losses - nb points -caution
                    
    # 				else how can reduce lost points - review where won, why did you make a mistake, why could he attack you, how could you nullify it?
        else:
            if firstreturn_lost > firstreturn_won:
                firstreturn_eff_rec2 = "Loss rate is under control.  Good job."
            else:
                firstreturn_eff_rec2 = "Managing to win more than losing on 1st Return.  Great job"
            firstreturn_eff_rec3 = "To further strengthen the effectiveness, you need to grow the win & rally rates."
            firstreturn_eff_rec4 = "If you filmed this game, review the 1st Return points, especially where you won or got to rally.  What behaviours can you replicate?"
            # firstreturn_eff_rec4 = "Review the First Return Video, particularly the Rally & Won points.  What behaviours can you replicate?"
            firstreturn_eff_rec5 = ""
    # 		if winning more than losing		
    # 			how can reduce lost points - review where won, why did you make a mistake, why could he attack you, how could you nullify it?	
                    
    # >67	beyond parity - points won & lost including how are balancing out & are in your favour.  This is beyond expectation.			
    else:
        first_return_home1 = "Keep it up."
        firstreturn_eff_rec = "First Return Score is High. You are balancing out won & lost points, and how the points end are in your favour."
    # 		75% chance of winning if here		
        firstreturn_eff_rec2 = "This is beyond expectation. You have a 75% chance of winning with this strong a return game."
    # 		if win > loss, able to win more than you lose within return +1.  Bravo.		
        if firstreturn_won > firstreturn_lost:
            firstreturn_eff_rec3 = "You are winning more than losing within 'Return + 1'. Great job!"
            firstreturn_eff_rec4 = "If you recorded this game, review the First Return points to see how you play and replicate this behaviour."
            firstreturn_eff_rec5 = ""
    # 			Review video & see how play, to replicate this behaviour	
    # 		else - getting very close to parity here		
        else:
            firstreturn_eff_rec3 = "You are very close to winning as many points as you are losing on First Return."
            firstreturn_eff_rec4 = "If you filmed this match, review the First Return points to see how you neutralise the serve & where you are winning and see how you can replicate this."
            firstreturn_eff_rec5 = ""
    # 			Review video & see how play, to see where win & neutralise serve to see where can replicate.	
                    
    if firstreturn_eff < 50:
        firstreturn_eff_rating = "WorkOn"
    else :
        firstreturn_eff_rating = "OK"

    # 2nd Return			
    # 1 in3 typical to face - want to be trying to attack here - focus on win rather than loss			
    # RR - if low, flag, though might work out			
                
    # 30 pts need to look at this			
   
    if (type == "single") & (secondreturns_freq < 20): 
        secondreturn_eff_rec = f"There are only {secondreturns_freq: .0f} 2nd Return points. The data is volatile at this level.  Therefore no recommendations are shown."
        secondreturn_eff_rec2 = "However aggregated insights on 2nd Return Performance are generated in your Personal Summary once there are at least 30 points to assess."
        secondreturn_eff_rec3 = ""
        secondreturn_eff_rec4 = ""
        second_return_home1 = "There aren't enough points to assess 2nd Return points for this match.  Check out the insights in your Personal Summary."
    # 37	Losing quite clearly on average		
    elif secondreturn_eff <= 37:
        secondreturn_eff_rec = "Second Return Effectiveness is Low."
        second_return_home1 = "The Second Return is an opportunity to win points.  Your low score says this opportunity isn't being taken.  See the Return section to understand what is happening and what can be improved."
    # 	where loss rate is >30% highlight this as high		
        if secondreturn_lost > 0.3:
            secondreturn_eff_rec2 = f"Loss rate on 2nd Returns is High at{secondreturn_lost: .0%}.  Look to reduce this rate."
    # 		if win !> +10pts, not offsetting with wins	
            secondreturn_eff_rec3 = "Not offsetting lost points with enough won points.  Address this imbalance."
            # if (secondreturn_won - secondreturn_lost) < 0.1:
            #     secondreturn_eff_rec3 = "Not offsetting lost points with enough won points.  Address this imbalance."
            # else: 
            #     secondreturn_eff_rec3 = "Review the 2nd Return video to see where you lose.  What can you do to reduce this?"
    # 		address imbalance	
            
    # 	else, loss rate under control		
        else:
            secondreturn_eff_rec2 = f"Loss rate is under control at {secondreturn_lost: .0%}."
            secondreturn_eff_rec3 = "However you're not offsetting lost points with enough won points.  Address this imbalance."
            # if (secondreturn_won - secondreturn_lost) < 0.1:
            #     secondreturn_eff_rec3 = "However you're not offsetting lost points with enough won points.  Address this imbalance."
            # else: 
            #     secondreturn_eff_rec3 = ""
    # 		if win !> +10pts, not offsetting with wins	
        secondreturn_eff_rec4 = "Look to be more aggressive and create pressure with your return. This will draw errors."
        # secondreturn_eff_rec4 = "Review the 2nd Return video to see where you can win more points.  Where can you create pressure? Where did the opponent seem less comfortable when you returned?"
    # -1		where can you make more points?	
    # 37-50	Negative but only slightly		
    elif secondreturn_eff < 50:
    # 0		where lost >= won, getting towards parity but not winning more than losing	
        secondreturn_eff_rec = "Second Return Effectiveness is Medium - but on the low end."
        second_return_home1 = "The Second Return is an opportunity to win points.  The balance isn't sufficiently in your favour.  See the Return section to understand what is happening and what can be improved."
        if secondreturn_lost > secondreturn_won:
            secondreturn_eff_rec2 = f"Getting towards parity, but losing ({secondreturn_lost:.0%}) more than winning ({secondreturn_won:.0%})"
    # 		else won - lost is positive - but can be improved	
        else:
            secondreturn_eff_rec2 = f"You are winning more points ({secondreturn_won:.0%}) than you're losing ({secondreturn_lost:.0%})"
        # secondreturn_eff_rec3 = "Review the 2nd Return video.  How can you win more points?  How can you create more pressure?  Where was the opponent uncomfortable?"
        secondreturn_eff_rec3 = "If you recorded the match, review the 2nd Return points and ask the following."
        secondreturn_eff_rec4 = "How can you win more points?  How can you create more pressure?  Where was the opponent uncomfortable?"
        
                
    # 50-63	parity - balancing out points - getting towards rally		
    elif secondreturn_eff < 63:
        secondreturn_eff_rec = "Second Return Effectiveness is Medium - but on the high end."
        second_return_home1 = "The Second Return is an opportunity to win points.  Whilst you win some, the balance can be improved.  See the Return section to understand what is happening and what can be improved."
        if secondreturn_rally > 0.4:
            secondreturn_eff_rec2 = f"A high percentage ({secondreturn_eff_rec:.0%}) of points are progressing to Rally."
            secondreturn_eff_rec3 = "Aim here should be to attack and create pressure so opponent feels pressure to make 1st serves."
            secondreturn_eff_rec4 = "If you filmed this match, review the 2nd Return points and see how points which progressed to Rally could be converted to won points for you."
            # secondreturn_eff_rec4 = "Review the 2nd Return video and see how Rally points could be converted to won points for you."
    # 1		if rally more than 40%, review to see where can convert these points to a win - be more aggressive	
    # 		if win more than 40%, compliment, see where can maintain this whilst reducing losses	
        else:
            secondreturn_eff_rec2 = f"You are able to win a good percentage ({secondreturn_won:.0%}) of points on 2nd Return.  Great job!"
            secondreturn_eff_rec3 = "Make a note of the tactics you employed, so you can replicate this strategy going forwards."
            # secondreturn_eff_rec3 = "Review the 2nd Return video and see how you win, so you can replicate this strategy going forwards."
            secondreturn_eff_rec4 = ""
                
    # >=63	on average getting to rally and winning more		
    else:
    # 		have a winning balance - review to see how win to see how to replicate this	
        secondreturn_eff_rec = "Second Return Effectiveness is High."
        secondreturn_eff_rec2 = f"You are winning more points ({secondreturn_won:.0%}) than you're losing ({secondreturn_lost:.0%})"
        secondreturn_eff_rec3 = "Great job! You are creating pressure on the 2nd serve which will put pressure on the opponents 1st Serve."
        secondreturn_eff_rec4 = "Make a note of the tactics you employed, so you can replicate this strategy going forwards."
        # secondreturn_eff_rec4 = "Review the 2nd Return video to see how you do this, so you can replicate this strategy going forwards."
        second_return_home1 = "Keep it up."
        
    if (secondreturn_eff < 63) & (secondreturns_freq >= 30):
        secondreturn_eff_rating = "WorkOn"
    else :
        secondreturn_eff_rating = "OK"


    # how play on 1st serve return
    # FirstServeReturn_by_CriticalHL_OutcomeGen	Return_Critical_Lost Return_NonCritical_Lost
    #calc the non loss rate of critical and non critical 1st serve points
    crit_1streturn_loss = df[(df.Label_0 == "FirstServeReturn_by_CriticalHL_OutcomeGen") & (df.Label == "Return_Critical_Lost") & (df.variable == "Frequency")].sum(axis = 1).sum()
    crit_1streturn_nonlossRATE = crit_1streturn_loss / crit_1streturnHIT_freq

    noncrit_1streturn_loss = df[(df.Label_0 == "FirstServeReturn_by_CriticalHL_OutcomeGen") & (df.Label == "Return_NonCritical_Lost") & (df.variable == "Frequency")].sum(axis = 1).sum()
    noncrit_1streturn_nonlossRATE = noncrit_1streturn_loss / noncrit_1streturnHIT_freq

    #caution - if less than 15, nothing, 15 to 50 caution & define, else just define
    if crit_1streturnHIT_freq < 15:
        crit_1streturnHIT_rec = f"There are only{crit_1streturnHIT_freq: .0f} critical 1st return points.  The data is too volatile for recommendations.  Play more to get insights."
        crit_1streturnHIT_rec2 = ""
        crit_1streturnHIT_rec3 = ""
        crit_1streturnHIT_rec4 = ""
        crit_1streturnHIT_rec5 = ""
        crit_1streturnHIT_rating = "OK"
        first_return_home2 = ""
        
    elif crit_1streturnHIT_freq < 35:
        crit_1streturnHIT_rec = f"Note: this comparison of critical to non-critical performance is only based on{crit_1streturnHIT_freq: .0f} critical 1st return points.  This data could be volatile, but the tactics are still worth considering."
        crit_1streturnHIT_rec2 = "Critical points are points which could result in a break in serve (30-40), or are points that could lead to a breakpoint if won (30-30)."
        crit_1streturnHIT_rec3 = f"There are 3 outcomes from a Return.  Win, Loss, or progress to Rally. On return, we want to prevent the Server from winning quickly.  Our goal is therefore 'Non-Loss' - either we win or progress to rally."
        if crit_1streturn_nonlossRATE < noncrit_1streturn_nonlossRATE:
            crit_1streturnHIT_rec4 = f"Non Loss rate on Critical 1st Serves is{crit_1streturn_nonlossRATE: .0%} vs{noncrit_1streturn_nonlossRATE: .0%} on Non-critical points."
            crit_1streturnHIT_rec5 = "Review how you can neutralise the serve more and apply this, especially during these key points."
            crit_1streturnHIT_rating = "WorkOn"
            first_return_home2 = "When facing 1st returns, an initial aim should be to not lose.  This is especially true on critical points.  The non-loss rate on critical points is worse than non critical points.  See the Return section for more details. "
        else:
            crit_1streturnHIT_rec4 = f"Non Loss rate on Critical 1st Serves is{crit_1streturn_nonlossRATE: .0%} vs{noncrit_1streturn_nonlossRATE: .0%} on Non-critical points."
            crit_1streturnHIT_rec5 = "Good job. Review how you neutralised the serve and keep applying this, especially during these key points."
            crit_1streturnHIT_rating = "OK"
            first_return_home2 = "Keep it up."
    else:
        crit_1streturnHIT_rec = "Critical points are points which could result in a break in serve (30-40), or are points that could lead to a breakpoint if won (30-30)."
        crit_1streturnHIT_rec2 = f"There are 3 outcomes from a Return.  Win, Loss, or progress to Rally. On return, we want to prevent the Server from winning quickly.  Our goal is therefore 'Non-Loss' - either we win or progress to rally."
        crit_1streturnHIT_rec3 = ""
        if crit_1streturn_nonlossRATE < noncrit_1streturn_nonlossRATE:
            crit_1streturnHIT_rec4 = f"Non Loss rate on Critical 1st Serves is{crit_1streturn_nonlossRATE: .0%} vs{noncrit_1streturn_nonlossRATE: .0%} on Non-critical points."
            crit_1streturnHIT_rec5 = "Review how you can neutralise the serve more and apply this, especially during these key points."
            crit_1streturnHIT_rating = "WorkOn"
            first_return_home2 = "When facing 1st returns, an initial aim should be to not lose.  This is especially true on critical points.  The non-loss rate on critical points is worse than non critical points.  See the Return section for more details. "
        else:
            crit_1streturnHIT_rec4 = f"Non Loss rate on Critical 1st Serves is{crit_1streturn_nonlossRATE: .0%} vs{noncrit_1streturn_nonlossRATE: .0%} on Non-critical points."
            crit_1streturnHIT_rec5 = "Good job. Review how you neutralised the serve and keep applying this, especially during these key points."
            crit_1streturnHIT_rating = "OK"
            first_return_home2 = "Keep it up."
            
    # extend / change logic for breakpoints



    #caution
    if rally_points <30 :
        rally_eff_caution = f"There are only {rally_points:.0f} rally points to analyse - the recommendations can therefore be volatile.  Note this highlights that serve and return points are the key focus areas. Review this section in your Personal Summary to understand the general trends in your rally game."
    else :
        rally_eff_caution = ""
    # Rally recommendations
    # 45 - medium low & win balance is X % & risk balance is X		
    if rally_eff < 26:
        rally_eff_rec = f"Rally Score is Low, Rally Win Balance is {rally_win_balance2}."#, Rally Risk Balance is {rally_risk_balance2}."
        rally_eff_action = "WorkOn"
        rally_eff_home1 = "When points progress to Rally, (further than Serve + 1 or Return +1), you have a losing balance.  See details in the Rally section about how to improve."
    elif rally_eff < 46:
        rally_eff_rec = f"Rally Score is Medium but on a Low end; Rally Win Balance is {rally_win_balance2}."#, Rally Risk Balance is {rally_risk_balance2}."
        rally_eff_action = "WorkOn"
        rally_eff_home1 = "When points progress to Rally, (further than Serve + 1 or Return +1), either the winning balance is around 0 or rally lengths are very high.  See details in the Rally section about how to improve."
    elif rally_eff < 60:
        rally_eff_rec = f"Rally Score is Medium but on the High End, Rally Win Balance is {rally_win_balance2}."#, Rally Risk Balance is {rally_risk_balance2}."
        rally_eff_action = "OK"
        rally_eff_home1 = "You have a reasonable winning margin when points progress to Rally. Keep it up."
    else:
        rally_eff_rec = f"Rally Score is High, Rally Win Balance is {rally_win_balance2}."#, Rally Risk Balance is {rally_risk_balance2}."
        rally_eff_action = "OK"
        rally_eff_home1 = "You are dominating the points when the points get to rally.  Keep it up."
    # explain what the win balance risk balance means in insights		
            
    # rally - score lower because of this.		
    if rally_prop >= 0.5:
        rally_eff_rec2 = f"When points are progressing to rally, they are on average {rally_length*2 + 4:.1f} shots long - including the serve & return part."
        #length of rally is lowering this score - conserve energy
        rally_eff_rec3 = "The length of the rallies is hurting you. Shorter rallies will conserve your energy."
        rally_eff_rec4 = "Where you can end them sooner.  How can you use space better? Can you use more variation to hurt the opponent?"
        # rally_eff_rec3 = "Review points and see where you can end them sooner.  How can you use space better? Can you use more variation to hurt the opponent?"
        # 	see where can end point sooner	
    # 		variation - better use of space - how hurt him
        rally_eff_rec5 = "NB: Long rallies until they make a mistake is a valid tactic but it costs lots of energy.  Hence recommendations to determine the point more."
        rally_eff_action2 = "WorkOn"
    # 		nb long rallies are a valid tactic but cost you lots of energy - hence recommendations to look to determine more
            
    else :
        rally_eff_rec2 = ""
        rally_eff_rec3 = ""
        rally_eff_rec4 = ""
        rally_eff_rec5 = ""
        rally_eff_action2 = "OK"
    # If Determine > 0, determine more than winners hit - replicate		
    # if rally_determine > 0:
    #     rally_eff_rec6 = f"In rallys, the 'Winners & Forced Errors Balance' is in your favour ({rally_determine2}). Great job! Keep it up!"
    #     rally_eff_rec7 = "Review video to see how you did this so you can replicate this behaviour."
    #     rally_eff_action3 = "OK"
    # # 	if less or equal - how can you stop them hitting winners against you	
    # else:
    #     rally_eff_rec6 = f"In rallys, the 'Winners & Forced Errors Balance' is NOT in your favour ({rally_determine2})."
    #     rally_eff_rec7 = "Review video to see how you were able to determine points so you can replicate this behaviour, and look where points were determined against you.  How can you stop the opponent hurting you?"
    #     rally_eff_action3 = "WorkOn"
    # # If Mistakes > 0 , getting more mistakes from them - replicate		
    # if rally_mistakes > 0 :
    #     rally_eff_rec8 = f"In rallys, the 'Unforced Error Balance' is in your favour ({rally_mistakes2})."
    #     rally_eff_rec9 = "You are making less unforced errors in the rally than your opponent."
    #     rally_eff_action4 = "OK"
    # else: 
        # rally_eff_rec8 = f"In rallys, the 'Unforced Error Balance' is NOT in your favour ({rally_mistakes2})."
        # rally_eff_rec9 = "Review the video and see where you make unforced errors and see where can cut these out."
        # rally_eff_action4 = "WorkOn"
    # 	if less or equal, review where making errors and how can reduce	
    # 		use this as both - 50% rally is an additional

    rally_eff_rec10 = f"Shot Mix & Effectiveness is based on {All_shots:.0f} shots.  Note where frequencies are low, data can be volatile."



    if FH_ratio >= 1.5:
        if FH_eff > 46:
            rally_eff_rec11 = f"Forehand shots are played {FH_ratio-1:.0%} more than Backhand & Slice combined, and its effectiveness is good.  This is great use of the Forehand as the Sword to win points."
            rally_eff_action5 = "OK"
            rally_eff_home2 = "Keep it up."
        else:
            rally_eff_rec11 = f"Forehand shots are played {FH_ratio-1:.0%} more than Backhand & Slice combined, however Forehand effectiveness is NOT good.  The intention is good to try to bring more Forehands in, but it is not effective.  Review to see where can use optimally to create pressure."
            rally_eff_action5 = "WorkOn"
            rally_eff_home2 = ""
    elif FH_ratio < 1:
        if ((BH_eff > 70) & (Slice_eff > 50)):
            rally_eff_rec11 = f"Forehand shots are played {1-FH_ratio:.0%} less than Backhand & Slice combined. Typically the Forehand is the Sword used to win points.  However your Backhand & Slice are effective at winning points.  Look where you can replicate this."
            rally_eff_action5 = "OK"
            rally_eff_home2 = "Keep it up."
        elif ((BH_prop >= 0.25) & (BH_eff > 70)):
            rally_eff_rec11 = f"Forehand shots are played {1-FH_ratio:.0%} less than Backhand & Slice combined. Typically the Forehand is the Sword used to win points.  However your Backhand is effective at winning points. Look at where you can replicate this."
            rally_eff_action5 = "OK"
            rally_eff_home2 = "Keep it up."
        elif ((Slice_prop >= 0.25) & (Slice_eff > 50)):
            rally_eff_rec11 = f"Forehand shots are played {1-FH_ratio:.0%} less than Backhand & Slice combined. Typically the Forehand is the Sword used to win points.  However your Slice is effective at winning points. Look at where you can replicate this."
            rally_eff_action5 = "OK"
            rally_eff_home2 = "Keep it up."
        elif FH_eff > 46:
            rally_eff_rec11 = f"Forehand shots are played {1-FH_ratio:.0%} less than Backhand & Slice combined, but its effectiveness is good.  Look at where you can bring in the Forehand more to create pressure."
            rally_eff_action5 = "WorkOn"
            rally_eff_home2 = ""
        else:
            rally_eff_rec11 = f"Forehand shots are played {1-FH_ratio:.0%} less than Backhand & Slice combined, and its effectiveness is NOT good, and is not offset by Backhand or Slice.  Look at where you can bring in the Forehand more & look to create more pressure with it."
            rally_eff_action5 = "WorkOn"
            rally_eff_home2 = ""
    else:
        if ((BH_prop >= 0.2) & (BH_eff > 56) & (FH_eff > 46) ):
            rally_eff_rec11 = f"You play slightly more Forehands than Backhand & Slice combined, and the effectiveness of your Forehand and Backhand is good.  Keep it up."
            rally_eff_action5 = "OK"
            rally_eff_home2 = "Keep it up."
        elif ((Slice_prop >= 0.2) & (Slice_eff <= 50) & (FH_eff > 46) ):
            rally_eff_rec11 = f"You play slightly more Forehands than Backhand & Slice combined, and the effectiveness of your Forehand & Slice is good.  Work on your Backhand as your shield to reduce where you lose points and look to bring in your Forehand more."
            rally_eff_action5 = "WorkOn"
            rally_eff_home2 = ""
        elif ((BH_prop >= 0.2) & (BH_eff <= 56) & (FH_eff > 46) ):
            rally_eff_rec11 = f"You play slightly more Forehands than Backhand & Slice combined, and the effectiveness of your Forehand is good but Backhand is low.  Work on your Backhand as your shield to reduce where you lose points and look to bring in your Forehand more."
            rally_eff_action5 = "WorkOn"
            rally_eff_home2 = ""
        elif ((Slice_prop >= 0.2) & (Slice_eff <= 50) & (FH_eff > 46) ):
            rally_eff_rec11 = f"You play slightly more Forehands than Backhand & Slice combined, and the effectiveness of your Forehand is good but Slice is low.  Work on your Backhand as your shield to reduce where you lose points and look to bring in your Forehand more."
            rally_eff_action5 = "WorkOn"
            rally_eff_home2 = ""
        else:
            if ((BH_eff > 70) & (Slice_eff < 50) & (BH_Slice_ratio < 1)) :
                rally_eff_rec11 = f"You play slightly more Forehands than Backhand & Slice combined, however the effectiveness of your Forehand is NOT good. Forehand should be the sword with which you win points. Look to see how you can create more pressure with this.  Also your Backhand is effective but used less than your Slice.  Trust it more."
                rally_eff_action5 = "WorkOn"
                rally_eff_home2 = ""
            else:
                rally_eff_rec11 = f"You play slightly more Forehands than Backhand & Slice combined, however the effectiveness of your Forehand is NOT good. Forehand should be the sword with which you win points. Look to see how you can create more pressure with this."
                rally_eff_action5 = "WorkOn"
                rally_eff_home2 = ""




    #note to later self - colour is defined in line - if blank then white
    recs_serve_data = [
        ["1st Serve Rate",won_serve_projected_comment, won_serve_projected_rating,2 ,firstserve_eff_home5 ],
        ["1st Serve Rate FollowUp",won_serve_projected_comment2, won_serve_projected_rating,2 ,"" ],
        ["First Serve Effectivness", firstserve_eff_rec, firstserve_eff_rating, 1, firstserve_eff_home1 ],
        ["First Serve Eff2", firstserve_eff_rec2, firstserve_eff_rating, 1, ""],
        ["First Serve Eff3", firstserve_eff_rec3, firstserve_eff_rating, 1, ""],
        ["First Serve Eff4", firstserve_eff_rec4, firstserve_eff_rating, 1, ""],
        ["Definiing critical pts", define_criticalpts_rec, define_criticalpts_rating, 4, ""],
        ["Critical 1st Serve Rate", critical_1stserve_rec,critical_1stserve_rating, 5, firstserve_eff_home2],
        ["Breakpt 1st Serve Rate", break_1strate_rec, break_1strate_ranking ,5, firstserve_eff_home3 ],
        ["Pre Break pt 1st Serve Rate",crit_nonbreak_1strate_rec ,crit_nonbreak_1strate_rating ,5, firstserve_eff_home4 ],
        ["Second Serve Effectivness", secondserve_eff_rec,secondserve_eff_rating, 3, secondserve_eff_home1],
        # ["Second Serve Caution", secondserve_eff_caution,secondserve_eff_rating, 3,secondserve_eff_caution ], 
        ["Second Serve Eff2", secondserve_eff_rec2, secondserve_eff_rating, 3, ""],
        ["Second Serve Eff3", secondserve_eff_rec3, secondserve_eff_rating, 3, ""],
        ["Second Serve Eff4", secondserve_eff_rec4, secondserve_eff_rating, 3, ""],
        ["Second Serve Eff5", secondserve_eff_rec5, secondserve_eff_rating, 3, ""]
        
    ]
    recs_serve = pd.DataFrame(recs_serve_data, columns = ["Label", "Recommendation", "Focus", "InherentRanking", "HomeScreenRec"])


    recs_serve_action = recs_serve[(recs_serve.Focus == "WorkOn") & (recs_serve.Recommendation != "")].sort_values("InherentRanking")
    recs_serve_action_ls = list(recs_serve_action.Recommendation)
    recs_serve_action_out = ""
    for i in range(len(recs_serve_action_ls)):
        recs_serve_action_out = recs_serve_action_out + recs_serve_action_ls[i] + nl + nl

    recs_serve_NOaction = recs_serve[(recs_serve.Focus == "OK")  & (recs_serve.Recommendation != "") ].sort_values("InherentRanking")
    recs_serve_NOaction_ls = list(recs_serve_NOaction.Recommendation)
    recs_serve_NOaction_out = ""
    for i in range(len(recs_serve_NOaction_ls)):
        recs_serve_NOaction_out = recs_serve_NOaction_out + recs_serve_NOaction_ls[i] + nl
        
    recs_serve_out_fin = recs_serve_action_out + recs_serve_NOaction_out


    recs_serve_all = recs_serve[(recs_serve.Recommendation != "")].sort_values("InherentRanking")
    recs_serve_all_recls = list(recs_serve_all.Recommendation)
    recs_serve_all_rankingls = list(recs_serve_all.InherentRanking)
    recs_serve_all_out = ""
    for i in range(len(recs_serve_all_recls)):
    #     print(recs_serve_all_rankingls[i])
        if recs_serve_all_rankingls[i] == max(recs_serve_all_rankingls):
            recs_serve_all_out = recs_serve_all_out + recs_serve_all_recls[i] + nl
        elif recs_serve_all_rankingls[i] == recs_serve_all_rankingls[i+1]:
            recs_serve_all_out = recs_serve_all_out + recs_serve_all_recls[i] + nl
        else:
            recs_serve_all_out = recs_serve_all_out + recs_serve_all_recls[i] + nl + nl

    recs_return_data = [    
        ["First Return Caution", firstreturn_eff_rec_caution, firstreturn_eff_rating , 1, ""], # same rating as these things are linked
        ["First Return Effectivness", firstreturn_eff_rec, firstreturn_eff_rating, 1, first_return_home1],
        ["First Return Eff2", firstreturn_eff_rec2, firstreturn_eff_rating ,1, ""],
        ["First Return Eff3", firstreturn_eff_rec3, firstreturn_eff_rating, 1, ""],
        ["First Return Eff4", firstreturn_eff_rec4, firstreturn_eff_rating, 1, ""],
        ["First Return Eff5", firstreturn_eff_rec5, firstreturn_eff_rating, 1, ""],
        # ["Second Return Caution", secondreturn_eff_rec_caution, secondreturn_eff_rating , 3, ""], # same rating as these things are linked
        ["Second Return Effectivness", secondreturn_eff_rec, secondreturn_eff_rating, 3, second_return_home1],
        ["Second Return Eff2", secondreturn_eff_rec2, secondreturn_eff_rating, 3, ""],
        ["Second Return Eff3", secondreturn_eff_rec3, secondreturn_eff_rating, 3, ""],
        ["Second Return Eff4", secondreturn_eff_rec4, secondreturn_eff_rating, 3, ""],
        ["Critical 1st Return Performance", crit_1streturnHIT_rec,crit_1streturnHIT_rating ,2 , first_return_home2],
        ["Critical 1st Return2", crit_1streturnHIT_rec2,crit_1streturnHIT_rating ,2 , ""],
        ["Critical 1st Return3", crit_1streturnHIT_rec3,crit_1streturnHIT_rating ,2 , ""],
        ["Critical 1st Return4", crit_1streturnHIT_rec4,crit_1streturnHIT_rating ,2 , ""],
        ["Critical 1st Return5", crit_1streturnHIT_rec5,crit_1streturnHIT_rating ,2 , ""],
        
    ]
    recs_return = pd.DataFrame(recs_return_data, columns = ["Label", "Recommendation", "Focus", "InherentRanking", "HomeScreenRec"])
    recs_return_action = recs_return[(recs_return.Focus == "WorkOn") & (recs_return.Recommendation != "")].sort_values("InherentRanking")

    recs_return_action_ls = list(recs_return_action.Recommendation)
    recs_return_action_out = ""
    for i in range(len(recs_return_action_ls)):
        recs_return_action_out = recs_return_action_out + recs_return_action_ls[i] + nl + nl

    recs_return_NOaction = recs_return[(recs_return.Focus == "OK") & (recs_return.Recommendation != "")].sort_values("InherentRanking")

    recs_return_NOaction = recs_return[(recs_return.Focus == "OK")  & (recs_return.Recommendation != "") ].sort_values("InherentRanking")
    recs_return_NOaction_ls = list(recs_return_NOaction.Recommendation)
    recs_return_NOaction_out = ""
    for i in range(len(recs_return_NOaction_ls)):
        recs_return_NOaction_out = recs_return_NOaction_out + recs_return_NOaction_ls[i] + nl
        
    recs_return_out_fin = recs_return_action_out + recs_return_NOaction_out


    recs_return_all = recs_return[(recs_return.Recommendation != "")].sort_values("InherentRanking")
    recs_return_all_recls = list(recs_return_all.Recommendation)
    recs_return_all_rankingls = list(recs_return_all.InherentRanking)
    recs_return_all_out = ""
    for i in range(len(recs_return_all_recls)):
        #     print(recs_serve_all_rankingls[i])
        if recs_return_all_rankingls[i] == max(recs_return_all_rankingls):
            recs_return_all_out = recs_return_all_out + recs_return_all_recls[i] + nl
        elif recs_return_all_rankingls[i] == recs_return_all_rankingls[i+1]:
            recs_return_all_out = recs_return_all_out + recs_return_all_recls[i] + nl
        else:
            recs_return_all_out = recs_return_all_out + recs_return_all_recls[i] + nl + nl



    recs_rally_data = [    
    #     ["Rally Caution", rally_eff_caution, "" , 1, ""], # same rating as these things are linked
        ["Rally Effectiveness", rally_eff_rec, rally_eff_action, 2, rally_eff_home1 ],
        ["Rally Effectiveness", rally_eff_caution, "" , 2, ""], # same rating as these things are linked
        ["Rally Length", rally_eff_rec2, rally_eff_action2 ,3, ""],
        ["Rally Length", rally_eff_rec3, "", 4, ""],
        ["Rally Length", rally_eff_rec4, "", 5, ""],
        ["Rally Length", rally_eff_rec5, "", 6, ""],
        #["Rally Determination", rally_eff_rec6, rally_eff_action3, 7, ""],
        #["Rally Determination", rally_eff_rec7, "", 8, ""],
        #["Rally Error Performance", rally_eff_rec8, rally_eff_action4, 9, ""],
        #["Rally Error Performance", rally_eff_rec9, "", 10, ""],
        ["Shots Proportion & Effectiveness", rally_eff_rec10, "", 11, ""],
        ["Shots Selection & Effectiveness", rally_eff_rec11, rally_eff_action5, 12, rally_eff_home2],
        
    ]
    recs_rally = pd.DataFrame(recs_rally_data, columns = ["Label", "Recommendation", "Focus", "InherentRanking", "HomeScreenRec"])
    recs_rally_action = recs_rally[(recs_rally.Recommendation != "")].sort_values("InherentRanking").reset_index()

    recs_rally_action_ls = list(recs_rally_action.Recommendation)
    recs_rally_action_out = ""
    for i in range(len(recs_rally_action_ls)):
        if i == (len(recs_rally_action_ls) -1):
            recs_rally_action_out = recs_rally_action_out + recs_rally_action_ls[i] + nl
        elif recs_rally_action.Label[i] == recs_rally_action.Label[i+1]:
            recs_rally_action_out = recs_rally_action_out + recs_rally_action_ls[i] + nl
        else:
            recs_rally_action_out = recs_rally_action_out + recs_rally_action_ls[i] + nl + nl

    recs_rally_out_fin = recs_rally_action_out



    #create the home screen recs

    #take serve 1sts actions
    home_rec_serve = list(recs_serve_action[(recs_serve_action.HomeScreenRec != "") & (recs_serve_action.InherentRanking <= 2)].HomeScreenRec)
    home_rec_return = list(recs_return_action[(recs_return_action.HomeScreenRec != "") & (recs_return_action.InherentRanking == 1)].HomeScreenRec)
    home_rec_firsts = home_rec_serve + home_rec_return

    if recs_rally_action[(recs_rally_action.InherentRanking == 2)].Focus.values[0] == "WorkOn":
        home_rec_rallies = ["In Rallies, (beyond Serve + 1 and Return + 1), you are losing more than winning.  See the Rallies section for details on how to address."]
    else:
        home_rec_rallies = list()
        
    home_rec_serve2 = list(recs_serve_action[(recs_serve_action.HomeScreenRec != "") & (recs_serve_action.InherentRanking > 2)].HomeScreenRec)
    home_rec_return2 = list(recs_return_action[(recs_return_action.HomeScreenRec != "") & (recs_return_action.InherentRanking != 1)].HomeScreenRec)

    if recs_rally_action[(recs_rally_action.InherentRanking == 12)].Focus.values[0] == "WorkOn":
        home_rec_rallies2 = ["Your shot selection & the effectiveness of those shots needs improvement.  Details and specific recommendations can be found in the Rally section."]
    else:
        home_rec_rallies2 = list()

    #check lengths - adding in focus on serve & return
    if home_insights2 == "":
        home_rec_workons = home_rec_serve + home_rec_return + home_rec_rallies + home_rec_serve2 + home_rec_return2 + home_rec_rallies2 

        if len(home_rec_workons) >3:
            home_rec_workons_fin = home_rec_workons[:3] #upto & included 3rd
        else:
            home_rec_workons_fin = home_rec_workons
    else:
        home_rec_focus = [f"{under4_per:.0%} of points were finished within 4 shots.  Focus primarily on improving Serve & Return."]
        home_rec_workons = home_rec_focus + home_rec_serve + home_rec_return + home_rec_rallies + home_rec_serve2 + home_rec_return2 + home_rec_rallies2 
        
        if len(home_rec_workons) >4:
            home_rec_workons_fin = home_rec_workons[:4] #upto & included 3rd
        else:
            home_rec_workons_fin = home_rec_workons
        
    #create the keep it ups
    home_rec_serve_kiu = list(recs_serve_NOaction[(recs_serve_NOaction.HomeScreenRec == "Keep it up.") ].Label)
    home_rec_return_kiu = list(recs_return_NOaction[(recs_return_NOaction.HomeScreenRec == "Keep it up.")].Label)
    recs_rally_action_kiu = list(recs_rally_action[(recs_rally_action.HomeScreenRec == "Keep it up.") ].Label)

    recs_kiu = home_rec_serve_kiu + home_rec_return_kiu + recs_rally_action_kiu
    if len(recs_kiu)>3:
        recs_kiu_fin = recs_kiu[:3]
    else:
        recs_kiu_fin = recs_kiu

    if len(recs_kiu_fin) == 0:
        recs_kiu_fin_out = ""
    else:
        recs_kiu_fin_out = "The following are areas of strength.  Check out the details in the relevant areas to see why and how to maintain these."
    for i in range(len(recs_kiu_fin)):
        recs_kiu_fin_out = recs_kiu_fin_out + nl
        recs_kiu_fin_out = recs_kiu_fin_out + recs_kiu_fin[i]
        
    home_rec_fin_out = ""
    for i in range(len(home_rec_workons_fin)):
        home_rec_fin_out += home_rec_workons_fin[i] + nl + nl
        
    home_rec_fin_out += recs_kiu_fin_out



    nl = "\n"
    home_insights = str(f"This analysis is based on {ptsplayed:.0f} points, and is personalised based on your Effectiveness Scores (defined below).{nl}{nl}"
                        f"{home_insights1}{nl}"
        f"{home_insights2}{nl}{nl}"
                    f"{home_insights3}{nl}"
                        f"{home_insights3a}{nl}{nl}"
                    f"{home_insights4}{nl}"
                    f"{home_insights4a}{nl}{nl}"
                    f"{home_insights5}{nl}"
                    f"{home_insights5a}{nl}{nl}"
                    f"{rally_length_totalav_ins}{nl}{nl}"
                    )
    #                    f"{HR_Hard_ins}{nl}"
    #                    f"{HR_Max_Ins}{nl}"
    #                    f"{HR_def_ins}{nl}{nl}"
    #                    f"__Effectiveness Defined__ {nl}"
    #                    f"_Serve here is considered as 2 shots, Serve & Serve+1.  You may not win directly from your serve, but it should set you up to be able to create maximum pressure._ {nl} "
    #                    f"_Effectiveness is based on the outcome.  Do you win or lose within these first 2 shots, or does it continue to rally?_ {nl} "
    #                    f"_How the point ends is factored into this too - winning with a winner is more highly weighted than winning via an unforced error._ {nl} "
    #                    f"_This same logic is applied to Return - ie the Return is evalulated as the Return & Return+1._ {nl} "
    #                    f"_Each of the scores are scaled differently based on expectations and averages of data collected._")


    json_out = {
    "totalPointsWonOrLost": {
        "pointWonWithPointsLostPercentage": won /(won+lost) *100,
        "winnersOfPointsWonPercentage": {
            "value": 0,#(Won_thru_Winner + Won_thru_Forced) /won * 100, # this is how much of my own points do I determine
            "winners": Won_thru_Winner / (won + lost) *100, # these are in the chart so add to 100%
            "forcedError": 0,#Won_thru_Forced / (won + lost) *100,
            "mistake": 0,#Won_thru_Mistake / (won + lost) *100
        },
        "unforcedErrorsOfPointsLostPercentage": {
            "value": Lost_thru_Mistake / lost *100, # % lost because of your own mistakes
            "winners": 0,#Lost_thru_Winner / (won + lost) *100, # these are in the chart so add to 100%
            "forcedError": 0,#Lost_thru_Forced / (won + lost) *100 ,
            "mistake": Lost_thru_Mistake / (won + lost) *100
        }
    },
    "recommendations": [
        home_insights,
        home_rec_fin_out
    #       "uuid-456"
    ],
    "scores": {
        "serveScore": {
            "points": totalserve_eff,#round((serve_eff_freq +3)/ (8/100),0),#making all positive (add min & spread over range)
            "firstServePoints": firstserve_eff,
            "secondServePoints" : secondserve_eff,
            "recommendations": [
                serve_ins_out_fin,
                recs_serve_all_out #changed230415
            ],
            "breakdown": {
                "detailedPoints": {
                "lost": {
                    "lostToWinnerFromOpponent": 5,
                    "lostToForcedWinner": 7,
                    "lostToMistakeByYou": 10
                },
                "rallyContinued": 1,
                "won": {
                    "wonWithWiner": 10,
                    "wonWithForced": 5,
                    "wonWithMistakeOpponent": 5
                }
                },
                "first": {
                "totalPercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#firstServe_total_lostwinner / firstserves_freq * 100,
                        "lostToForcedWinner": 0,#firstServe_total_lostforced / firstserves_freq * 100,
                        "lostToMistakeByYou": firstServe_total_lostmistake / firstserves_freq * 100
                    },
                    "rallyContinued": firstServe_total_rally / firstserves_freq * 100,
                    "won": {
                        "wonWithWiner": firstServe_total_wonwinner / firstserves_freq * 100,
                        "wonWithForced": 0,#firstServe_total_wonforced / firstserves_freq * 100,
                        "wonWithMistakeOpponent": 0,#firstServe_total_wonmistake / firstserves_freq * 100
                    }
                },
                "deucePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#deuce_firstServe_total_lostwinner / deuce_firstserves_freq * 100,
                        "lostToForcedWinner": 0,#deuce_firstServe_total_lostforced / deuce_firstserves_freq * 100,
                        "lostToMistakeByYou": deuce_firstServe_total_lostmistake / (deuce_firstServe_total_lostmistake + deuce_firstServe_total_rally + deuce_firstServe_total_wonwinner ) * 100
                    },
                    "rallyContinued": deuce_firstServe_total_rally / (deuce_firstServe_total_lostmistake + deuce_firstServe_total_rally + deuce_firstServe_total_wonwinner ) * 100,
                    "won": {
                        "wonWithWiner": deuce_firstServe_total_wonwinner / (deuce_firstServe_total_lostmistake + deuce_firstServe_total_rally + deuce_firstServe_total_wonwinner ) * 100,
                        "wonWithForced": 0,#deuce_firstServe_total_wonforced / deuce_firstserves_freq * 100,
                        "wonWithMistakeOpponent": 0,#deuce_firstServe_total_wonmistake / deuce_firstserves_freq * 100
                    }
                },
                "advantagePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#adv_firstServe_total_lostwinner / adv_firstserves_freq * 100,
                        "lostToForcedWinner": 0,#adv_firstServe_total_lostforced / adv_firstserves_freq * 100,
                        "lostToMistakeByYou": adv_firstServe_total_lostmistake / (adv_firstServe_total_lostmistake + adv_firstServe_total_rally + adv_firstServe_total_wonwinner) * 100
                    },
                    "rallyContinued": adv_firstServe_total_rally / (adv_firstServe_total_lostmistake + adv_firstServe_total_rally + adv_firstServe_total_wonwinner) * 100,
                    "won": {
                        "wonWithWiner": adv_firstServe_total_wonwinner / (adv_firstServe_total_lostmistake + adv_firstServe_total_rally + adv_firstServe_total_wonwinner) * 100,
                        "wonWithForced": 0,#adv_firstServe_total_wonforced / adv_firstserves_freq * 100,
                        "wonWithMistakeOpponent": 0,#adv_firstServe_total_wonmistake / adv_firstserves_freq * 100
                    }
                }
                },
                "second": {
                "totalPercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#secondServe_total_lostwinner / secondserves_freq * 100,
                        "lostToForcedWinner": 0,#secondServe_total_lostforced / secondserves_freq * 100,
                        "lostToMistakeByYou": secondServe_total_lostmistake / secondserves_freq * 100
                    },
                    "rallyContinued": secondServe_total_rally / secondserves_freq * 100,
                    "won": {
                        "wonWithWiner": secondServe_total_wonwinner / secondserves_freq * 100,
                        "wonWithForced": 0,#secondServe_total_wonforced / secondserves_freq * 100,
                        "wonWithMistakeOpponent": 0,#secondServe_total_wonmistake / secondserves_freq * 100
                    }
                },
                "deucePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#Deuce_secondServe_total_lostwinner / Deuce_secondserves_freq * 100,
                        "lostToForcedWinner": 0,#Deuce_secondServe_total_lostforced / Deuce_secondserves_freq * 100,
                        "lostToMistakeByYou": Deuce_secondServe_total_lostmistake / (Deuce_secondServe_total_lostmistake + Deuce_secondServe_total_rally + Deuce_secondServe_total_wonwinner ) * 100
                    },
                    "rallyContinued": Deuce_secondServe_total_rally /  (Deuce_secondServe_total_lostmistake + Deuce_secondServe_total_rally + Deuce_secondServe_total_wonwinner ) * 100,
                    "won": {
                        "wonWithWiner": Deuce_secondServe_total_wonwinner /  (Deuce_secondServe_total_lostmistake + Deuce_secondServe_total_rally + Deuce_secondServe_total_wonwinner ) * 100,
                        "wonWithForced": 0,#Deuce_secondServe_total_wonforced / Deuce_secondserves_freq * 100,
                        "wonWithMistakeOpponent": 0,#Deuce_secondServe_total_wonmistake / Deuce_secondserves_freq * 100
                    }
                },
                "advantagePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#Adv_secondServe_total_lostwinner / Adv_secondserves_freq * 100,
                        "lostToForcedWinner": 0,#Adv_secondServe_total_lostforced / Adv_secondserves_freq * 100,
                        "lostToMistakeByYou": Adv_secondServe_total_lostmistake / (Adv_secondServe_total_lostmistake + Adv_secondServe_total_rally + Adv_secondServe_total_wonwinner) * 100
                    },
                    "rallyContinued": Adv_secondServe_total_rally / (Adv_secondServe_total_lostmistake + Adv_secondServe_total_rally + Adv_secondServe_total_wonwinner) * 100,
                    "won": {
                        "wonWithWiner": Adv_secondServe_total_wonwinner / (Adv_secondServe_total_lostmistake + Adv_secondServe_total_rally + Adv_secondServe_total_wonwinner) * 100,
                        "wonWithForced": 0,#Adv_secondServe_total_wonforced / Adv_secondserves_freq * 100,
                        "wonWithMistakeOpponent": 0,#Adv_secondServe_total_wonmistake / Adv_secondserves_freq * 100
                    }
                }
                }
            },
            "rateAndEffectiveness": {
                "ratePercentage": firstserverate * 100,
                "lostPercentage": {
                "firstServe": firstserve_lost * 100,
                "secondServe": secondserve_lost * 100
                },
                "rallyPercentage": {
                "firstServe": firstserve_rally * 100,
                "secondServe": secondserve_rally * 100
                },
                "wonPercentage": {
                "firstServe": firstserve_won * 100,
                "secondServe": secondserve_won * 100
                }
            }
        },
        "returnScore": {
            "points": totalreturn_eff ,#round((return_eff_freq +3)/ (6/100),0),#making all positive (add min & spread over range),
            "firstServePoints": firstreturn_eff ,
            "secondServePoints" : secondreturn_eff ,
            "recommendations": [
                return_insights_out_fin,
                recs_return_all_out # recs_return_out_fin updated 230415
            ],
            "breakdown": {
                "detailedPoints": {
                "lost": {
                    "lostToWinnerFromOpponent": 5,
                    "lostToForcedWinner": 7,
                    "lostToMistakeByYou": 10
                },
                "rallyContinued": 1,
                "won": {
                    "wonWithWiner": 10,
                    "wonWithForced": 5,
                    "wonWithMistakeOpponent": 5
                }
                },
                "first": {
                "totalPercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#firstReturn_total_lostwinner / firstreturns_freq * 100,
                        "lostToForcedWinner": 0,#firstReturn_total_lostforced / firstreturns_freq * 100,
                        "lostToMistakeByYou": firstReturn_total_lostmistake / firstreturns_freq * 100
                    },
                    "rallyContinued": firstReturn_total_rally / firstreturns_freq * 100,
                    "won": {
                        "wonWithWiner": firstReturn_total_wonwinner / firstreturns_freq * 100,
                        "wonWithForced": 0,#firstReturn_total_wonforced / firstreturns_freq * 100,
                        "wonWithMistakeOpponent": 0,#firstReturn_total_wonmistake / firstreturns_freq * 100
                    }
                },
                "deucePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#deuce_firstReturn_total_lostwinner / deuce_firstreturns_freq * 100,
                        "lostToForcedWinner": 0,#deuce_firstReturn_total_lostforced / deuce_firstreturns_freq * 100,
                        "lostToMistakeByYou": deuce_firstReturn_total_lostmistake / (deuce_firstReturn_total_lostmistake + deuce_firstReturn_total_rally + deuce_firstReturn_total_wonwinner) * 100
                    },
                    "rallyContinued": deuce_firstReturn_total_rally / (deuce_firstReturn_total_lostmistake + deuce_firstReturn_total_rally + deuce_firstReturn_total_wonwinner) * 100,
                    "won": {
                        "wonWithWiner": deuce_firstReturn_total_wonwinner / (deuce_firstReturn_total_lostmistake + deuce_firstReturn_total_rally + deuce_firstReturn_total_wonwinner) * 100,
                        "wonWithForced": 0,#deuce_firstReturn_total_wonforced / deuce_firstreturns_freq * 100,
                        "wonWithMistakeOpponent": 0,#deuce_firstReturn_total_wonmistake / deuce_firstreturns_freq * 100
                    }
                },
                "advantagePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#adv_firstReturn_total_lostwinner / adv_firstreturns_freq * 100,
                        "lostToForcedWinner": 0,#adv_firstReturn_total_lostforced / adv_firstreturns_freq * 100,
                        "lostToMistakeByYou": adv_firstReturn_total_lostmistake / (adv_firstReturn_total_lostmistake + adv_firstReturn_total_rally + adv_firstReturn_total_wonwinner) * 100
                    },
                    "rallyContinued": adv_firstReturn_total_rally / (adv_firstReturn_total_lostmistake + adv_firstReturn_total_rally + adv_firstReturn_total_wonwinner)  * 100,
                    "won": {
                        "wonWithWiner": adv_firstReturn_total_wonwinner / (adv_firstReturn_total_lostmistake + adv_firstReturn_total_rally + adv_firstReturn_total_wonwinner)  * 100,
                        "wonWithForced": 0,#adv_firstReturn_total_wonforced / adv_firstreturns_freq * 100,
                        "wonWithMistakeOpponent": 0,#adv_firstReturn_total_wonmistake / adv_firstreturns_freq * 100
                    }
                }
                },
                "second": {
                "totalPercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#secondReturn_total_lostwinner / secondreturns_freq *100,
                        "lostToForcedWinner": 0,#secondReturn_total_lostforced / secondreturns_freq *100,
                        "lostToMistakeByYou": secondReturn_total_lostmistake / secondreturns_freq *100
                    },
                    "rallyContinued": secondReturn_total_rally / secondreturns_freq *100,
                    "won": {
                        "wonWithWiner": secondReturn_total_wonwinner / secondreturns_freq *100,
                        "wonWithForced": 0,#secondReturn_total_wonforced / secondreturns_freq *100,
                        "wonWithMistakeOpponent": 0,#secondReturn_total_wonmistake / secondreturns_freq *100
                    }
                },
                "deucePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#Deuce_secondReturn_total_lostwinner / Deuce_secondreturns_freq *100,
                        "lostToForcedWinner": 0,#Deuce_secondReturn_total_lostforced / Deuce_secondreturns_freq *100,
                        "lostToMistakeByYou": Deuce_secondReturn_total_lostmistake / (Deuce_secondReturn_total_lostmistake + Deuce_secondReturn_total_rally + Deuce_secondReturn_total_wonwinner) *100
                    },
                    "rallyContinued": Deuce_secondReturn_total_rally / (Deuce_secondReturn_total_lostmistake + Deuce_secondReturn_total_rally + Deuce_secondReturn_total_wonwinner) *100,
                    "won": {
                        "wonWithWiner": Deuce_secondReturn_total_wonwinner / (Deuce_secondReturn_total_lostmistake + Deuce_secondReturn_total_rally + Deuce_secondReturn_total_wonwinner) *100,
                        "wonWithForced": 0,#Deuce_secondReturn_total_wonforced / Deuce_secondreturns_freq *100,
                        "wonWithMistakeOpponent": 0,#Deuce_secondReturn_total_wonmistake / Deuce_secondreturns_freq *100
                    }
                },
                "advantagePercentage": {
                    "lost": {
                        "lostToWinnerFromOpponent": 0,#Adv_secondReturn_total_lostwinner / Adv_secondreturns_freq *100,
                        "lostToForcedWinner": 0,#Adv_secondReturn_total_lostforced / Adv_secondreturns_freq *100,
                        "lostToMistakeByYou": Adv_secondReturn_total_lostmistake / (Adv_secondReturn_total_lostmistake + Adv_secondReturn_total_rally + Adv_secondReturn_total_wonwinner) *100
                    },
                    "rallyContinued": Adv_secondReturn_total_rally / (Adv_secondReturn_total_lostmistake + Adv_secondReturn_total_rally + Adv_secondReturn_total_wonwinner) *100,
                    "won": {
                        "wonWithWiner": Adv_secondReturn_total_wonwinner / (Adv_secondReturn_total_lostmistake + Adv_secondReturn_total_rally + Adv_secondReturn_total_wonwinner) *100,
                        "wonWithForced": 0,#Adv_secondReturn_total_wonforced / Adv_secondreturns_freq *100,
                        "wonWithMistakeOpponent": 0,#Adv_secondReturn_total_wonmistake / Adv_secondreturns_freq *100
                    }
                }
                }
            },
            "rateAndEffectiveness": {
                
                "ratePercentage":  {"firstReturnrate" : firstreturn_rate ,#made_firstreturns_freq / firstreturns_freq * 100,
                                    "secondReturnrate" : secondreturn_rate ,#made_secondreturns_freq / secondreturns_freq * 100,
                                    "returnRateChange" : secondreturn_rate - firstreturn_rate# ((made_secondreturns_freq / secondreturns_freq) - (made_firstreturns_freq / firstreturns_freq)) *100 
                                    },
                "lostPercentage": {
                    "firstServe": firstreturn_lost * 100,
                    "secondServe": secondreturn_lost *100
                },
                "rallyPercentage": {
                    "firstServe": firstreturn_rally *100,
                    "secondServe": secondreturn_rally *100
                },
                "wonPercentage": {
                    "firstServe": firstreturn_won *100,
                    "secondServe": secondreturn_won *100
                }     
            }
        },
        "rallyScore": {
            "points": rally_eff ,#round((rally_eff_freq +1.5)/ (5.5/100),0),
            "recommendations": [
                rally_ins_out,
                recs_rally_out_fin
            ],
            "breakdown": {
                "detailedPoints": {
                "lost": {
                    "lostToWinnerFromOpponent": 5,
                    "lostToForcedWinner": 7,
                    "lostToMistakeByYou": 10
                },
                "rallyContinued": 1,
                "won": {
                    "wonWithWiner": 10,
                    "wonWithForced": 5,
                    "wonWithMistakeOpponent": 5
                }
                },
                "totalPercentage": {
                "lost": {
                    "lostToWinnerFromOpponent": 0,#rally_lostwinner / rally_freq *100,
                    "lostToForcedWinner": 0,#rally_lostforced / rally_freq *100,
                    "lostToMistakeByYou": rally_lostmistake / rally_freq *100
                },
                "rallyContinued": rally_rally / rally_freq *100,
                "won": {
                    "wonWithWiner": rally_wonwinner / rally_freq *100,
                    "wonWithForced": 0,#rally_wonforced / rally_freq *100,
                    "wonWithMistakeOpponent": 0,#rally_wonmistake / rally_freq *100
    
                }
                }
            },
            "rallyLengthsAndPerformancePercentage": {
                "serve": {
                "lost": serve_lost / (won + lost) *100,
                "won": serve_won / (won + lost) *100
                },
                "returnPerfomance": {
                "lost": return_lost / (won + lost) *100,
                "won": return_won / (won + lost) *100
                },
                "rally": {
                "lost": rally_324_lost / (won + lost) *100,
                "won": rally_324_won / (won + lost) *100
                },
                "longRally": {
                "lost": rally_5plus_lost / (won + lost) *100,
                "won": rally_5plus_won / (won + lost) *100
                }
            }
        }
    },
    "criticalPointsPerformanceTotal": {
        "firstServeRate": {
            "criticalPoints": crit_1stserve_freq, # this is the number of critical points
            "nonCriticalPoints": noncrit_1stserve_freq, # this is the number of critical points
            "criticalPercentage": crit_1stserve_freq / crit_totalserve_freq *100 ,
            "nonCriticalPercentage": noncrit_1stserve_freq / noncrit_totalserve_freq *100 ,
            "up": up_1st_serve
                    
        },
        "firstServeReturn": {
            "criticalPoints": crit_1streturnHIT_freq,
            "nonCriticalPoints": noncrit_1streturnHIT_freq,
            "criticalPercentage":  crit_1streturnMADE_freq / crit_1streturnHIT_freq *100,
            "nonCriticalPercentage": noncrit_1streturnMADE_freq / noncrit_1streturnHIT_freq*100,
            "up": up_1st_return # true
        },
        "secondServeNonLossRate": {
            "criticalPoints": crit_secondserve_freq,
            "nonCriticalPoints": noncrit_2ndserve_freq,
            "criticalPercentage": crit_2ndservenonloss_freq / crit_secondserve_freq *100,
            "nonCriticalPercentage": noncrit_2ndservenonloss_freq / noncrit_2ndserve_freq *100,
            "up": up_2nd_serve
        },
        "secondServeReturn": {
            "criticalPoints": crit_2ndreturnHIT_freq,
            "nonCriticalPoints": noncrit_2ndreturnHIT_freq,
            "criticalPercentage":  criticalPercentage2ndReturns,
            "nonCriticalPercentage": noncrit_2ndreturnMADE_freq / noncrit_2ndreturnHIT_freq*100,
            "up": up_2nd_return #false
        }
    }
    }

    # with open(f'{base}{baseplus}{who}_{when}_{type}.json', 'w') as outfile:
    #     json.dump(json_out, outfile, indent=4)
    # print("Anaysis Complete")
    return json_out