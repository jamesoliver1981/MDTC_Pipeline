import pandas as pd
import numpy as np


def create_stats_basis(d_in, shots):
    
    pts = d_in.drop("ServeFrom", axis=1).rename(columns={"ServeFrom2": "ServeFrom"})

    # these are the non shot points that need to add to the data
    # use this to filter and create the append value
    # print(pts.minimum[~round(pts.minimum,3).isin(round(shots.TimeTrueStrike,3))])
    x_points = d_in[d_in.minimum.isin(pts.minimum[~round(pts.minimum, 3).isin(round(shots.TimeTrueStrike, 3))])]
    x_points["TimeTrueStrike"] = x_points.Seconds - 1.5

    x_points2 = x_points[list(x_points.columns[x_points.columns.isin(shots.columns)])]

    shots1 = pd.concat([shots, x_points2], axis=0)
    shots1 = shots1.sort_values("TimeTrueStrike", ascending=True).reset_index(drop=True)

    shots2 = pd.merge_asof(shots1.drop(["GameCount", "PointInGame", "GamePoint", "minimum"], axis=1),
                        pts[["minimum", "maximum", "NumShotsinPt", "GameCount", "PointInGame", "FirstSecond",
                                "Score_PI_Fill2", "WinOrMis", "Tiebreak","OnServe_Corrected", "WonLost", "Win_Extra",
                                "ServeFrom"]], direction="nearest",
                        left_on="TimeTrueStrike", right_on="minimum")
    shots2[["TimeTrueStrike", "RealShot", "FirstSecond", "minimum", "maximum", "NumShotsinPt", "GameCount", "PointInGame",
        "Score_PI_Fill2"]]

    shots3 = shots2[
        (shots2.TimeTrueStrike >= (shots2.minimum - 0.01)) & (shots2.TimeTrueStrike <= (shots2.maximum + 0.01))]

    # is this line different gamepoint from prior, and one further... works for both serve & receiving
    # is the next line a different gamepoint combo, of if "forced" the next next = last
    # otherwise rally point
    # shots3["Game_Point"] = f"{str(shots3.GameCount)}_{str(shots3.PointInGame)}"
    shots3["Game_Point"] = shots3.GameCount.astype("str") + "_" + shots3.PointInGame.astype("str")
    shots3["Eval_Type"] = np.where(
        (shots3.Game_Point != shots3.Game_Point.shift(1)) | (shots3.Game_Point != shots3.Game_Point.shift(2)),
        "Serve_Receive",
        np.where((shots3.Game_Point != shots3.Game_Point.shift(-1)) |
                ((shots3.Game_Point != shots3.Game_Point.shift(-2)) & (shots3.WinOrMis == "Forced")), "Last", "Rally"))

    shots3["StartOrRally"] = np.where(
        (shots3.Game_Point != shots3.Game_Point.shift(1)) | (shots3.Game_Point != shots3.Game_Point.shift(2)),
        "Serve_Receive", "Rally")

    # identify if this is the last shot, can then combine this with the outcome (win loss and how to evaluate)
    # only challenge here is that forced where prior - change order and combine first with second as an extra and as nothing, then do first as third rule
    shots3["IdCriticalLastShot"] = np.where((shots3.Game_Point != shots3.Game_Point.shift(-1)) |
                                            ((shots3.Game_Point != shots3.Game_Point.shift(-2)) & (
                                                        shots3.Win_Extra == "Lost_thru_Forced")), "Last", "")
    # recover this - if prior is last & the game_point is the same then done
    shots3["IdCriticalLastShot2"] = np.where(
        (shots3.IdCriticalLastShot == "Last") & (shots3.IdCriticalLastShot.shift(1) == "Last") &
        (shots3.Game_Point == shots3.Game_Point.shift(1)), "", shots3.IdCriticalLastShot)

    alllast = shots3[shots3.IdCriticalLastShot2 == "Last"].Game_Point.value_counts().reset_index().rename(
        columns={"index": "Game_Point", "Game_Point": "Frequency"})
    allstarts = shots3[shots3.StartOrRally == "Serve_Receive"].Game_Point.value_counts().reset_index().rename(
        columns={"index": "Game_Point", "Game_Point": "Frequency"})
    allpts = shots3.drop_duplicates("Game_Point", keep="first")
    allpts2 = pd.merge(allpts, alllast, how="left", on="Game_Point")
    allpts3 = pd.merge(allpts, allstarts, how="left", on="Game_Point")

    # add in outcomes... this is the value system applying - combination of if carry on or if win directly

    # if serve receive, is there a rally after - then its rally, otherwise look directly at the results
    # reduce to rally, group by - gamecount - re merge
    rallies_base = shots3[shots3.StartOrRally == "Rally"]
    rallies = rallies_base.groupby("Game_Point")["Eval_Type"].count().reset_index().rename(
        columns={"Eval_Type": "Rally"})

    # these rallies include fake rallies, ie lost through a forced on serve or return - this overvalues the serve return

    real_ralliesbase = rallies_base[rallies_base.IdCriticalLastShot2 == "Last"]
    real_ralliesbase["Keep"] = 1
    rallies_base2 = pd.merge(rallies_base, real_ralliesbase[["Game_Point", "Keep"]], on="Game_Point", how="left")
    rallies2 = rallies_base2[rallies_base2.Keep == 1].groupby("Game_Point")["Eval_Type"].count().reset_index().rename(
        columns={"Eval_Type": "Rally"})

    shots4 = pd.merge(shots3, rallies2, how="left", on="Game_Point")
    shots4["Rally"] = shots4.Rally.fillna(0)

    # create the thing you are evaluating, Serve Receive FH etc.... note want to exclude the shot where made the error on forced
    shots4["Eval_Type2"] = np.where(shots4.StartOrRally == "Serve_Receive",
                                    np.where(shots4.OnServe_Corrected == 1, "Serve", "Return"),
                                    np.where((shots4.IdCriticalLastShot == "Last") & (shots4.IdCriticalLastShot2 == ""),
                                            "", shots4.RealShot))

    # outcome building
    # if Serve or Receive and rally >0, rally, otherwise Win_extra
    # if rally and not last, rally, else Win_Extra
    shots4["Outcome"] = np.where((shots4.StartOrRally == "Serve_Receive") & (shots4.Rally > 0), "Rally",
                                np.where((shots4.StartOrRally == "Rally") & (shots4.IdCriticalLastShot2 != "Last"),
                                        "Rally", shots4.Win_Extra))

    # need to trim back the serve receive to one shot in the counts
    shots4["Eval_Type3"] = np.where(
        (shots4.StartOrRally == "Serve_Receive") & (shots4.StartOrRally.shift(-1) == "Serve_Receive") &
        (shots4.Game_Point == shots4.Game_Point.shift(-1)), "", shots4.Eval_Type2)
    shots4["Eval_Type3"] = np.where(
        (shots4.StartOrRally == "Serve_Receive") & (shots4.StartOrRally.shift(-1) == "Serve_Receive") &
        (shots4.Game_Point == shots4.Game_Point.shift(-1)) & (shots4.IdCriticalLastShot2 == "Last"), shots4.Eval_Type2,
        np.where((shots4.StartOrRally == "Serve_Receive") & (shots4.StartOrRally.shift(-1) == "Serve_Receive") &
                (shots4.Game_Point == shots4.Game_Point.shift(-1)), "",
                np.where((shots4.StartOrRally == "Serve_Receive") & (shots4.StartOrRally.shift(1) == "Serve_Receive") &
                        (shots4.Game_Point == shots4.Game_Point.shift(1)) & (
                                    shots4.IdCriticalLastShot2.shift(1) == "Last"), "", shots4.Eval_Type2)))

    shots5 = shots4[shots4.Win_Extra.notnull()]
    shots5["Outcome_WL"] = np.where(shots5.Eval_Type3 == "", "", np.where(shots5.Outcome.str[:3] == "Won", "Won",
                                                                        np.where(shots5.Outcome.str[:3] == "Los",
                                                                                "Lost", "Rally")))

    shots5["ServeReceive_Part"] = np.where(shots5.StartOrRally == "Serve_Receive",
                                        np.where((shots5.Game_Point == shots5.Game_Point.shift(1)), "Two", "One"),
                                        "")

    # creating a better effective metric - include the second part of the shot in the FH* effectiveness
    # create a FH* logic and then combine them to create an effectiveness_Fin...
    shots5["Eval_Type4"] = np.where(
        ((shots5.Eval_Type3 == "Serve") | (shots5.Eval_Type3 == "Return")) & (shots5.ServeReceive_Part == "Two"),
        shots5.RealShot, shots5.Eval_Type3)

    shots5["Eval_Type5"] = np.where(shots5.Eval_Type3.isin(["FH", "BH", "Slice", "OH", "Volley"]),
                                    "Rally", shots5.Eval_Type3)

    # define the critical shots
    # which shot is which - the score represents what happens post this action
    # means you need to shift it down one to align it correctly
    # need to do this on a point basis not a shot basis
    pts_shft = shots5[["OnServe_Corrected", "Game_Point", "Score_PI_Fill2", "WonLost"]].drop_duplicates("Game_Point",
                                                                                                        keep="first")
    pts_shft["Game_Point_Key"] = pts_shft.Game_Point.shift(1)

    # need to know the combination of OnServe_Corrected_Score_PI_Fill2 -
    pts_shft["Starting_Pt"] = pts_shft["Score_PI_Fill2"].shift(1).fillna("MissingData")

    # from this I can create the critical points labels( general, & breakball, pre breakball)

    shots5 = pd.merge(shots5, pts_shft[["Game_Point", "Starting_Pt"]], how="left", on="Game_Point")

    # create a list - then if isin, critical point
    critical = ["Deuce_Deuce", "Three_Three", "Five_Three", "Zero_Three",
                "AdvRet_AdvRet", "Three_Four", "Five_Four", "Zero_Four"]
    #adjust to include all TieBreak points
    # shots5["CriticalPoint"] = np.where(shots5.Score_PI_Fill2.isin(["Five_Zero", "Zero_Five"]), "NonCritical",
    #                                 np.where(shots5.Starting_Pt.isin(critical), "Critical", "NonCritical"))
    # shots5["CriticalPoint_Lv2"] = np.where(shots5.Score_PI_Fill2.isin(["Five_Zero", "Zero_Five"]), "NonCritical",
    #                                     np.where(shots5.Starting_Pt.isin(critical[:4]), "Crit_NonBreak",
    #                                                 np.where(shots5.Starting_Pt.isin(critical[4:]), "Crit_Breakball",
    #                                                         "NonCritical")))
    
    shots5["CriticalPoint"] = np.where(shots5.Tiebreak ==1, "Critical", np.where(shots5.Score_PI_Fill2.isin(["Five_Zero", "Zero_Five"]), "NonCritical",
                                    np.where(shots5.Starting_Pt.isin(critical), "Critical", "NonCritical")))
    shots5["CriticalPoint_Lv2"] = np.where(shots5.Tiebreak == 1, "Crit_Breakball", np.where(shots5.Score_PI_Fill2.isin(["Five_Zero", "Zero_Five"]), "NonCritical",
                                        np.where(shots5.Starting_Pt.isin(critical[:4]), "Crit_NonBreak",
                                                    np.where(shots5.Starting_Pt.isin(critical[4:]), "Crit_Breakball",
                                                            "NonCritical"))))

    # create further variants of that list based on the split you want....breakball, non breakball - on serve or not based on OnServeCorrected

    # from that can look at those actions - win loss, first serve rate etc

    # create the effectivness metric where weight the outcome - weights can change but need to see how working
    # for serve - win thru winner +10, forced +10, mistake +5, rally +1 , lost winner -10, forced -7, mistake -5
    # for return - same

    dic2 = {}
    dic2["Lost_thru_Forced"] = -7
    dic2["Lost_thru_Winner"] = -5
    dic2["Lost_thru_Mistake"] = -10
    dic2["Rally"] = 1
    dic2["Won_thru_Forced"] = 10
    dic2["Won_thru_Winner"] = 10
    dic2["Won_thru_Mistake"] = 5
    dic2[""] = 0
    dic2["0_thru_Mistake"] = 0
    dic2["0_thru_Winner"] = 0
    
    # adjusting the effective score
    #change mapping to an initial effective value
    # shots5["Effective_1"] = shots5.Outcome.apply(lambda x: dic2[x])
    shots5["Effective"] = shots5.Outcome.apply(lambda x: dic2[x])

    #for the 4 elements add in extra details specific to that type
    #where outcome is rally +4, if numshots and serve and first and won +10 etc
        #1st serve -1 for rally, 5 if won with serve, -5 if lost with serve + the 50% rule
        #2nd +4 for rally, 10 if won with serve, -5 if lost
        #1st return +4 for rally
        #2nd return +5* won
    shots5["Effective_Extra"] = np.where((shots5.Eval_Type5 == "Serve") & (shots5.FirstSecond == "First") & (shots5.Outcome == "Rally"), -1,
                                        np.where((shots5.Eval_Type5 == "Serve") & (shots5.FirstSecond == "First") & (shots5.NumShotsinPt == 1) & (shots5.WonLost == "Won"),5,
                                                np.where((shots5.Eval_Type5 == "Serve") & (shots5.FirstSecond == "First") & (shots5.NumShotsinPt == 1) & (shots5.WonLost == "Lost"),-5,
                                np.where((shots5.Eval_Type5 == "Serve") & (shots5.FirstSecond == "Second") & (shots5.Outcome == "Rally"), 4,
                                        np.where((shots5.Eval_Type5 == "Serve") & (shots5.FirstSecond == "Second") & (shots5.NumShotsinPt == 1) & (shots5.WonLost == "Won"),10,
                                                np.where((shots5.Eval_Type5 == "Serve") & (shots5.FirstSecond == "Second") & (shots5.NumShotsinPt == 1) & (shots5.WonLost == "Lost"),-5,
                                np.where((shots5.Eval_Type5 == "Return") & (shots5.FirstSecond == "First") & (shots5.Outcome == "Rally"), 4,
                                        np.where((shots5.Eval_Type5 == "Return") & (shots5.FirstSecond == "Second") & (shots5.NumShotsinPt == 1) & (shots5.WonLost == "Won"), 5,0))))))))

    # combine the 2 elements
    shots5["Effective_1"] = shots5.Effective + shots5.Effective_Extra

    shots5["ReturnShot"] = np.where((shots5.Eval_Type2 == "Return") & (shots5.ServeReceive_Part == "One"),
                                    shots5.RealShot, "")
    shots5["ReturnPlus1Shot"] = np.where((shots5.Eval_Type2 == "Return") & (shots5.ServeReceive_Part == "Two"),
                                        shots5.RealShot, "")
    shots5["PostServeShot"] = np.where((shots5.Eval_Type2 == "Serve") & (shots5.ServeReceive_Part == "Two"),
                                    shots5.RealShot, "")

    returnoutcomes = shots5[(shots5.Eval_Type3 == "Return")]
    returnoutcomes["ReturnOutcomes"] = returnoutcomes.ServeReceive_Part + "_" + returnoutcomes.Outcome
    shots5 = pd.merge(shots5, returnoutcomes[["Game_Point", "ReturnOutcomes"]], on="Game_Point", how="left")

    return shots5, pts

def create_stats(shots5, raw2,  pts, gender, meta_id, bornYear, rating_lev, rating_typ, matchResult, matchLevel, matchSurface, matchType, missing_sensor):
    # if game_point same as above (shift), & this line is return, then return_ shift Return shot, serve + post serve (blank means there was none)
    # if serve or return, if game point is same, then do return_ shift Return shot, serve + post serve,
    # if no second same line for return - write serve for serve
    shots5["ServeReturn_Shot"] = np.where(shots5.Eval_Type5.isin(["Return", "Serve"]),
                                        np.where(shots5.Game_Point == shots5.Game_Point.shift(1),
                                                np.where(shots5.Eval_Type5 == "Return",
                                                            "Return" + "_" + shots5.ReturnShot.shift(1),
                                                            np.where(shots5.Eval_Type5 == "Serve",
                                                                    "Serve" + "_" + shots5.PostServeShot, "")),
                                                np.where(shots5.Eval_Type5 == "Return",
                                                            "Return" + "_" + shots5.ReturnShot,
                                                            np.where(shots5.Eval_Type5 == "Serve", "Serve_Serve", ""))),
                                        "")

    servereturn_shoteff = shots5[shots5.Eval_Type5.isin(["Serve", "Return"])].groupby("ServeReturn_Shot")[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    servereturn_shoteff["Label_0"] = "ServeReturn_ShotEff"
    servereturn_shoteff["Label"] = "Total" + servereturn_shoteff.ServeReturn_Shot

    servereturn_shoteff_1st = \
    shots5[(shots5.Eval_Type5.isin(["Serve", "Return"])) & (shots5.FirstSecond == "First")].groupby("ServeReturn_Shot")[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    servereturn_shoteff_1st["Label_0"] = "ServeReturn_ShotEff"
    servereturn_shoteff_1st["Label"] = "First" + servereturn_shoteff_1st.ServeReturn_Shot

    servereturn_shoteff_2nd = \
    shots5[(shots5.Eval_Type5.isin(["Serve", "Return"])) & (shots5.FirstSecond == "Second")].groupby(
        "ServeReturn_Shot")["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    servereturn_shoteff_2nd["Label_0"] = "ServeReturn_ShotEff"
    servereturn_shoteff_2nd["Label"] = "Second" + servereturn_shoteff_2nd.ServeReturn_Shot

    average_rally = shots5.drop_duplicates(["GameCount", "PointInGame"], keep="first").groupby("Eval_Type2")[
        "NumShotsinPt"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    average_rally["Label_0"] = "AverageRally_Length"
    average_rally["Label"] = "CountandSum_" + average_rally.Eval_Type2

    returns = shots5[shots5.ReturnShot != ""].ReturnShot.value_counts().reset_index().rename(
        columns={"ReturnShot": "Frequency"})
    returns["Label_0"] = "ReturnShot"
    returns["Label"] = "Count" + "_" + returns["index"]
    # firsts only
    returns_1st = shots5[
        (shots5.ReturnShot != "") & (shots5.FirstSecond == "First")].ReturnShot.value_counts().reset_index().rename(
        columns={"ReturnShot": "Frequency"})
    returns_1st["Label_0"] = "ReturnShot_First"
    returns_1st["Label"] = "Count" + "_" + returns_1st["index"]
    # seconds
    returns_2nd = shots5[
        (shots5.ReturnShot != "") & (shots5.FirstSecond == "Second")].ReturnShot.value_counts().reset_index().rename(
        columns={"ReturnShot": "Frequency"})
    returns_2nd["Label_0"] = "ReturnShot_Second"
    returns_2nd["Label"] = "Count" + "_" + returns["index"]

    postserve = shots5[shots5.PostServeShot != ""].PostServeShot.value_counts().reset_index().rename(
        columns={"PostServeShot": "Frequency"})
    postserve["Label_0"] = "PostServeShot"
    postserve["Label"] = "Count" + "_" + postserve["index"]
    # firsts only
    postserve_1st = shots5[(shots5.PostServeShot != "") & (
                shots5.FirstSecond == "First")].PostServeShot.value_counts().reset_index().rename(
        columns={"PostServeShot": "Frequency"})
    postserve_1st["Label_0"] = "PostServeShot_First"
    postserve_1st["Label"] = "Count" + "_" + postserve_1st["index"]
    # seconds
    postserve_2nd = shots5[(shots5.PostServeShot != "") & (
                shots5.FirstSecond == "Second")].PostServeShot.value_counts().reset_index().rename(
        columns={"PostServeShot": "Frequency"})
    postserve_2nd["Label_0"] = "PostServeShot_Second"
    postserve_2nd["Label"] = "Count" + "_" + postserve_2nd["index"]

    # adding in return missed binary
    twos = shots5[(shots5.Eval_Type2 == "Return") & (shots5.ServeReceive_Part == "Two")].groupby("Game_Point")[
        "Key"].count().reset_index().rename(columns={"Key": "HasSecondPartReturn"})
    shots5 = pd.merge(shots5, twos, on="Game_Point", how="left")
    shots5["HasSecondPartReturn"].fillna(0, inplace=True)

    shots5["FailedReturn"] = np.where((shots5.Eval_Type5 == "Return") & (shots5.HasSecondPartReturn == 0) &
                                    ((shots5.Outcome == "Lost_thru_Mistake") | (
                                                shots5.Outcome == "Lost_thru_Forced")), 1, 0)
    shots5["MadeReturn"] = np.where((shots5.Eval_Type5 == "Return") & (shots5.FailedReturn == 0), 1, 0)

    shots5.groupby("FirstSecond")["MadeReturn"].sum()

    # what does the rally mean here?
    # where can I get and use the end in Serve, Return, Rally
    # serve first shot is easier because thats shot 2
    # on return, is is harder? What data do I have here?
    # outcome for serve & return is the outcome of the entire (2 shot sequence)
    # for return, I need to know what happens as a result of the first shot - need a level deeper first so can determine if it was a weak shot
    # where lost to wor lost to forced on line 2 then needs to be pushed back to the return
    # where second is lost to mistake - first should be rally
    # second is lost to winner
    # where won on first, stays as won
    # where won on won on forced or winner on second, needs to be pushed back up to return too

    # need to additionally fix the assignment of lost thru forced back to the serve return element

    # did point start with serve or return
    serveOrreturn = shots5.drop_duplicates("Game_Point", keep="first").groupby(["Game_Point", "Eval_Type2"])[
        "Effective_1"].count().reset_index().rename(columns={"Eval_Type2": "PointsStartWith"})
    shots5 = pd.merge(shots5, serveOrreturn[["Game_Point", "PointsStartWith"]], on="Game_Point", how="left")

    # create the view of how rallies go when start with first or second, and split by return and serve
    rally_outcomes_from_start = \
    shots5[(shots5.StartOrRally == "Rally") & (shots5.Outcome_WL.isin(["Won", "Lost"]))].groupby(
        ["PointsStartWith", "FirstSecond", "Outcome_WL"])["Effective_1"].count().reset_index().rename(
        columns={"Effective_1": "Frequency"})
    rally_outcomes_from_start["Label_0"] = "RallyOutcomes_FromHowStart"
    rally_outcomes_from_start[
        "Label"] = rally_outcomes_from_start.FirstSecond + "_" + rally_outcomes_from_start.PointsStartWith + "_" + rally_outcomes_from_start.Outcome_WL

    # first view of shots - of course will be interesting how these then sum together to create an overview....

    effective_score = shots5[shots5.Eval_Type3 != ""].groupby(["Eval_Type5"])["Effective_1"].agg(
        {"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score["Label_0"] = "Effective_Score"
    effective_score["Label"] = "Effective_Score" + "_" + effective_score["Eval_Type5"]

    # create different types of effective score - first second, deuce, ad
    effective_score_first = shots5[(shots5.Eval_Type3 != "") & (shots5.FirstSecond == "First")].groupby(["Eval_Type5"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_first["Label_0"] = "Effective_Score_First"
    effective_score_first["Label"] = "Effective_Score_First" + "_" + effective_score_first["Eval_Type5"]

    effective_score_second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.FirstSecond == "Second")].groupby(["Eval_Type5"])["Effective_1"].agg(
        {"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_second["Label_0"] = "Effective_Score_Second"
    effective_score_second["Label"] = "Effective_Score_Second" + "_" + effective_score_second["Eval_Type5"]

    # deuce all
    effective_score_deuce = shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce")].groupby(["Eval_Type5"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_deuce["Label_0"] = "Effective_Score_Deuce"
    effective_score_deuce["Label"] = "Effective_Score_Deuce" + "_" + effective_score_deuce["Eval_Type5"]

    # ad all
    effective_score_ad = shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv")].groupby(["Eval_Type5"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_ad["Label_0"] = "Effective_Score_Adv"
    effective_score_ad["Label"] = "Effective_Score_Adv" + "_" + effective_score_ad["Eval_Type5"]

    # deuce first
    effective_score_deuce_first = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce") & (shots5.FirstSecond == "First")].groupby(
        ["Eval_Type5"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_deuce_first["Label_0"] = "Effective_Score_Deuce_First"
    effective_score_deuce_first["Label"] = "Effective_Score_Deuce_First" + "_" + effective_score_deuce_first[
        "Eval_Type5"]

    # deuce second
    effective_score_deuce_second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce") & (shots5.FirstSecond == "Second")].groupby(
        ["Eval_Type5"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_deuce_second["Label_0"] = "Effective_Score_Deuce_Second"
    effective_score_deuce_second["Label"] = "Effective_Score_Deuce_Second" + "_" + effective_score_deuce_second[
        "Eval_Type5"]

    # ad first
    effective_score_ad_first = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv") & (shots5.FirstSecond == "First")].groupby(
        ["Eval_Type5"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_ad_first["Label_0"] = "Effective_Score_Adv_First"
    effective_score_ad_first["Label"] = "Effective_Score_Adv_First" + "_" + effective_score_ad_first["Eval_Type5"]

    # ad second
    effective_score_ad_second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv") & (shots5.FirstSecond == "Second")].groupby(
        ["Eval_Type5"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_ad_second["Label_0"] = "Effective_Score_Adv_Second"
    effective_score_ad_second["Label"] = "Effective_Score_Adv_Second" + "_" + effective_score_ad_second["Eval_Type5"]

    # generic win loss of shots
    WL_Generic_OutcomeGen_Freq = shots5[shots5.Eval_Type3 != ""].groupby(["Eval_Type5", "Outcome_WL"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})

    # simple win loss behind first & second serves

    WL_Generic_OutcomeGen_Freq[
        "Label"] = WL_Generic_OutcomeGen_Freq.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq.Outcome_WL
    WL_Generic_OutcomeGen_Freq["Label_0"] = "Shots_By_OutComeGen"
    WL_Generic_OutcomeGen_Freq

    # recreate for first second, deuce ad
    WL_Generic_OutcomeGen_Freq_First = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.FirstSecond == "First")].groupby(["Eval_Type5", "Outcome_WL"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_First[
        "Label"] = WL_Generic_OutcomeGen_Freq_First.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_First.Outcome_WL
    WL_Generic_OutcomeGen_Freq_First["Label_0"] = "Shots_By_OutComeGen_First"

    # second
    WL_Generic_OutcomeGen_Freq_Second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.FirstSecond == "Second")].groupby(["Eval_Type5", "Outcome_WL"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_Second[
        "Label"] = WL_Generic_OutcomeGen_Freq_Second.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_Second.Outcome_WL
    WL_Generic_OutcomeGen_Freq_Second["Label_0"] = "Shots_By_OutComeGen_Second"

    # deuce
    WL_Generic_OutcomeGen_Freq_Deuce = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce")].groupby(["Eval_Type5", "Outcome_WL"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_Deuce[
        "Label"] = WL_Generic_OutcomeGen_Freq_Deuce.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_Deuce.Outcome_WL
    WL_Generic_OutcomeGen_Freq_Deuce["Label_0"] = "Shots_By_OutComeGen_Deuce"
    # ad
    WL_Generic_OutcomeGen_Freq_Ad = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv")].groupby(["Eval_Type5", "Outcome_WL"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_Ad[
        "Label"] = WL_Generic_OutcomeGen_Freq_Ad.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_Ad.Outcome_WL
    WL_Generic_OutcomeGen_Freq_Ad["Label_0"] = "Shots_By_OutComeGen_Adv"

    # deuce first
    WL_Generic_OutcomeGen_Freq_Deuce_First = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce") & (shots5.FirstSecond == "First")].groupby(
        ["Eval_Type5", "Outcome_WL"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_Deuce_First[
        "Label"] = WL_Generic_OutcomeGen_Freq_Deuce_First.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_Deuce_First.Outcome_WL
    WL_Generic_OutcomeGen_Freq_Deuce_First["Label_0"] = "Shots_By_OutComeGen_Deuce_First"

    # deuce second
    WL_Generic_OutcomeGen_Freq_Deuce_Second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce") & (shots5.FirstSecond == "Second")].groupby(
        ["Eval_Type5", "Outcome_WL"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_Deuce_Second[
        "Label"] = WL_Generic_OutcomeGen_Freq_Deuce_Second.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_Deuce_Second.Outcome_WL
    WL_Generic_OutcomeGen_Freq_Deuce_Second["Label_0"] = "Shots_By_OutComeGen_Deuce_Second"

    # adv first
    WL_Generic_OutcomeGen_Freq_Ad_First = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv") & (shots5.FirstSecond == "First")].groupby(
        ["Eval_Type5", "Outcome_WL"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_Ad_First[
        "Label"] = WL_Generic_OutcomeGen_Freq_Ad_First.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_Ad_First.Outcome_WL
    WL_Generic_OutcomeGen_Freq_Ad_First["Label_0"] = "Shots_By_OutComeGen_Adv_First"

    # adv second
    WL_Generic_OutcomeGen_Freq_Ad_Second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv") & (shots5.FirstSecond == "Second")].groupby(
        ["Eval_Type5", "Outcome_WL"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})

    WL_Generic_OutcomeGen_Freq_Ad_Second[
        "Label"] = WL_Generic_OutcomeGen_Freq_Ad_Second.Eval_Type5 + "_" + WL_Generic_OutcomeGen_Freq_Ad_Second.Outcome_WL
    WL_Generic_OutcomeGen_Freq_Ad_Second["Label_0"] = "Shots_By_OutComeGen_Adv_Second"

    # detailed win loss of shots
    WL_Outcome_Eff = shots5[shots5.Eval_Type3 != ""].groupby(["Eval_Type5", "Outcome"])["Effective_1"].agg(
        {"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})

    # simple win loss behind first & second serves

    WL_Outcome_Eff["Label"] = WL_Outcome_Eff.Eval_Type5 + "_" + WL_Outcome_Eff.Outcome
    WL_Outcome_Eff["Label_0"] = "Eval1_By_OutCome_Effective"

    # create the splits by first second deuce and ad
    WL_Outcome_Eff_First = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.FirstSecond == "First")].groupby(["Eval_Type5", "Outcome"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_First["Label"] = WL_Outcome_Eff_First.Eval_Type5 + "_" + WL_Outcome_Eff_First.Outcome
    WL_Outcome_Eff_First["Label_0"] = "Eval1_By_OutCome_Effective_First"

    # second
    WL_Outcome_Eff_Second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.FirstSecond == "Second")].groupby(["Eval_Type5", "Outcome"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_Second["Label"] = WL_Outcome_Eff_Second.Eval_Type5 + "_" + WL_Outcome_Eff_Second.Outcome
    WL_Outcome_Eff_Second["Label_0"] = "Eval1_By_OutCome_Effective_Second"

    # deuce
    WL_Outcome_Eff_Deuce = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce")].groupby(["Eval_Type5", "Outcome"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_Deuce["Label"] = WL_Outcome_Eff_Deuce.Eval_Type5 + "_" + WL_Outcome_Eff_Deuce.Outcome
    WL_Outcome_Eff_Deuce["Label_0"] = "Eval1_By_OutCome_Effective_Deuce"
    # ad
    WL_Outcome_Eff_Ad = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv")].groupby(["Eval_Type5", "Outcome"])[
        "Effective_1"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_Ad["Label"] = WL_Outcome_Eff_Ad.Eval_Type5 + "_" + WL_Outcome_Eff_Ad.Outcome
    WL_Outcome_Eff_Ad["Label_0"] = "Eval1_By_OutCome_Effective_Adv"

    # deuce first
    WL_Outcome_Eff_Deuce_First = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce") & (shots5.FirstSecond == "First")].groupby(
        ["Eval_Type5", "Outcome"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_Deuce_First[
        "Label"] = WL_Outcome_Eff_Deuce_First.Eval_Type5 + "_" + WL_Outcome_Eff_Deuce_First.Outcome
    WL_Outcome_Eff_Deuce_First["Label_0"] = "Eval1_By_OutCome_Effective_Deuce_First"
    # deuce second
    WL_Outcome_Eff_Deuce_Second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Deuce") & (shots5.FirstSecond == "Second")].groupby(
        ["Eval_Type5", "Outcome"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_Deuce_Second[
        "Label"] = WL_Outcome_Eff_Deuce_Second.Eval_Type5 + "_" + WL_Outcome_Eff_Deuce_Second.Outcome
    WL_Outcome_Eff_Deuce_Second["Label_0"] = "Eval1_By_OutCome_Effective_Deuce_Second"

    # adv first
    WL_Outcome_Eff_Ad_First = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv") & (shots5.FirstSecond == "First")].groupby(
        ["Eval_Type5", "Outcome"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_Ad_First["Label"] = WL_Outcome_Eff_Ad_First.Eval_Type5 + "_" + WL_Outcome_Eff_Ad_First.Outcome
    WL_Outcome_Eff_Ad_First["Label_0"] = "Eval1_By_OutCome_Effective_Adv_First"
    # adv second

    WL_Outcome_Eff_Ad_Second = \
    shots5[(shots5.Eval_Type3 != "") & (shots5.ServeFrom == "Adv") & (shots5.FirstSecond == "Second")].groupby(
        ["Eval_Type5", "Outcome"])["Effective_1"].agg({"count", "sum"}).reset_index().rename(
        columns={"count": "Frequency", "sum": "Effective_1"})
    WL_Outcome_Eff_Ad_Second["Label"] = WL_Outcome_Eff_Ad_Second.Eval_Type5 + "_" + WL_Outcome_Eff_Ad_Second.Outcome
    WL_Outcome_Eff_Ad_Second["Label_0"] = "Eval1_By_OutCome_Effective_Adv_Second"

    # create the effective score for the shots - use eval_4

    effective_score_shots = shots5[~shots5.Eval_Type4.isin(["", "Serve", "Return"])].groupby(["Eval_Type4"])[
        "Effective"].agg({"count", "sum"}).reset_index().rename(columns={"count": "Frequency", "sum": "Effective_1"})
    effective_score_shots["Label_0"] = "Effective_Score2_Shots"
    effective_score_shots["Label"] = "Effective_Score" + "_" + effective_score_shots["Eval_Type4"]

    # generic win loss of shots - but at the lower level where second shot of return or serve included
    Shotslower_OutcomeGen_Freq = \
    shots5[~shots5.Eval_Type4.isin(["", "Serve", "Return"])].groupby(["Eval_Type4", "Outcome_WL"])[
        "Effective_1"].count().reset_index().rename(columns={"Effective_1": "Frequency"})

    # simple win loss behind first & second serves

    Shotslower_OutcomeGen_Freq[
        "Label"] = Shotslower_OutcomeGen_Freq.Eval_Type4 + "_" + Shotslower_OutcomeGen_Freq.Outcome_WL
    Shotslower_OutcomeGen_Freq["Label_0"] = "ShotsEval2_By_OutComeGen_Effective"

    # detailed win loss of shots - but at the lower level where second shot of return or serve included
    # shots5["Outcome_WL"] = np.where(shots5.Eval_Type3 == "", "", np.where(shots5.Outcome.str[:3] == "Won", "Won",
    #                                 np.where(shots5.Outcome.str[:3] == "Los", "Lost", "Rally")))
    Shotslower_Outcome_Eff = \
    shots5[~shots5.Eval_Type4.isin(["", "Serve", "Return"])].groupby(["Eval_Type4", "Outcome"])[
        "Effective_1"].count().reset_index().rename(columns={"Effective_1": "Frequency"})

    # simple win loss behind first & second serves

    Shotslower_Outcome_Eff["Label"] = Shotslower_Outcome_Eff.Eval_Type4 + "_" + Shotslower_Outcome_Eff.Outcome
    Shotslower_Outcome_Eff["Label_0"] = "Eval2_By_OutCome_Effective"

    # for serves & returns only, identify the part of the serve
    # look to see if the gamePoint matches and is eval_type2 is same as above

    WL_Generic_Part_OutcomeGen_Freq = \
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "ServeReceive_Part", "Outcome_WL"])["Effective_1"].count().reset_index().rename(
        columns={"Effective_1": "Frequency"})

    WL_Generic_Part_OutcomeGen_Freq[
        "Label"] = WL_Generic_Part_OutcomeGen_Freq.Eval_Type3 + "_" + WL_Generic_Part_OutcomeGen_Freq.ServeReceive_Part + "_" + WL_Generic_Part_OutcomeGen_Freq.Outcome_WL
    WL_Generic_Part_OutcomeGen_Freq["Label_0"] = "Shots_By_Part_OutComeGen"

    # showing more detail on the win loss part - by full outcome
    WL_Generic_Part_Outcome_Freq = shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "ServeReceive_Part", "Outcome"])["Effective_1"].count().reset_index().rename(
        columns={"Effective_1": "Frequency"})

    WL_Generic_Part_Outcome_Freq[
        "Label"] = WL_Generic_Part_Outcome_Freq.Eval_Type3 + "_" + WL_Generic_Part_Outcome_Freq.ServeReceive_Part + "_" + WL_Generic_Part_Outcome_Freq.Outcome
    WL_Generic_Part_Outcome_Freq["Label_0"] = "Shots_By_Part_OutCome"

    # showing more detail on the win loss part - by shot now
    WL_Generic_Part_Outcome_Shot_Freq = \
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "ServeReceive_Part", "Outcome", "RealShot"])["Effective_1"].count().reset_index().rename(
        columns={"Effective_1": "Frequency"})

    WL_Generic_Part_Outcome_Shot_Freq[
        "Label"] = WL_Generic_Part_Outcome_Shot_Freq.Eval_Type3 + "_" + WL_Generic_Part_Outcome_Shot_Freq.ServeReceive_Part + "_" + WL_Generic_Part_Outcome_Shot_Freq.Outcome + "_" + WL_Generic_Part_Outcome_Shot_Freq.RealShot
    WL_Generic_Part_Outcome_Shot_Freq["Label_0"] = "Shots_By_Part_OutCome_Shot"

    # ### Create extra split of First & Second
    # no effectiveness score  -  its a visual - detail behind is important for now

    # simple win loss behind first & second serves
    WL_FirstSecond_OutcomeGen_Freq = \
    shots5[shots5.Eval_Type3 != ""].groupby(["Eval_Type3", "FirstSecond", "Outcome_WL"])[
        "Effective_1"].count().reset_index().rename(columns={"Effective_1": "Frequency"})

    # simple win loss behind first & second serves

    WL_FirstSecond_OutcomeGen_Freq[
        "Label"] = WL_FirstSecond_OutcomeGen_Freq.Eval_Type3 + "_" + WL_FirstSecond_OutcomeGen_Freq.FirstSecond + "_" + WL_FirstSecond_OutcomeGen_Freq.Outcome_WL
    WL_FirstSecond_OutcomeGen_Freq["Label_0"] = "Start_SplitBy_Serve_Type_OutComeGen"

    # where winning losing on the serve part of the game
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "FirstSecond", "ServeReceive_Part", "Outcome_WL"])[
        "Effective_1"].count()  # .reset_index().rename(columns = {"Effective_1": "Frequency"})

    # showing more detail on the win loss part - by full outcome
    WL_FirstSecond_Part_Outcome_Freq = \
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "FirstSecond", "ServeReceive_Part", "Outcome"])["Effective_1"].count().reset_index().rename(
        columns={"Effective_1": "Frequency"})

    WL_FirstSecond_Part_Outcome_Freq[
        "Label"] = WL_FirstSecond_Part_Outcome_Freq.Eval_Type3 + "_" + WL_FirstSecond_Part_Outcome_Freq.FirstSecond + "_" + WL_FirstSecond_Part_Outcome_Freq.ServeReceive_Part + "_" + WL_FirstSecond_Part_Outcome_Freq.Outcome
    WL_FirstSecond_Part_Outcome_Freq["Label_0"] = "Start_SplitBy_Serve_Type_Part_OutCome"

    # showing more detail on the win loss part - by shot now
    WL_FirstSecond_Part_Outcome_Shot_Freq = \
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "FirstSecond", "ServeReceive_Part", "Outcome", "RealShot"])[
        "Effective_1"].count().reset_index().rename(columns={"Effective_1": "Frequency"})

    WL_FirstSecond_Part_Outcome_Shot_Freq[
        "Label"] = WL_FirstSecond_Part_Outcome_Shot_Freq.Eval_Type3 + "_" + WL_FirstSecond_Part_Outcome_Shot_Freq.FirstSecond + "_" + WL_FirstSecond_Part_Outcome_Shot_Freq.ServeReceive_Part + "_" + WL_FirstSecond_Part_Outcome_Shot_Freq.Outcome + "_" + WL_FirstSecond_Part_Outcome_Shot_Freq.RealShot
    WL_FirstSecond_Part_Outcome_Shot_Freq["Label_0"] = "Start_SplitBy_Serve_Type_Part_OutCome_Shot"

    # ### Splittng First & Second into Deuce & Ad Now

    # simple win loss behind first & second serves
    WL_FirstSecond_From_OutcomeGen_Freq = \
    shots5[shots5.Eval_Type3 != ""].groupby(["Eval_Type3", "FirstSecond", "ServeFrom", "Outcome_WL"])[
        "Effective_1"].count().reset_index().rename(columns={"Effective_1": "Frequency"})
    WL_FirstSecond_From_OutcomeGen_Freq[
        "Label"] = WL_FirstSecond_From_OutcomeGen_Freq.Eval_Type3 + "_" + WL_FirstSecond_From_OutcomeGen_Freq.FirstSecond + "_" + WL_FirstSecond_From_OutcomeGen_Freq.ServeFrom + "_" + WL_FirstSecond_From_OutcomeGen_Freq.Outcome_WL
    WL_FirstSecond_From_OutcomeGen_Freq["Label_0"] = "Start_SplitBy_Serve_Type_Origin_OutComeGen"

    # showing more detail on the win loss part - by full outcome
    WL_FirstSecond_From_Part_Outcome_Freq = \
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "FirstSecond", "ServeFrom", "ServeReceive_Part", "Outcome"])[
        "Effective_1"].count().reset_index().rename(columns={"Effective_1": "Frequency"})
    WL_FirstSecond_From_Part_Outcome_Freq[
        "Label"] = WL_FirstSecond_From_Part_Outcome_Freq.Eval_Type3 + "_" + WL_FirstSecond_From_Part_Outcome_Freq.FirstSecond + "_" + WL_FirstSecond_From_Part_Outcome_Freq.ServeFrom + "_" + WL_FirstSecond_From_Part_Outcome_Freq.ServeReceive_Part + "_" + WL_FirstSecond_From_Part_Outcome_Freq.Outcome
    WL_FirstSecond_From_Part_Outcome_Freq["Label_0"] = "Start_SplitBy_Serve_Type_Origin_Part_OutCome"

    # showing more detail on the win loss part - by shot now
    WL_FirstSecond_From_Part_Outcome_Shot_Freq = \
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "FirstSecond", "ServeFrom", "ServeReceive_Part", "Outcome", "RealShot"])[
        "Effective_1"].count().reset_index().rename(columns={"Effective_1": "Frequency"})
    WL_FirstSecond_From_Part_Outcome_Shot_Freq[
        "Label"] = WL_FirstSecond_From_Part_Outcome_Shot_Freq.Eval_Type3 + "_" + WL_FirstSecond_From_Part_Outcome_Shot_Freq.FirstSecond + "_" + WL_FirstSecond_From_Part_Outcome_Shot_Freq.ServeFrom + "_" + WL_FirstSecond_From_Part_Outcome_Shot_Freq.ServeReceive_Part + "_" + WL_FirstSecond_From_Part_Outcome_Shot_Freq.Outcome + "_" + WL_FirstSecond_From_Part_Outcome_Shot_Freq.RealShot
    WL_FirstSecond_From_Part_Outcome_Shot_Freq["Label_0"] = "Start_SplitBy_Serve_Type_Origin_Part_OutCome_Shot"

    # Return/serve_deuce/ad_outcomeHL
    WL_FirstSecond_From_OutcomeGen = \
    shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "FirstSecond", "ServeFrom", "Outcome_WL"])["Effective_1"].count().reset_index().rename(
        columns={"Effective_1": "Frequency"})
    WL_FirstSecond_From_OutcomeGen[
        "Label"] = WL_FirstSecond_From_OutcomeGen.Eval_Type3 + "_" + WL_FirstSecond_From_OutcomeGen.FirstSecond + "_" + WL_FirstSecond_From_OutcomeGen.ServeFrom + "_" + WL_FirstSecond_From_OutcomeGen.Outcome_WL

    WL_FirstSecond_From_OutcomeGen["Label_0"] = "Start_SplitBy_Serve_Type_Origin_OutComeHL"

    # Return/serve_deuce/ad_outcome deeper
    WL_FirstSecond_From_Outcome = shots5[(shots5.StartOrRally == "Serve_Receive") & (shots5.Eval_Type3 != "")].groupby(
        ["Eval_Type3", "FirstSecond", "ServeFrom", "Outcome"])["Effective_1"].count().reset_index().rename(
        columns={"Effective_1": "Frequency"})
    WL_FirstSecond_From_Outcome[
        "Label"] = WL_FirstSecond_From_Outcome.Eval_Type3 + "_" + WL_FirstSecond_From_Outcome.FirstSecond + "_" + WL_FirstSecond_From_Outcome.ServeFrom + "_" + WL_FirstSecond_From_Outcome.Outcome

    WL_FirstSecond_From_Outcome["Label_0"] = "Start_SplitBy_Serve_Type_Origin_OutCome"

    # Calculate first serve rate, split by deuce & second
    # take the first part of the shot - gives you how the point started
    part1 = shots5[shots5.Eval_Type3 != ""].drop_duplicates(["Game_Point"], keep="first").drop("Eval_Type3", axis=1)

    # create an element which has the outcome in it - where EvalType 2 is Return or Serve
    serveret_outcomes = shots5[(shots5.Eval_Type3.isin(["Return", "Serve"]))][["Game_Point", "Eval_Type3"]]
    # merge the 2
    part1_2 = pd.merge(part1, serveret_outcomes, on="Game_Point", how="left")

    # from this then have the total number of points -
    total_serves = part1_2[part1_2.Eval_Type3 == "Serve"].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    total_serves["ServeFrom"] = "Total"
    total_serves_split = part1_2[part1_2.Eval_Type3 == "Serve"].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    first_serves = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First")].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    first_serves["ServeFrom"] = "Total"
    first_serves_split = \
    part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First")].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    first_serves_1 = pd.concat([first_serves_split, first_serves], axis=0)
    total_serves_1 = pd.concat([total_serves_split, total_serves], axis=0)

    first_serves_1["Label"] = first_serves_1.Eval_Type3 + "_" + first_serves_1.ServeFrom

    first_serves_1["Label_0"] = "FirstServes_by_ServeFrom_Freq"

    total_serves_1["Label"] = total_serves_1.Eval_Type3 + "_" + total_serves_1.ServeFrom

    total_serves_1["Label_0"] = "TotalServes_by_ServeFrom_Freq"

    # add critical elements and with filters to this
    # format in a way that can add to the eval table
    # SERVES
    # need total critical and non critical points
    serves_critnon = part1_2[part1_2.Eval_Type3 == "Serve"].groupby(["Eval_Type3", "CriticalPoint"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    serves_critnon["Label"] = serves_critnon.Eval_Type3 + "_" + serves_critnon.CriticalPoint
    serves_critnon["Label_0"] = "TotalServes_by_CriticalHL"

    # critical and non critical - first serve
    servesfirst_critnon = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "CriticalPoint"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servesfirst_critnon["Label"] = servesfirst_critnon.Eval_Type3 + "_" + servesfirst_critnon.CriticalPoint
    servesfirst_critnon["Label_0"] = "FirstServes_by_CriticalHL"
    # critical and non critical - first serve - deuce ad

    servesfirst_critnon_from = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servesfirst_critnon_from[
        "Label"] = servesfirst_critnon_from.Eval_Type3 + "_" + servesfirst_critnon_from.CriticalPoint + "_" + servesfirst_critnon_from.ServeFrom

    servesfirst_critnon_from["Label_0"] = "FirstServes_by_CriticalHL_Origin"

    # critical and non critical - second serve
    servessecond_critnon = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "CriticalPoint"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servessecond_critnon["Label"] = servessecond_critnon.Eval_Type3 + "_" + servessecond_critnon.CriticalPoint
    servessecond_critnon["Label_0"] = "SecondServes_by_CriticalHL"

    # critical and non critical - second serve - deuce ad
    servessecond_critnon_from = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servessecond_critnon_from[
        "Label"] = servessecond_critnon_from.Eval_Type3 + "_" + servessecond_critnon_from.CriticalPoint + "_" + servessecond_critnon_from.ServeFrom
    servessecond_critnon_from["Label_0"] = "SecondServes_by_CriticalHL_Origin"

    # outcomes of
    # critical and non critical - first serve - outcomes
    servesfirst_critnon_outcomes = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servesfirst_critnon_outcomes[
        "Label"] = servesfirst_critnon_outcomes.Eval_Type3 + "_" + servesfirst_critnon_outcomes.CriticalPoint + "_" + servesfirst_critnon_outcomes.Outcome_WL
    servesfirst_critnon_outcomes["Label_0"] = "FirstServes_by_CriticalHL_OutcomeGen"

    # critical and non critical - first serve - deuce ad outcmes
    servesfirst_critnon_from_outcomes = \
    part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servesfirst_critnon_from_outcomes[
        "Label"] = servesfirst_critnon_from_outcomes.Eval_Type3 + "_" + servesfirst_critnon_from_outcomes.CriticalPoint + "_" + servesfirst_critnon_from_outcomes.ServeFrom + "_" + servesfirst_critnon_from_outcomes.Outcome_WL
    servesfirst_critnon_from_outcomes["Label_0"] = "FirstServes_by_CriticalHL_Origin_OutcomeGen"

    # critical and non critical - second serve outcomes
    servessecond_critnon_outcome = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servessecond_critnon_outcome[
        "Label"] = servessecond_critnon_outcome.Eval_Type3 + "_" + servessecond_critnon_outcome.CriticalPoint + "_" + servessecond_critnon_outcome.Outcome_WL
    servessecond_critnon_outcome["Label_0"] = "SecondServes_by_CriticalHL_OutcomeGen"
    # critical and non critical - second serve - deuce ad outcomes
    servessecond_critnon_from_outcome = \
    part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servessecond_critnon_from_outcome[
        "Label"] = servessecond_critnon_from_outcome.Eval_Type3 + "_" + servessecond_critnon_from_outcome.CriticalPoint + "_" + servessecond_critnon_from_outcome.ServeFrom + "_" + servessecond_critnon_from_outcome.Outcome_WL
    servessecond_critnon_from_outcome["Label_0"] = "SecondServes_by_CriticalHL_Origin_OutcomeGen"

    # same as above
    # filter to critical points and split on second level critical
    # change the Label element
    # & rename variable
    # change label_0

    # breakballs & non breakballs split - filter on critical only
    # SERVES
    # need total critical and non critical points
    serves_critbreak = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.CriticalPoint == "Critical")].groupby(
        ["Eval_Type3", "CriticalPoint_Lv2"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    serves_critbreak["Label"] = serves_critbreak.Eval_Type3 + "_" + serves_critbreak.CriticalPoint_Lv2
    serves_critbreak["Label_0"] = "TotalServes_by_Critical_lv2"

    # critical and non critical - first serve
    servesfirst_critbreak = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servesfirst_critbreak["Label"] = servesfirst_critbreak.Eval_Type3 + "_" + servesfirst_critbreak.CriticalPoint_Lv2
    servesfirst_critbreak["Label_0"] = "FirstServes_by_Critical_lv2"
    # critical and non critical - first serve - deuce ad

    servesfirst_critbreak_from = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servesfirst_critbreak_from[
        "Label"] = servesfirst_critbreak_from.Eval_Type3 + "_" + servesfirst_critbreak_from.CriticalPoint_Lv2 + "_" + servesfirst_critbreak_from.ServeFrom

    servesfirst_critbreak_from["Label_0"] = "FirstServes_by_Critical_lv2_Origin"

    # critical and non critical - second serve
    servessecond_critbreak = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servessecond_critbreak["Label"] = servessecond_critbreak.Eval_Type3 + "_" + servessecond_critbreak.CriticalPoint_Lv2
    servessecond_critbreak["Label_0"] = "SecondServes_by_Critical_lv2"

    # critical and non critical - second serve - deuce ad
    servessecond_critbreak_from = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servessecond_critbreak_from[
        "Label"] = servessecond_critbreak_from.Eval_Type3 + "_" + servessecond_critbreak_from.CriticalPoint_Lv2 + "_" + servessecond_critbreak_from.ServeFrom
    servessecond_critbreak_from["Label_0"] = "SecondServes_by_Critical_lv2_Origin"

    # outcomes of
    # critical and non critical - first serve - outcomes
    servesfirst_critbreak_outcomes = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2", "Outcome_WL"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servesfirst_critbreak_outcomes[
        "Label"] = servesfirst_critbreak_outcomes.Eval_Type3 + "_" + servesfirst_critbreak_outcomes.CriticalPoint_Lv2 + "_" + servesfirst_critbreak_outcomes.Outcome_WL
    servesfirst_critbreak_outcomes["Label_0"] = "FirstServes_by_Critical_lv2_OutcomeGen"

    # critical and non critical - first serve - deuce ad outcmes
    servesfirst_critbreak_from_outcomes = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "First") & (
                part1_2.CriticalPoint == "Critical")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servesfirst_critbreak_from_outcomes[
        "Label"] = servesfirst_critbreak_from_outcomes.Eval_Type3 + "_" + servesfirst_critbreak_from_outcomes.CriticalPoint_Lv2 + "_" + servesfirst_critbreak_from_outcomes.ServeFrom + "_" + servesfirst_critbreak_from_outcomes.Outcome_WL
    servesfirst_critbreak_from_outcomes["Label_0"] = "FirstServes_by_Critical_lv2_Origin_OutcomeGen"

    # critical and non critical - second serve outcomes
    servessecond_critbreak_outcome = part1_2[(part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2", "Outcome_WL"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    servessecond_critbreak_outcome[
        "Label"] = servessecond_critbreak_outcome.Eval_Type3 + "_" + servessecond_critbreak_outcome.CriticalPoint_Lv2 + "_" + servessecond_critbreak_outcome.Outcome_WL
    servessecond_critbreak_outcome["Label_0"] = "SecondServes_by_Critical_lv2_OutcomeGen"
    # critical and non critical - second serve - deuce ad outcomes
    servessecond_critbreak_from_outcome = part1_2[
        (part1_2.Eval_Type3 == "Serve") & (part1_2.FirstSecond == "Second") & (
                    part1_2.CriticalPoint == "Critical")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    servessecond_critbreak_from_outcome[
        "Label"] = servessecond_critbreak_from_outcome.Eval_Type3 + "_" + servessecond_critbreak_from_outcome.CriticalPoint_Lv2 + "_" + servessecond_critbreak_from_outcome.ServeFrom + "_" + servessecond_critbreak_from_outcome.Outcome_WL
    servessecond_critbreak_from_outcome["Label_0"] = "SecondServes_by_Critical_lv2_Origin_OutcomeGen"

    # creating the number of returns for return rate
    # this creates the number of return points started
    total_return = part1_2[(part1_2.Eval_Type3 == "Return")].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    total_return["ServeFrom"] = "Total"
    total_return_split = part1_2[part1_2.Eval_Type3 == "Return"].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    first_serves_return = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First")].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    first_serves_return["ServeFrom"] = "Total"
    first_serves_return_split = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First")].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    second_serves_return = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second")].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    second_serves_return["ServeFrom"] = "Total"
    second_serves_return_split = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second")].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    second_serves_return1 = pd.concat([second_serves_return_split, second_serves_return], axis=0)
    first_serves_return1 = pd.concat([first_serves_return_split, first_serves_return], axis=0)
    total_servesreturn_1 = pd.concat([total_return_split, total_return], axis=0)

    second_serves_return1["Label"] = second_serves_return1.Eval_Type3 + "_" + second_serves_return1.ServeFrom

    second_serves_return1["Label_0"] = "SecondServesReturnHit_by_ServeFrom_Freq"

    first_serves_return1["Label"] = first_serves_return1.Eval_Type3 + "_" + first_serves_return1.ServeFrom

    first_serves_return1["Label_0"] = "FirstServesReturnHit_by_ServeFrom_Freq"

    total_servesreturn_1["Label"] = total_servesreturn_1.Eval_Type3 + "_" + total_servesreturn_1.ServeFrom

    total_servesreturn_1["Label_0"] = "TotalServesReturnHit_by_ServeFrom_Freq"

    # this is the number of returns made
    # removing if lost in first part
    # total_return_made = part1_2[(part1_2.Eval_Type3 == "Return") &
    #                             (part1_2.Outcome_WL + "_" +part1_2.ServeReceive_Part + "_" + part1_2.WinOrMis != "Lost_One_Mistake")].groupby(["Eval_Type3"])["Outcome"].count().reset_index().rename(columns = {"Outcome": "Frequency"})
    # total_return_made["ServeFrom"] = "Total"
    # total_return_made_split = part1_2[(part1_2.Eval_Type3 == "Return")
    #                                   & (part1_2.Outcome_WL + "_" +part1_2.ServeReceive_Part + "_" +part1_2.WinOrMis != "Lost_One_Mistake")].groupby(["Eval_Type3","ServeFrom"])["Outcome"].count().reset_index().rename(columns = {"Outcome": "Frequency"})

    # first_serves_return_made = part1_2[(part1_2.Eval_Type3 == "Return") & ( part1_2.FirstSecond == "First") &
    #                                    (part1_2.Outcome_WL + "_" +part1_2.ServeReceive_Part+ "_" +part1_2.WinOrMis != "Lost_One_Mistake")].groupby(["Eval_Type3"])["Outcome"].count().reset_index().rename(columns = {"Outcome": "Frequency"})
    # first_serves_return_made["ServeFrom"] = "Total"
    # first_serves_returnmade_split = part1_2[(part1_2.Eval_Type3 == "Return") & ( part1_2.FirstSecond == "First")&
    #                                         (part1_2.Outcome_WL + "_" +part1_2.ServeReceive_Part + "_" +part1_2.WinOrMis != "Lost_One_Mistake")].groupby(["Eval_Type3","ServeFrom"])["Outcome"].count().reset_index().rename(columns = {"Outcome": "Frequency"})

    # second_serves_return_made = part1_2[(part1_2.Eval_Type3 == "Return") & ( part1_2.FirstSecond == "Second") &
    #                                     (part1_2.Outcome_WL + "_" +part1_2.ServeReceive_Part + "_" +part1_2.WinOrMis != "Lost_One_Mistake")].groupby(["Eval_Type3"])["Outcome"].count().reset_index().rename(columns = {"Outcome": "Frequency"})
    # second_serves_return_made["ServeFrom"] = "Total"
    # second_serves_returnmade_split = part1_2[(part1_2.Eval_Type3 == "Return") & ( part1_2.FirstSecond == "Second")&
    #                                          (part1_2.Outcome_WL + "_" +part1_2.ServeReceive_Part+ "_" +part1_2.WinOrMis != "Lost_One_Mistake")].groupby(["Eval_Type3","ServeFrom"])["Outcome"].count().reset_index().rename(columns = {"Outcome": "Frequency"})
    # adding in Lost One Forced

    total_return_made = part1_2[part1_2.MadeReturn == 1].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    total_return_made["ServeFrom"] = "Total"
    total_return_made_split = part1_2[part1_2.MadeReturn == 1].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    first_serves_return_made = \
    part1_2[(part1_2.MadeReturn == 1) & (part1_2.FirstSecond == "First")].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    first_serves_return_made["ServeFrom"] = "Total"
    first_serves_returnmade_split = \
    part1_2[(part1_2.MadeReturn == 1) & (part1_2.FirstSecond == "First")].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    second_serves_return_made = \
    part1_2[(part1_2.MadeReturn == 1) & (part1_2.FirstSecond == "Second")].groupby(["Eval_Type3"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    second_serves_return_made["ServeFrom"] = "Total"
    second_serves_returnmade_split = \
    part1_2[(part1_2.MadeReturn == 1) & (part1_2.FirstSecond == "Second")].groupby(["Eval_Type3", "ServeFrom"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    second_serves_returnmade1 = pd.concat([second_serves_returnmade_split, second_serves_return_made], axis=0)
    first_serves_returnmade1 = pd.concat([first_serves_returnmade_split, first_serves_return_made], axis=0)
    total_servesreturnmdae_1 = pd.concat([total_return_made_split, total_return_made], axis=0)

    second_serves_returnmade1[
        "Label"] = second_serves_returnmade1.Eval_Type3 + "_" + second_serves_returnmade1.ServeFrom

    second_serves_returnmade1["Label_0"] = "SecondServesReturnMade_by_ServeFrom_Freq"

    first_serves_returnmade1["Label"] = first_serves_returnmade1.Eval_Type3 + "_" + first_serves_returnmade1.ServeFrom

    first_serves_returnmade1["Label_0"] = "FirstServesReturnMade_by_ServeFrom_Freq"

    total_servesreturnmdae_1["Label"] = total_servesreturnmdae_1.Eval_Type3 + "_" + total_servesreturnmdae_1.ServeFrom

    total_servesreturnmdae_1["Label_0"] = "TotalServesReturnMade_by_ServeFrom_Freq"

    # add critical elements and with filters to this
    # format in a way that can add to the eval table
    # RETURNS
    # need total critical and non critical points
    returns_critnon = part1_2[part1_2.Eval_Type3 == "Return"].groupby(["Eval_Type3", "CriticalPoint"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returns_critnon["Label"] = returns_critnon.Eval_Type3 + "_" + returns_critnon.CriticalPoint
    returns_critnon["Label_0"] = "TotalReturns_by_CriticalHL"

    # critical and non critical - first serve
    returnsfirst_critnon = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "CriticalPoint"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnsfirst_critnon["Label"] = returnsfirst_critnon.Eval_Type3 + "_" + returnsfirst_critnon.CriticalPoint
    returnsfirst_critnon["Label_0"] = "FirstServeReturn_by_CriticalHL"
    # critical and non critical - first serve - deuce ad

    returnsfirst_critnon_from = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnsfirst_critnon_from[
        "Label"] = returnsfirst_critnon_from.Eval_Type3 + "_" + returnsfirst_critnon_from.CriticalPoint + "_" + returnsfirst_critnon_from.ServeFrom

    returnsfirst_critnon_from["Label_0"] = "FirstServeReturn_by_CriticalHL_Origin"

    # critical and non critical - second serve
    returnssecond_critnon = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "CriticalPoint"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnssecond_critnon["Label"] = returnssecond_critnon.Eval_Type3 + "_" + returnssecond_critnon.CriticalPoint
    returnssecond_critnon["Label_0"] = "SecondServeReturn_by_CriticalHL"

    # critical and non critical - second serve - deuce ad
    returnssecond_critnon_from = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnssecond_critnon_from[
        "Label"] = returnssecond_critnon_from.Eval_Type3 + "_" + returnssecond_critnon_from.CriticalPoint + "_" + returnssecond_critnon_from.ServeFrom
    returnssecond_critnon_from["Label_0"] = "SecondServeReturn_by_CriticalHL_Origin"

    # outcomes of
    # critical and non critical - first serve - outcomes
    returnsfirst_critnon_outcomes = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnsfirst_critnon_outcomes[
        "Label"] = returnsfirst_critnon_outcomes.Eval_Type3 + "_" + returnsfirst_critnon_outcomes.CriticalPoint + "_" + returnsfirst_critnon_outcomes.Outcome_WL
    returnsfirst_critnon_outcomes["Label_0"] = "FirstServeReturn_by_CriticalHL_OutcomeGen"

    # critical and non critical - first serve - deuce ad outcmes
    returnsfirst_critnon_from_outcomes = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnsfirst_critnon_from_outcomes[
        "Label"] = returnsfirst_critnon_from_outcomes.Eval_Type3 + "_" + returnsfirst_critnon_from_outcomes.CriticalPoint + "_" + returnsfirst_critnon_from_outcomes.ServeFrom + "_" + returnsfirst_critnon_from_outcomes.Outcome_WL
    returnsfirst_critnon_from_outcomes["Label_0"] = "FirstServeReturn_by_CriticalHL_Origin_OutcomeGen"

    # critical and non critical - second serve outcomes
    returnssecond_critnon_outcome = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnssecond_critnon_outcome[
        "Label"] = returnssecond_critnon_outcome.Eval_Type3 + "_" + returnssecond_critnon_outcome.CriticalPoint + "_" + returnssecond_critnon_outcome.Outcome_WL
    returnssecond_critnon_outcome["Label_0"] = "SecondServeReturn_by_CriticalHL_OutcomeGen"
    # critical and non critical - second serve - deuce ad outcomes
    returnssecond_critnon_from_outcome = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnssecond_critnon_from_outcome[
        "Label"] = returnssecond_critnon_from_outcome.Eval_Type3 + "_" + returnssecond_critnon_from_outcome.CriticalPoint + "_" + returnssecond_critnon_from_outcome.ServeFrom + "_" + returnssecond_critnon_from_outcome.Outcome_WL
    returnssecond_critnon_from_outcome["Label_0"] = "SecondServeReturn_by_CriticalHL_Origin_OutcomeGen"

    # critical and non critical - first serve
    returnsfirst_critnon_made = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First") & (part1_2.MadeReturn == 1)].groupby(
        ["Eval_Type3", "CriticalPoint"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnsfirst_critnon_made[
        "Label"] = returnsfirst_critnon_made.Eval_Type3 + "_" + returnsfirst_critnon_made.CriticalPoint
    returnsfirst_critnon_made["Label_0"] = "FirstServeReturnMADE_by_CriticalHL"

    # critical and non critical - second serve
    returnssecond_critnon_made = \
    part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second") & (part1_2.MadeReturn == 1)].groupby(
        ["Eval_Type3", "CriticalPoint"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnssecond_critnon_made[
        "Label"] = returnssecond_critnon_made.Eval_Type3 + "_" + returnssecond_critnon_made.CriticalPoint
    returnssecond_critnon_made["Label_0"] = "SecondServeReturnMADE_by_CriticalHL"

    # breakballs & non breakballs split - filter on critical only
    # RETURNS
    # need total critical and non critical points
    returns_critbreak = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.CriticalPoint == "Critical")].groupby(
        ["Eval_Type3", "CriticalPoint_Lv2"])["Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returns_critbreak["Label"] = returns_critbreak.Eval_Type3 + "_" + returns_critbreak.CriticalPoint_Lv2
    returns_critbreak["Label_0"] = "TotalReturns_by_Critical_lv2"

    # critical and non critical - first serve
    returnsfirst_critbreak = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnsfirst_critbreak["Label"] = returnsfirst_critbreak.Eval_Type3 + "_" + returnsfirst_critbreak.CriticalPoint_Lv2
    returnsfirst_critbreak["Label_0"] = "FirstServeReturns_by_Critical_lv2"
    # critical and non critical - first serve - deuce ad

    returnsfirst_critbreak_from = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnsfirst_critbreak_from[
        "Label"] = returnsfirst_critbreak_from.Eval_Type3 + "_" + returnsfirst_critbreak_from.CriticalPoint_Lv2 + "_" + returnsfirst_critbreak_from.ServeFrom

    returnsfirst_critbreak_from["Label_0"] = "FirstServeReturn_by_Critical_lv2_Origin"

    # critical and non critical - second serve
    returnssecond_critbreak = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnssecond_critbreak[
        "Label"] = returnssecond_critbreak.Eval_Type3 + "_" + returnssecond_critbreak.CriticalPoint_Lv2
    returnssecond_critbreak["Label_0"] = "SecondServeReturn_by_Critical_lv2"

    # critical and non critical - second serve - deuce ad
    returnssecond_critbreak_from = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnssecond_critbreak_from[
        "Label"] = returnssecond_critbreak_from.Eval_Type3 + "_" + returnssecond_critbreak_from.CriticalPoint_Lv2 + "_" + returnssecond_critbreak_from.ServeFrom
    returnssecond_critbreak_from["Label_0"] = "SecondServeReturn_by_Critical_lv2_Origin"

    # outcomes of
    # critical and non critical - first serve - outcomes
    returnfirst_critbreak_outcomes = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2", "Outcome_WL"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnfirst_critbreak_outcomes[
        "Label"] = returnfirst_critbreak_outcomes.Eval_Type3 + "_" + returnfirst_critbreak_outcomes.CriticalPoint_Lv2 + "_" + returnfirst_critbreak_outcomes.Outcome_WL
    returnfirst_critbreak_outcomes["Label_0"] = "FirstServeReturn_by_Critical_lv2_OutcomeGen"

    # critical and non critical - first serve - deuce ad outcmes
    returnfirst_critbreak_from_outcomes = part1_2[
        (part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "First") & (
                    part1_2.CriticalPoint == "Critical")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnfirst_critbreak_from_outcomes[
        "Label"] = returnfirst_critbreak_from_outcomes.Eval_Type3 + "_" + returnfirst_critbreak_from_outcomes.CriticalPoint_Lv2 + "_" + returnfirst_critbreak_from_outcomes.ServeFrom + "_" + returnfirst_critbreak_from_outcomes.Outcome_WL
    returnfirst_critbreak_from_outcomes["Label_0"] = "FirstServeReturn_by_Critical_lv2_Origin_OutcomeGen"

    # critical and non critical - second serve outcomes
    returnsecond_critbreak_outcome = part1_2[(part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second") & (
                part1_2.CriticalPoint == "Critical")].groupby(["Eval_Type3", "CriticalPoint_Lv2", "Outcome_WL"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    returnsecond_critbreak_outcome[
        "Label"] = returnsecond_critbreak_outcome.Eval_Type3 + "_" + returnsecond_critbreak_outcome.CriticalPoint_Lv2 + "_" + returnsecond_critbreak_outcome.Outcome_WL
    returnsecond_critbreak_outcome["Label_0"] = "SecondServeReturn_by_Critical_lv2_OutcomeGen"
    # critical and non critical - second serve - deuce ad outcomes
    returnsecond_critbreak_from_outcome = part1_2[
        (part1_2.Eval_Type3 == "Return") & (part1_2.FirstSecond == "Second") & (
                    part1_2.CriticalPoint == "Critical")].groupby(
        ["Eval_Type3", "ServeFrom", "CriticalPoint_Lv2", "Outcome_WL"])["Outcome"].count().reset_index().rename(
        columns={"Outcome": "Frequency"})
    returnsecond_critbreak_from_outcome[
        "Label"] = returnsecond_critbreak_from_outcome.Eval_Type3 + "_" + returnsecond_critbreak_from_outcome.CriticalPoint_Lv2 + "_" + returnsecond_critbreak_from_outcome.ServeFrom + "_" + returnsecond_critbreak_from_outcome.Outcome_WL
    returnsecond_critbreak_from_outcome["Label_0"] = "SecondServeReturn_by_Critical_lv2_Origin_OutcomeGen"

    # rally outcomes
    # means I need this on a point basis
    # use number of shots in point (NumShotsinPt)
    # 3 or more, rally label & WonLost
    # split and double the rally length - just labelling - can change later
    # 2 or less Eval_Type2 & WonLost
    shots5["RallyLengths"] = np.where(shots5.NumShotsinPt <= 2, shots5.Eval_Type2 + "_" + shots5.WonLost,
                                    np.where(shots5.NumShotsinPt <= 4, "Rally_3to4_shots_" + shots5.WonLost,
                                            "Rally_over4_shots_" + shots5.WonLost))
    # drop duplicates (game_point) and groupby
    rallylengths = shots5.drop_duplicates("Game_Point", keep="first").groupby("RallyLengths")[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    rallylengths["Label"] = "RallyLen_" + rallylengths.RallyLengths
    rallylengths["Label_0"] = "RallyLen_Breakdown"

    # topline stats
    # number won
    # number lost
    # how won / lost
    WonLost = shots5.drop_duplicates("Game_Point", keep="first").groupby(["WonLost"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})
    WonLost["Label"] = WonLost.WonLost
    WonLost["Label_0"] = "TotalPointsWonOrLost_Freq"

    WonLost_details = shots5.drop_duplicates("Game_Point", keep="first").groupby(["Win_Extra"])[
        "Outcome"].count().reset_index().rename(columns={"Outcome": "Frequency"})

    WonLost_details["Label"] = WonLost_details.Win_Extra
    WonLost_details["Label_0"] = "TotalPointsWonOrLost_Freq"

    # create the HR data
    # raw2 = raw[raw.Seconds > start]
    raw2["Seconds2"] = round(raw2.Seconds, 0)
    raw3 = raw2.groupby("Seconds2")["heartRate"].max().reset_index()
    
    age = (int(meta_id[:4]) - int(bornYear))
    if gender == "MALE":
        maxhr = 220 - age
    if gender == "FEMALE":
        maxhr = 206 - (0.88 * age)

    raw3["HR_Zone"] = np.where(raw3.heartRate >= (maxhr * 0.9), "Max Effort 90%+",
                            np.where(raw3.heartRate >= (maxhr * 0.8), "Hard Effort 80-90%",
                                        np.where(raw3.heartRate >= (maxhr * 0.7), "Moderate Effort 70-80%",
                                                np.where(raw3.heartRate >= (maxhr * 0.6), "Light Effort 60-70%",
                                                        "Very Light <60%"
                                                        ))))
    HRtime = raw3.groupby("HR_Zone")["Seconds2"].count().reset_index().reset_index().rename(
        columns={"Seconds2": "Frequency"})
    HRtime["Label"] = HRtime.HR_Zone
    HRtime["Label_0"] = "HRTimeSpentInZone"

    # merge the hr & zone at the max (end) of the point and calculate the grouped win and error rates

    pts["Seconds2"] = round(pts.maximum, 0)
    pts2 = pd.merge(pts, raw3, how="left", on="Seconds2")
    pts2["Game_Point"] = pts2.GameCount.astype("str") + "_" + pts2.PointInGame.astype("str")
    pts2 = pts2.drop_duplicates("Game_Point", keep="first")[pts2.WinOrMis.notnull()]

    winrate_HR = pts2[pts2.WonLost == "Won"].groupby(["HR_Zone", "WonLost"])["minimum"].count().reset_index().rename(
        columns={"minimum": "Frequency"})
    winrate_HR["Label"] = winrate_HR.HR_Zone + "_" + winrate_HR.WonLost
    winrate_HR["Label_0"] = "Wins_byHRZone_Freq"

    errorrate_HR = pts2[pts2.Win_Extra == "Lost_thru_Mistake"].groupby(["HR_Zone", "Win_Extra"])[
        "minimum"].count().reset_index().rename(columns={"minimum": "Frequency"})
    errorrate_HR["Label"] = errorrate_HR.HR_Zone + "_" + errorrate_HR.Win_Extra
    errorrate_HR["Label_0"] = "Mistakes_byHRZone_Freq"

    ptsby_HR = pts2[pts2.WonLost.isin(["Won", "Lost"])].groupby(["HR_Zone"])["minimum"].count().reset_index().rename(
        columns={"minimum": "Frequency"})
    ptsby_HR["Label"] = ptsby_HR.HR_Zone + "_TotalPts"
    ptsby_HR["Label_0"] = "TotalPts_byHRZone_Freq"

    shots5["Shot_Breakdown"] = np.where(shots5.Eval_Type4 == "", np.where(shots5.Eval_Type2 == "", shots5.RealShot,
                                                                        shots5.Eval_Type2), shots5.Eval_Type4)
    shotcounts = shots5.Shot_Breakdown.value_counts().reset_index().rename(
        columns={"Shot_Breakdown": "Frequency", "index": "Shot"})
    shotcounts["Label"] = shotcounts.Shot + "_Freq"
    shotcounts["Label_0"] = "ShotBreakdown_Frequency"

    # shotcounts

    blnks = list(shots5.Game_Point[shots5.Shot_Breakdown == ""])
    shots5[shots5.Game_Point.isin(blnks)][['Game_Point', 'Eval_Type',

                                        'Eval_Type2', 'Outcome', 'Eval_Type3', 'Outcome_WL',
                                        'ServeReceive_Part', 'Eval_Type4', 'Eval_Type5',
                                        'RallyLengths', "RealShot",
                                        'Shot_Breakdown']]
    metaToSum = pd.DataFrame({"Label_0": ["age", "gender","rating_lev", "rating_typ", "matchResult", "matchLevel", "matchSurface", "matchType"],
                                "Label":[age, gender,rating_lev, rating_typ, matchResult, matchLevel, matchSurface, matchType],
                                "Frequency": [0,0,0,0,0,0,0,0],
                                "Effective_1":[0,0,0,0,0,0,0,0]})
    
    # ### Creating output for video labels

    # 	1. From shots5, take Game_Point - de dup - first
    vid = shots5[["Game_Point", "TimeTrueStrike", "Win_Extra", "FirstSecond", "OnServe_Corrected",
                "RallyLengths"]].drop_duplicates("Game_Point", keep="first")
    # 		a. Inc TrueStrikeTime, Win_Extra, First_Second, OnServe_Corrected
    # 	2. From Shots5, take game_point, dedup - keep last
    vid_end = shots5[["Game_Point", "TimeTrueStrike"]].drop_duplicates("Game_Point", keep="last").rename(
        columns={"TimeTrueStrike": "PointEndTime"})
    vid2 = pd.merge(vid, vid_end, how="left", on="Game_Point")

    # 	3. Adjust times - start
    # 		a. Onserve - 1
    # 		b. Return - 3
    # 	4. Adjust times - end
    # 		a. General + 3
    vid2["Gen_Start"] = np.where(vid2.OnServe_Corrected == 1, vid2.TimeTrueStrike - 1, vid2.TimeTrueStrike - 3)
    vid2["Gen_End"] = vid2.PointEndTime + 3
    # 	5. For serving & return, take shots5  ShotInGame == 2, merge time ( rename)
    servereturnends = shots5[shots5.ShotInGame == 2][["Game_Point", "TimeTrueStrike"]].rename(
        columns={"TimeTrueStrike": "EndServeReturnTime"})
    vid3 = pd.merge(vid2, servereturnends, on="Game_Point", how="left")
    vid3["Type"] = vid3.RallyLengths.str[:6]

    # 	6. Where missing - start +5, else fin +5
    # 	7. Serve Return start same as adjusted start above
    vid3["Serve_Return_End"] = np.where(vid3.EndServeReturnTime.isnull(), vid3.Gen_Start + 5,
                                        vid3.EndServeReturnTime + 5)

    # 	8. Rally - shot5 = rally - keep first - merge time
    rallystarts = shots5[shots5.StartOrRally == "Rally"][["Game_Point", "TimeTrueStrike"]].rename(
        columns={"TimeTrueStrike": "RallyStartTime"})
    vid4 = pd.merge(vid3, rallystarts, on="Game_Point", how="left")
    vid4["Rally_Start"] = vid4.RallyStartTime - 5

    from datetime import datetime
    # adding vid timings to json file
    timings = vid4
    
    # meta_raw = glob.glob(f"{base}{basevid}/{who}_{what}/Session*json")
    # meta = pd.read_json(meta_raw[0], typ = "series")
    # meta = meta_out.copy()
    SessionStart_TimeStamp = meta_id
    SessionStart_TimeStamp2 = SessionStart_TimeStamp[:-5]

    dt = datetime.fromisoformat(SessionStart_TimeStamp2)
    # to UNIX time:
    ts = dt.timestamp()
    adj = 0
    timings = timings.drop_duplicates("Game_Point", keep="first")
    timings["Gen_Start_adj"] = (timings.Gen_Start - adj).astype("str")
    timings["Gen_End_adj"] = (timings.Gen_End - adj).astype("str")
    timings["Serve_Return_End_adj"] = (timings.Serve_Return_End - adj).astype("str")
    timings["Rally_Start_adj"] = (timings.Rally_Start - adj).astype("str")
    # print(timings)
    winners = timings[timings.Win_Extra.isin(["Won_thru_Forced", "Won_thru_Winner"])][
        ["Game_Point", "Gen_Start_adj", "Gen_End_adj"]]
    win_tuples = list(winners.itertuples(index=False, name=None))

    serve_first = timings[(timings.FirstSecond == "First") & (timings.Type == "Serve_")][
        ["Game_Point", "Gen_Start_adj", "Serve_Return_End_adj"]]
    serve_first_tuple = list(serve_first.itertuples(index=False, name=None))

    serve_second = timings[(timings.FirstSecond == "Second") & (timings.Type == "Serve_")][
        ["Game_Point", "Gen_Start_adj", "Serve_Return_End_adj"]]
    serve_second_tuple = list(serve_second.itertuples(index=False, name=None))

    return_first = timings[(timings.FirstSecond == "First") & (timings.Type == "Return")][
        ["Game_Point", "Gen_Start_adj", "Serve_Return_End_adj"]]
    return_first_tuple = list(return_first.itertuples(index=False, name=None))

    return_second = timings[(timings.FirstSecond == "Second") & (timings.Type == "Return")][
        ["Game_Point", "Gen_Start_adj", "Serve_Return_End_adj"]]
    return_second_tuple = list(return_second.itertuples(index=False, name=None))

    rallies = timings[(timings.Type == "Rally_") & (timings.Rally_Start_adj != "nan")][["Game_Point", "Rally_Start_adj", "Gen_End_adj"]]
    rallies_tuple = list(rallies.itertuples(index=False, name=None))
    # print(rallies.iloc[:,2])
    def find_nans(d_in, cut):
        if len(d_in[d_in.iloc[:, 1].str.contains("nan")]) != 0:
            num = len(d_in[d_in.iloc[:, 1].str.contains("nan")])
            # d_in = d_in.replace("nan","0")
            print(f"Found {num} nans in {cut}!")
            print(d_in[d_in.iloc[:, 1].str.contains("nan")])

    find_nans(rallies, "rallies")
    find_nans(serve_first, "serve_first")
    find_nans(serve_second, "serve_second")
    find_nans(return_first, "return_first")
    find_nans(return_second, "return_second")
    find_nans(winners, "winners")

    video_json = {
        "sessionStartTimestamp": ts,
        "processedVideos": [
            {
                "id": "47260756-c17a-11ed-afa1-0242ac120001",
                "videoType": "1st Serve",
                "timeframes": serve_first_tuple
            },
            {
                "id": "4d5c5dd2-c17a-11ed-afa1-0242ac120002",
                "videoType": "2nd Serve",
                "timeframes": serve_second_tuple
            },
            {
                "id": "4d5c5dd2-c17a-11ed-afa1-0242ac120003",
                "videoType": "1st Return",
                "timeframes": return_first_tuple
            },
            {
                "id": "4d5c5dd2-c17a-11ed-afa1-0242ac120004",
                "videoType": "2nd Return",
                "timeframes": return_second_tuple
            },
            {
                "id": "4d5c5dd2-c17a-11ed-afa1-0242ac120005",
                "videoType": "Rallies",
                "timeframes": rallies_tuple
            },
            {
                "id": "4d5c5dd2-c17a-11ed-afa1-0242ac120006",
                "videoType": "Winners & Forced Errors",
                "timeframes": win_tuples
            }
        ]
    }

    missingtable = pd.DataFrame({'Label_0': ["Missing", "Missing"], 'Label': ["NoSensor", "NoShots"], 'Frequency': [missing_sensor, 0], 'Effective_1': [0, 0]})

    eval_table = pd.concat([
        WonLost,
        WonLost_details,
        effective_score,
        effective_score_first,
        effective_score_second,
        effective_score_deuce,
        effective_score_ad,
        effective_score_deuce_first,
        effective_score_deuce_second,
        effective_score_ad_first,
        effective_score_ad_second,
        WL_Generic_OutcomeGen_Freq,
        WL_Generic_OutcomeGen_Freq_First,
        WL_Generic_OutcomeGen_Freq_Second,
        WL_Generic_OutcomeGen_Freq_Deuce,
        WL_Generic_OutcomeGen_Freq_Ad,
        WL_Generic_OutcomeGen_Freq_Deuce_First,
        WL_Generic_OutcomeGen_Freq_Deuce_Second,
        WL_Generic_OutcomeGen_Freq_Ad_First,
        WL_Generic_OutcomeGen_Freq_Ad_Second,

        WL_Outcome_Eff,
        WL_Outcome_Eff_First,
        WL_Outcome_Eff_Second,
        WL_Outcome_Eff_Deuce,
        WL_Outcome_Eff_Ad,
        WL_Outcome_Eff_Deuce_First,
        WL_Outcome_Eff_Deuce_Second,
        WL_Outcome_Eff_Ad_First,
        WL_Outcome_Eff_Ad_Second,

        effective_score_shots,
        Shotslower_OutcomeGen_Freq,
        Shotslower_Outcome_Eff,

        WL_Generic_Part_OutcomeGen_Freq,
        WL_Generic_Part_Outcome_Freq,
        WL_Generic_Part_Outcome_Shot_Freq,

        # adding split of first & second
        WL_FirstSecond_OutcomeGen_Freq,
        WL_FirstSecond_Part_Outcome_Freq,
        WL_FirstSecond_Part_Outcome_Shot_Freq,
        # adding serve where from
        WL_FirstSecond_From_OutcomeGen_Freq,
        WL_FirstSecond_From_Part_Outcome_Freq,
        WL_FirstSecond_From_Part_Outcome_Shot_Freq,

        # adding in win or loss in Rally based on how starts
        rally_outcomes_from_start,
        # adding ser rates and splits by critical
        WL_FirstSecond_From_OutcomeGen,
        WL_FirstSecond_From_Outcome,
        first_serves_1,
        total_serves_1,
        serves_critnon,
        servesfirst_critbreak,
        servesfirst_critnon,
        servesfirst_critnon_from,
        servessecond_critnon,
        servessecond_critnon_from,
        servesfirst_critnon_outcomes,

        servesfirst_critnon_from_outcomes,
        servessecond_critnon_outcome,

        servessecond_critnon_from_outcome,
        # adding critical breakpoints,
        serves_critbreak,
        servesfirst_critbreak_from,
        servessecond_critbreak,
        servessecond_critbreak_from,

        servesfirst_critbreak_outcomes,
        servesfirst_critbreak_from_outcomes,
        servessecond_critbreak_outcome,
        servessecond_critbreak_from_outcome,
        # adding return rates
        total_servesreturn_1,
        first_serves_return1,
        second_serves_return1,

        total_servesreturnmdae_1,
        first_serves_returnmade1,
        second_serves_returnmade1,
        # return critical points,
        returns_critnon,

        returnsfirst_critnon,
        returnsfirst_critnon_made,
        returnsfirst_critnon_from,

        returnssecond_critnon,
        returnssecond_critnon_made,
        returnssecond_critnon_from,

        returnsfirst_critnon_outcomes,
        returnsfirst_critnon_from_outcomes,

        returnssecond_critnon_outcome,
        returnssecond_critnon_from_outcome,
        # return critcal lv2 break balls
        returns_critbreak,

        returnsfirst_critbreak,
        returnsfirst_critbreak_from,

        returnssecond_critbreak,
        returnssecond_critbreak_from,

        returnfirst_critbreak_outcomes,
        returnfirst_critbreak_from_outcomes,

        returnsecond_critbreak_outcome,
        returnsecond_critbreak_from_outcome,

        # returncounts added
        returns,
        returns_1st,
        returns_2nd,
        # rallylen
        rallylengths,
        # plus average rally length by serve & return
        average_rally,
        # shotbreakdown
        shotcounts,
        # servereturn_effectiveness added (includes frequency of points)
        servereturn_shoteff,
        servereturn_shoteff_1st,
        servereturn_shoteff_2nd,
        # HR stats
        HRtime,
        winrate_HR,
        errorrate_HR,
        ptsby_HR,
        missingtable,
        metaToSum
    ], axis=0)

    eval_table_out = eval_table[["Label_0", "Label", "Frequency", "Effective_1"]].reset_index(drop=True)
    return eval_table_out