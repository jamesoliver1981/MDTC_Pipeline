import pandas as pd
import numpy as np

def create_touch_data(motion, points_start_end_4):
    # adding elements of points been pressed from others
    # want a won or a lost point
    motion = motion[motion.Seconds.notnull()]
    alltouch = motion[(motion.won == 1) | (motion.lost == 1)][["Seconds", "won", "lost"]]
    
    #recalculate seconds here to factor in the 3 second delay factor
    alltouch["Seconds"] = np.where(alltouch.Seconds<3, 2,alltouch.Seconds -3)

    alltouch["Dups"] = np.where((alltouch.Seconds.shift(-1) - alltouch.Seconds) < 3, 1, 0)
    touch = alltouch[alltouch.Dups == 0]
    # then attach the point detail from points_start - but only that previous
    touch1 = pd.merge_asof(touch, points_start_end_4, left_on="Seconds", right_on="minimum", direction="backward")
    touch1["minimum"].fillna(0, inplace=True)
    # then attach second information - forwards & backwards
    seconds = motion[motion.secondServe == 1][["Seconds", "secondServe"]].rename(
        columns={"Seconds": "secondPress_back"})

    touch2 = pd.merge_asof(touch1, seconds, left_on="Seconds", right_on="secondPress_back", direction="backward")
    seconds.rename(columns={"secondPress_back": "secondPress_forw", "secondServe": "secondServe_forw"}, inplace=True)
    touch2_2 = pd.merge_asof(touch2, seconds, left_on="Seconds", right_on="secondPress_forw", direction="forward")
    touch2_2["SecondsShiftOneDown"] = touch2_2.Seconds.shift(1)
    touch2_2["SecondsShiftOneUp"] = touch2_2.Seconds.shift(-1)
    # then use the time stamps to decide how to align the second serve data

    # my actions could be,
    # I press second and immediately won or lost
    # means need secondPressback and its timestamp must be greater than SecondShiftOneDown
    # I press won or lost, and then second
    # means need secondPress_forw and its timestamp must be less than SecondShiftOneUp
    # I press second between first & second serve -
    # means I need secondPress_back - only if this is greater than Seconds_ShiftOne
    # there is no need to press second

    # this still needs to be enhanced for press between serves - currently just immediate
    touch2_2["RealSecondServe"] = np.where(((touch2_2.Seconds > touch2_2.secondPress_back) & (
                touch2_2.secondPress_back > touch2_2.SecondsShiftOneDown) & (
                                                        (touch2_2.Seconds - touch2_2.secondPress_back) < 10)) |
                                        ((touch2_2.Seconds < touch2_2.secondPress_forw) & (
                                                    touch2_2.secondPress_forw < touch2_2.SecondsShiftOneUp) & (
                                                        (touch2_2.secondPress_forw - touch2_2.Seconds) < 10)) |
                                        (touch2_2.secondPress_back - touch2_2.SecondsShiftOneDown > 10), 1, 0)

    # create a WonLost column
    touch2_2["WonLost"] = np.where(touch2_2.won == 1, "Won", "Lost")

    # end of game from motion
    endof = motion[motion.endOfGame == 1][["Seconds", "endOfGame"]].drop("endOfGame", axis=1)
    endof["endGameSeconds"] = endof.Seconds

    # endof2 = pd.merge_asof(endof,touch2_2[["Seconds"]], left_on = "endGameSeconds", right_on = "Seconds", direction = "nearest" )

    # create touch3 with the combination of endOfGame
    touch2_3 = pd.merge_asof(touch2_2, endof, left_on="Seconds", right_on="Seconds", direction="nearest")

    touch2_3["eog"] = np.where((touch2_3.endGameSeconds - touch2_3.Seconds) > -7, 1, 0)
    # dedup the end of games
    dup_eog = touch2_3[touch2_3.eog == 1][["Seconds", "endGameSeconds"]]
    dedup_eog = dup_eog.drop_duplicates(subset="endGameSeconds", keep="last").rename(
        columns={"endGameSeconds": "eogSeconds"})

    touch3 = pd.merge(touch2_3, dedup_eog, on="Seconds", how="left")
    touch3["eog_Lab"] = np.where(touch3.eogSeconds.notnull(),
                                np.where((touch3.eogSeconds.shift(1).notnull()) & (
                                            touch3.Seconds - touch3.Seconds.shift(1) < 60), 0, 1), 0)

    # touch3["endGameSeconds2"] = np.where(touch3.RowNum == 0, 0, touch3.endGameSeconds)
    # create other measures which eval if serve changed
    touch3["LongTimeLastPoint"] = np.where(touch3.Seconds.shift(-1) - touch3.Seconds > 60, 1, 0)
    touch3["RollServe_For"] = touch3.FirstIsServe.shift(-3).rolling(3).mean()
    touch3["RollServe_Back"] = touch3.FirstIsServe.rolling(3).mean()
    touch3["ChangeInServe"] = np.where(
        (touch3.RollServe_For - touch3.RollServe_Back < -0.6) | (touch3.RollServe_For - touch3.RollServe_Back > 0.6), 1,
        0)

    touch3 = touch3.reset_index().rename(columns={"index": "RowNum"})
    touch3["EndGameRowNum"] = np.where(touch3.RowNum == 0, 0, np.where(touch3.eog_Lab == 1, touch3.RowNum, np.nan))
    touch3["EndGameRowNum_ffill"] = touch3.EndGameRowNum.ffill()
    touch3["PointsSinceEoG"] = touch3.RowNum - touch3.EndGameRowNum_ffill

    #where prior 3 were not serve and next 3 are serve, (and the opposite), record as change in serve
    touch3["Last3VsNext3"] = np.where((touch3.FirstIsServe == 1) & (touch3.FirstIsServe.shift(1) ==1) & (touch3.FirstIsServe.shift(2) ==1) & 
                                    (touch3.FirstIsServe.shift(-1) == 0) & (touch3.FirstIsServe.shift(-2) == 0) & (touch3.FirstIsServe.shift(-3) == 0), 1,
                                        np.where((touch3.FirstIsServe == 0) & (touch3.FirstIsServe.shift(1) == 0) & (touch3.FirstIsServe.shift(2) == 0) & 
                                    (touch3.FirstIsServe.shift(-1) == 1) & (touch3.FirstIsServe.shift(-2) == 1) & (touch3.FirstIsServe.shift(-3) == 1), 1,0)   )
    # combine these new vars to id where there is a manual change in serve
    touch3["manEOG"] = np.where(
        ((touch3.LongTimeLastPoint == 1) & (touch3.ChangeInServe == 1) & (touch3.PointsSinceEoG > 5) | touch3.Last3VsNext3 == 1), 1, 0)

    touch3["eog_fin"] = np.where((touch3.eog_Lab == 1) | (touch3.manEOG == 1), 1, 0)
    touch3["eog_count"] = touch3.eog_fin.cumsum().ffill().shift(1)
    touch3["eog_count"].fillna(0, inplace=True)

    #adding in Tiebreak logic
        #first how long is a game
    touch3["ChangeInFirstIsServe"] = np.where(touch3.FirstIsServe != touch3.FirstIsServe.shift(1), 1, 0)
    eog_len = touch3.groupby(["eog_count"])["PointsSinceEoG"].max().reset_index().rename(columns = {"PointsSinceEoG" : "LengthGame"})
    
    touch3 = pd.merge(touch3, eog_len, on = "eog_count", how = "left")
        #then how often does serve change in that game
    eog_changes = touch3.groupby(["eog_count"])["ChangeInFirstIsServe"].sum().reset_index().rename(columns = {"ChangeInFirstIsServe" : "CountChangesServe"})
    touch3 = pd.merge(touch3, eog_changes, on = "eog_count", how = "left")

    touch3["Tiebreak"] = np.where((touch3.LengthGame >= 7) & (touch3.CountChangesServe >= 4),1, 0)

    # touch3["eog_time"] = np.where(touch3.eog_fin == 1, touch3.RowNum, np.nan)

    serving = touch3.groupby(["eog_count"])["FirstIsServe"].mean().reset_index().rename(
        columns={"FirstIsServe": "average"})

    #renaming so its just OnServe and OnServe_Corrected can factor in Tiebreak
    # serving["OnServe_Corrected"] = np.where(serving.average > 0.5, 1, 0)
    serving["OnServe"] = np.where(serving.average > 0.5, 1, 0)

    # touch3["endGame_ffill"] = touch3.eogSeconds.ffill()
    # touch3["PointsSinceEOG"] = np.where(touch3.eog_Lab == 1, 1, touch3.eog_Lab + 1, touch3.eog_Lab)
    # print(touch3[touch3.endOfGame == 1])

    #renaming so its just OnServe and OnServe_Corrected can factor in Tiebreak
    # touch3 = pd.merge(touch3, serving[["eog_count", "OnServe_Corrected"]], on="eog_count", how="left")
    touch3 = pd.merge(touch3, serving[["eog_count", "OnServe"]], on="eog_count", how="left")
    touch3["OnServe_Corrected"] = np.where(touch3.Tiebreak == 1, touch3.FirstIsServe, touch3.OnServe)

    # create an iterative function that figures out the score...and assigns values - and an if within that...
        # add if tiebreak ends, its a change in serve too
    # touch3["ChangeServe"] = np.where(touch3.OnServe_Corrected != touch3.OnServe_Corrected.shift(1), 1, 0)
    touch3["ChangeServe"] = np.where((touch3.OnServe_Corrected != touch3.OnServe_Corrected.shift(1)) | (touch3.Tiebreak != touch3.Tiebreak.shift(1)), 1, 0)
    # know when change, so that has to be Five Zero or Zero Five

    touch3["S_Score"] = np.where(touch3.ChangeServe == 1,
                                np.where(touch3.FirstIsServe == 1, np.where(touch3.WonLost == "Won", "Five", "Zero"),
                                        np.where(touch3.WonLost == "Lost", "Five", "Zero")), "")

    touch3["R_Score"] = np.where(touch3.ChangeServe == 1,
                                np.where(touch3.FirstIsServe == 1, np.where(touch3.WonLost == "Won", "Zero", "Five"),
                                        np.where(touch3.WonLost == "Lost", "Zero", "Five")), "")
    touch3["Score"] = touch3.S_Score + "_" + touch3.R_Score

    d = {
        ('Won', 0, 'Five_Zero'): "Five_Five",
        ('Won', 1, 'Five_Zero'): "Three_Zero",
        ('Lost', 0, 'Five_Zero'): "Three_Zero",
        ('Lost', 1, 'Five_Zero'): "Five_Five",

        ('Won', 0, 'Five_Five'): "Five_Three",
        ('Lost', 0, 'Five_Five'): "Three_Five",
        ('Won', 1, 'Five_Five'): "Three_Five",
        ('Lost', 1, 'Five_Five'): "Five_Three",

        ('Won', 0, 'Three_Zero'): "Three_Five",
        ('Won', 1, 'Three_Zero'): "Four_Zero",
        ('Lost', 0, 'Three_Zero'): "Four_Zero",
        ('Lost', 1, 'Three_Zero'): "Three_Five",

        ('Won', 0, 'Three_Five'): "Three_Three",
        ('Won', 1, 'Three_Five'): "Four_Five",
        ('Lost', 0, 'Three_Five'): "Four_Five",
        ('Lost', 1, 'Three_Five'): "Three_Three",

        ('Won', 0, 'Five_Three'): "Five_Four",
        ('Won', 1, 'Five_Three'): "Three_Three",
        ('Lost', 0, 'Five_Three'): "Three_Three",
        ('Lost', 1, 'Five_Three'): "Five_Four",

        ('Won', 0, 'Three_Three'): "Three_Four",
        ('Won', 1, 'Three_Three'): "Four_Three",
        ('Lost', 0, 'Three_Three'): "Four_Three",
        ('Lost', 1, 'Three_Three'): "Three_Four",

        ('Won', 0, 'Three_Four'): "Game",
        ('Won', 1, 'Three_Four'): "Deuce_Deuce",
        ('Lost', 0, 'Three_Four'): "Deuce_Deuce",
        ('Lost', 1, 'Three_Four'): "Game",

        ('Won', 0, 'Zero_Five'): "Zero_Three",
        ('Won', 1, 'Zero_Five'): "Five_Five",
        ('Lost', 0, 'Zero_Five'): "Five_Five",
        ('Lost', 1, 'Zero_Five'): "Zero_Three",

        ('Won', 0, 'Zero_Three'): "Zero_Four",
        ('Won', 1, 'Zero_Three'): "Five_Three",
        ('Lost', 0, 'Zero_Three'): "Five_Three",
        ('Lost', 1, 'Zero_Three'): "Zero_Four",

        ('Won', 0, 'Four_Three'): "Deuce_Deuce",
        ('Won', 1, 'Four_Three'): "Game",
        ('Lost', 0, 'Four_Three'): "Game",
        ('Lost', 1, 'Four_Three'): "Deuce_Deuce",

        ('Won', 0, 'Four_Zero'): "Four_Five",
        ('Won', 1, 'Four_Zero'): "Game",
        ('Lost', 0, 'Four_Zero'): "Game",
        ('Lost', 1, 'Four_Zero'): "Four_Five",

        ('Won', 0, 'Four_Five'): "Four_Three",
        ('Won', 1, 'Four_Five'): "Game",
        ('Lost', 0, 'Four_Five'): "Game",
        ('Lost', 1, 'Four_Five'): "Four_Three",

        ('Won', 0, 'Zero_Four'): "Game",
        ('Won', 1, 'Zero_Four'): "Five_Four",
        ('Lost', 0, 'Zero_Four'): "Five_Four",
        ('Lost', 1, 'Zero_Four'): "Game",

        ('Won', 0, 'Five_Four'): "Game",
        ('Won', 1, 'Five_Four'): "Three_Four",
        ('Lost', 0, 'Five_Four'): "Three_Four",
        ('Lost', 1, 'Five_Four'): "Game",

        ('Won', 0, 'Deuce_Deuce'): "AdvRet_AdvRet",
        ('Won', 1, 'Deuce_Deuce'): "AdvServe_AdvServe",
        ('Lost', 0, 'Deuce_Deuce'): "AdvServe_AdvServe",
        ('Lost', 1, 'Deuce_Deuce'): "AdvRet_AdvRet",

        ('Won', 0, 'AdvRet_AdvRet'): "Game",
        ('Won', 1, 'AdvRet_AdvRet'): "Deuce_Deuce",
        ('Lost', 0, 'AdvRet_AdvRet'): "Deuce_Deuce",
        ('Lost', 1, 'AdvRet_AdvRet'): "Game",

        ('Won', 0, 'AdvServe_AdvServe'): "Deuce_Deuce",
        ('Won', 1, 'AdvServe_AdvServe'): "Game",
        ('Lost', 0, 'AdvServe_AdvServe'): "Game",
        ('Lost', 1, 'AdvServe_AdvServe'): "Deuce_Deuce",
    }

    for i in range(len(touch3.Seconds)):
        if touch3.Score[i] not in (["Five_Zero", "Zero_Five"]):
            touch3.Score[i] = d.get((touch3.WonLost[i], touch3.OnServe_Corrected[i], touch3.Score[i - 1]), "_")

    touch3[["Server_Score", "Returner_Score"]] = touch3.Score.str.split("_", 1, expand=True)

    touch3["WinOrMis"] = np.where(touch3.WonLost == "Won", "Winner", "Mistake")

    touch3["Win_Extra"] = touch3.WonLost + "_thru_" + touch3.WinOrMis
    touch3["Score_PI_Fill2"] = np.where(touch3.Server_Score == "Game", "Game",
                                        touch3.Server_Score + "_" + touch3.Returner_Score)
    touch3["Score_PI_Fill2"] = touch3.Score_PI_Fill2.fillna(method="ffill")

    touch3["ServeFrom"] = np.where(touch3.Score_PI_Fill2.isin(
        ["Five_Five", "Zero_Three", "Three_Zero", "Three_Three", "Four_Five", "Five_Four", "Four_Four",
        "Deuce_Deuce"
        ]), "Adv", "Deuce")

    touch3["ServeFrom2"] = np.where((touch3.Score_PI_Fill2 == "Game") & (touch3.ServeFrom.shift() == "Adv"), "Deuce",
                                    np.where((touch3.Score_PI_Fill2 == "Game") & (touch3.ServeFrom.shift() == "Deuce"),
                                            "Adv", touch3["ServeFrom"]))

    touch3["FirstSecond"] = np.where(touch3.RealSecondServe == 1, "Second", "First")
    
    return touch3

def correct_OnServe(touch3, eval_fin8):
    # correct first shot
    # merge the OnServeCorrected to Shot Preds (eval_fin8)
    touch3["TimeTrueStrike"] = touch3.minimum
    eval_fin9 = pd.merge(eval_fin8, touch3[["TimeTrueStrike", "OnServe_Corrected"]], how="left", on="TimeTrueStrike")
    # rename RealShot
    eval_fin9["old_RealShot"] = eval_fin9.RealShot
    # calculate if serve then serve, else what suggested, if first point, not volley
    eval_fin9["RealShot"] = np.where(eval_fin9.OnServe_Corrected == 1, "Serve",
                                    np.where(eval_fin9.OnServe_Corrected == 0,
                                            np.where(eval_fin9.old_RealShot == "Serve", eval_fin9.ComboPred,
                                                    np.where(eval_fin9.old_RealShot == "Volley", "Slice",
                                                                eval_fin9.old_RealShot)), eval_fin9.old_RealShot))
    eval_fin9.rename(columns={"OnServe_Corrected": "ONS"}, inplace=True)
    touch3.drop("TimeTrueStrike", axis=1, inplace=True)

    # correcting minimum where its duplicated
    touch3["GameCount"] = np.where(touch3.minimum == touch3.minimum.shift(1),
                                1000 + touch3.EndGameRowNum_ffill + touch3.PointsSinceEoG, touch3.GameCount)
    touch3["NumShotsinPt"] = np.where(touch3.minimum == touch3.minimum.shift(1), 0, touch3.NumShotsinPt)
    touch3["minimum"] = np.where(touch3.minimum == touch3.minimum.shift(1), touch3.Seconds - 2, touch3.minimum)
    touch3["maximum"] = np.where(touch3.maximum == touch3.maximum.shift(1), touch3.Seconds - 1, touch3.maximum)
    touch3["min_adj"] = np.where(touch3.min_adj == touch3.min_adj.shift(1), touch3.Seconds - 2, touch3.min_adj)
    touch3["max_adj"] = np.where(touch3.max_adj == touch3.max_adj.shift(1), touch3.Seconds - 1, touch3.max_adj)
    touch3["keep_end"] = np.where(touch3.keep_end == touch3.keep_end.shift(1), touch3.Seconds - 1, touch3.keep_end)
    touch3["keep_end2"] = np.where(touch3.keep_end2 == touch3.keep_end2.shift(1), touch3.Seconds - 1, touch3.keep_end2)

    return touch3