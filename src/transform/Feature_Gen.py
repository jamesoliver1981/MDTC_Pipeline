import pandas as pd
import numpy as np

def feat_gen(d_input_orig):
          
    ########################################
                #WORK TO DO HERE#
    # will need to adjust (add ifs) ########
    #for the frequency of the data capture #
    # time periods add at start
    # create if and vars at start and ref l8tr*
    ########################################
    """
    Identifies where the data shows shots through an iterative process 
        estimates based on values lower than x
        Then takes the minimum within that shot window and refines it
    """
    #look at the seconds - all sequential? Need to add this element in here
        #################
        #WORK TO DO HERE#
        #################

    d_input = d_input_orig.copy()
    d_input["SampleTimeFine_base"] = d_input.PacketCounter*16667

    d_input["Seconds"] = d_input.SampleTimeFine_base/1000000

    
    d_input["Strike"]=np.where(d_input.Acc_X <= -30,1,0)

    d_input["Strike2"]=d_input["Strike"].shift(-30)
    d_input["Start"]=np.where(d_input.Strike2==1,d_input.Seconds,np.NaN)
    
    d_input["Strike3"]=d_input["Strike"].shift(30)
    d_input["End"]=np.where(d_input.Strike3==1,d_input.Seconds,np.NaN)

    d_input["Shot"]=(d_input['Strike2'] - d_input['Strike3'].shift(1, fill_value=0)).cumsum()
    d_input["Shot"]=np.where(d_input["Shot"]==0,0,1)
    
    #create rolling vars
#     for col in d_input.columns[2:11]:
    for col in['Acc_X', 'Acc_Y', 'Acc_Z','Gyr_X', 'Gyr_Y', 'Gyr_Z']:
        d_input[col + "_roll"]=d_input[col].rolling(25).mean() 
    
    Shots=d_input[d_input.Shot==1]
    
    Shots["TDiff"]=Shots.SampleTimeFine_base.diff()
    Shots["NewShot"]=np.where(Shots.TDiff != 16667  , 1,np.NaN)
    Shots["ShotCount"]=Shots.NewShot.cumsum().fillna(method="ffill")

    # #identify the true min point of the shot
    mins=Shots.groupby("ShotCount")["Acc_X"].min().reset_index()
    mins=pd.merge(mins,Shots[["ShotCount","Acc_X","Seconds","Gyr_Y_roll"]],how="left", on=["ShotCount","Acc_X"])
    mins["TrueStrike"]=1
    mins["GyrY_LT_AccX"] = np.where(mins.Gyr_Y_roll <= mins.Acc_X, 1, -1 )
    mins=mins.iloc[1:,]

    #apply the TrueStrike back to the data and recalc the shots

    d_input=pd.merge(d_input, mins[["Seconds","TrueStrike"]], on="Seconds", how="left")
    d_input["TrueStrike"]=d_input["TrueStrike"].fillna(0)


    d_input["Strike2"]=d_input["TrueStrike"].shift(-30)
    d_input["Start"]=np.where(d_input.Strike2==1,d_input.Seconds,np.NaN)

    d_input["Strike3"]=d_input["TrueStrike"].shift(30)
    d_input["End"]=np.where(d_input.Strike3==1,d_input.Seconds,np.NaN)

    d_input["Shot2"]=(d_input['Strike2'] - d_input['Strike3'].shift(1, fill_value=0)).cumsum()

    d_input["Shot2"]=np.where(d_input["Shot2"]==0,0,1)

    # #restrict to shots again
    Shots=d_input[d_input.Shot2==1]
    Shots["TDiff"]=Shots.SampleTimeFine_base.diff()
    Shots["NewShot"]=np.where(Shots.TDiff != 16667, 1,np.NaN)
    Shots["ShotCount"]=Shots.NewShot.cumsum().fillna(method="ffill")
    
  
    return d_input, Shots, mins

def create_points_part1(df_shots):
          
    droppers = df_shots.groupby("ShotCount")["Seconds"].count().reset_index()
    droppers = droppers[droppers.Seconds != 61]["ShotCount"].to_list()
    
    shots = df_shots[ ~ df_shots.ShotCount.isin(droppers)]      
    
    points = shots.groupby(["ShotCount"])["Acc_X"].min().reset_index()

    points=pd.merge(points,shots[["ShotCount","Acc_X","Seconds"]],how="left", on=["ShotCount","Acc_X"])
    points["TimeBetweenShots"]=np.round(points.Seconds.diff(),0)
    points["TimeBetweenShots"].value_counts()
    points["TimeBack"] = points.Seconds.diff()
    points["TimeFor"] = points.Seconds.diff(-1)*-1
#     cleaning up the fake mid rally shots
    points["DropShot"] = np.where((points.TimeBack <3) & (points.Acc_X > -50) & (points.TimeFor < 3)
                                  , 1,0)

    points["NewPoint"] = np.where(points.Seconds.diff() > 5.5, 1, 0)
    points["PointCount"] = points.NewPoint.cumsum()
    points["PointCount"] = points["PointCount"] +1
    
    return points, shots

def add_key(shots, pointsplus):
        #grab the truestrike time for every shot (shotcount) and apply it back to the data so can merge on that after
    truestriketime = shots[shots.TrueStrike ==1][["ShotCount","Seconds"]].rename(columns = {"Seconds": "TimeTrueStrike"}).reset_index(drop=True)
    shots2 = pd.merge(shots, truestriketime, on ="ShotCount", how ="left")

    shots2["Key"] = shots2.ShotCount
    
    return shots2

def shot_prep(d_in):
    mid1 = d_in.copy()
    mid1['Shot_Part'] = np.tile(range(61), len(mid1)//61)

    mid2 = pd.melt(mid1, id_vars=["Key","Shot_Part"],value_vars = ['Acc_X_roll', 'Acc_Y_roll', 'Acc_Z_roll'])
    
    #Shot Type as Number so maintain order
    mid2["Num_as_String"] = np.where( mid2.Shot_Part < 10, "0"+ mid2.Shot_Part.astype(str), mid2.Shot_Part.astype("str"))
    mid2["Var_Shot"]=mid2.variable + "_" + mid2.Num_as_String
    
    #pivot on both label and shotcount (superflous but maybe useful for reference later) - reset index so can seperate out
    out = pd.pivot_table(mid2, values= "value", index = ["Key"], columns ="Var_Shot")
    out= out.reset_index()

    return out

def shot_prep2(d_in):
    mid1 = d_in.copy()
    mid1['Shot_Part'] = np.tile(range(61), len(mid1)//61)
    
    mid2 = pd.melt(mid1, id_vars=["Key","TimeTrueStrike","Shot_Part"],value_vars = ['Acc_X', 'Acc_Y', 'Acc_Z',
                                                                    'Gyr_X', 'Gyr_Y', 'Gyr_Z'])
    
    #Shot Type as Number so maintain order
    mid2["Num_as_String"] = np.where( mid2.Shot_Part < 10, "0"+ mid2.Shot_Part.astype(str), mid2.Shot_Part.astype("str"))
    mid2["Var_Shot"]=mid2.variable + "_" + mid2.Num_as_String
    
    #pivot on both label and shotcount (superflous but maybe useful for reference later) - reset index so can seperate out
    out = pd.pivot_table(mid2, values= "value", index = ["Key","TimeTrueStrike"], columns ="Var_Shot")
    out= out.reset_index()
       
    return out

def points_prep(points, shots_wide):
            # from the points, take new point =1 ie the start points - get the serve yes no information

        points["NewPoint"][0] = 1
        game_pre = points[points.NewPoint == 1]

        game_pre = pd.merge(game_pre, shots_wide[["Key", "preds", "Serve"]], how="left", left_on="ShotCount",
                            right_on="Key")

        game_pre["Serve_min1"] = game_pre.Serve.shift(-1)
        game_pre["Serve_min2"] = game_pre.Serve.shift(-2)
        game_pre["Serve_min3"] = game_pre.Serve.shift(-3)
        game_pre["Serve_min4"] = game_pre.Serve.shift(-4)
        game_pre["Serve_plus1"] = game_pre.Serve.shift(1)
        game_pre["Serve_plus2"] = game_pre.Serve.shift(2)
        game_pre["Serve_plus3"] = game_pre.Serve.shift(3)
        game_pre["Serve_plus4"] = game_pre.Serve.shift(4)

        game_pre["NewGame_Basic"] = np.where(game_pre.Serve != game_pre.Serve_plus1, 1, 0)
        game_pre["NewGame_Basic_sum"] = game_pre["NewGame_Basic"].cumsum()

        # add in a number of points to a game allowed
        ptspergame = game_pre.groupby("NewGame_Basic_sum")["Serve"].count().reset_index().rename(
            columns={"Serve": "PtsPerGame"})

        game_pre = pd.merge(game_pre, ptspergame, how="left", on="NewGame_Basic_sum")
        game_pre["Drop"] = np.where(game_pre.PtsPerGame < 4, 1, 0)
        game_pre["Drop_refine"] = np.where(
            (game_pre.Serve == game_pre.Serve_min4) | (game_pre.Serve == game_pre.Serve_min3) | (
                        game_pre.Serve == game_pre.Serve_min2), 0, game_pre.Drop)
        game_pre["NewGame_Post"] = np.where(game_pre.Drop_refine == 1, 0, game_pre.NewGame_Basic)
        game_pre["NewGame_Post_sum"] = game_pre["NewGame_Post"].cumsum()

        game_pre["NewGame_simple"] = np.where(game_pre.Drop == 1, 0, game_pre.NewGame_Basic)
        game_pre["NewGame_simple_sum"] = game_pre["NewGame_simple"].cumsum()

        ngs = game_pre.groupby("NewGame_Post_sum")["Seconds"].min().reset_index()

        # create the same RE data using the original method.  Save it and test it in jupyter see where my data gets me

        # add Seconds back to this data - would need the truestrike time... will be around somewhere

        # generate a list of game starts - compare this to the truth
        ng = ngs["Seconds"].to_list()
        return ng, points, game_pre

def create_match (points, df_all, ng):
    match = pd.merge(df_all, points[["Seconds", 
                                       #"Shots", 
                                       "NewPoint", "ShotCount","PointCount"]],on ="Seconds",  how="left")

    
    match["Shot3"]= np.where(match.NewPoint.notnull(),1,0)
    match["NewPoint"].fillna(0,inplace=True)
    match["PointCount"].fillna(method="ffill",inplace=True)

    #redefining Shot or not based on if I include the shot
    match["NewGame"] = 0

    for n in ng:
        mid = match[match.Seconds == n]

        match.NewGame[match.PacketCounter == mid.PacketCounter.min()] = 1

    match["GameCount"] = match.NewGame.cumsum()

    return match

def create_points_part2(points, match):
    points = pd.merge(points, match[["Seconds", "GameCount","NewGame"]], how = "left" , on ="Seconds")
    startpoints = points.groupby("GameCount")["PointCount"].min().reset_index().rename(columns={"PointCount":"StartPoint"})
    points = pd.merge(points, startpoints, on = "GameCount", how = "left")
    points["PointInGame"] = points.PointCount - points.StartPoint + 1

    startshot = points.groupby("PointCount")["ShotCount"].min().reset_index().rename(columns={"ShotCount":"StartShot"})
    points = pd.merge(points, startshot, on = "PointCount", how ="left")
    points["ShotInGame"] = points.ShotCount - points.StartShot + 1
    
    return points

def mk_pts_start_end(points_2, game_pre, df_all):
    # new variant of point start end based on first shot

    points_start_end = points_2.groupby(["GameCount", "PointInGame"])["Seconds"].agg(
        {"min", "max", "count"}).reset_index().rename(
        columns={"min": "minimum", "max": "maximum", "count": "NumShotsinPt"})
    # adjustmentperiod is for the play in the game vs audio file, but won't be there later
    adjustmentperiod = 0  # audio is actually ahead this time
    # adding in 1.5 pre first shot, highly unlikey to say something here
    points_start_end["min_adj"] = points_start_end.minimum - adjustmentperiod - 1.5

    points_start_end["max_adj"] = points_start_end["min_adj"]
    points_start_end["keep_end"] = points_start_end.min_adj.shift(-1)
    points_start_end["keep_end2"] = np.where(points_start_end.keep_end.isnull(),
                                            df_all.Seconds.max() - adjustmentperiod,
                                            points_start_end.keep_end)

    # adding if point starts with Serve or Not from Gamepre
    # merge Serve & to points_2 on shotcount
    points_3 = pd.merge(points_2, game_pre[["ShotCount", "preds", "Serve", "NewGame_simple"]], how="left",
                        on="ShotCount")
    # restrict to first point in game so can merge
    points_3 = points_3.drop_duplicates(["GameCount", "PointInGame"], keep="first")

    # merge that information to points_start_end based on gameCount and POintInGame
    points_start_end2 = pd.merge(points_start_end,
                                points_3[["GameCount", "PointInGame", "preds", "Serve", "NewGame_simple"]],
                                how="left", on=["GameCount", "PointInGame"]).rename(
        columns={"Serve": "FirstIsServe", "preds": "PredIsServe"})
    
    return points_start_end2

def fake_gen(d_input_orig):
          
    ########################################
                #WORK TO DO HERE#
    # will need to adjust (add ifs) ########
    #for the frequency of the data capture #
    # time periods add at start
    # create if and vars at start and ref l8tr*
    ########################################
    """
    Identifies where the data shows shots through an iterative process 
        estimates based on values lower than x
        Then takes the minimum within that shot window and refines it
    """
    #look at the seconds - all sequential? Need to add this element in here
        #################
        #WORK TO DO HERE#
        #################

    d_input = d_input_orig.copy()
    d_input["SampleTimeFine_base"] = d_input.PacketCounter*16667

    d_input["Seconds"] = d_input.SampleTimeFine_base/1000000

    
    d_input["Strike"]=np.where((d_input.Acc_X <= -20) & (d_input.Acc_X >= -50) ,1,0)

    d_input["Strike2"]=d_input["Strike"].shift(-2)
    d_input["Start"]=np.where(d_input.Strike2==1,d_input.Seconds,np.NaN)
    
    d_input["Strike3"]=d_input["Strike"].shift(2)
    d_input["End"]=np.where(d_input.Strike3==1,d_input.Seconds,np.NaN)

    d_input["Movement"]=(d_input['Strike2'] - d_input['Strike3'].shift(1, fill_value=0)).cumsum()
    d_input["Movement"]=np.where(d_input["Movement"]==0,0,1)
    
#     #create rolling vars
    
    Movements =d_input[d_input.Movement==1]

    Movements["TDiff"]=Movements.SampleTimeFine_base.diff()

    Movements["NewMovement"]=np.where(Movements.TDiff != 16667  , 1,np.NaN)
    Movements["MovementCount"]=Movements.NewMovement.cumsum().fillna(method="ffill")
    
    # #identify the true min point of the shot
    mins=Movements.groupby("MovementCount")["Acc_X"].min().reset_index()

    mins=pd.merge(mins,Movements[["MovementCount","Acc_X","Seconds"]],how="left", on=["MovementCount","Acc_X"])
    mins["TrueStrike"]=1

    mins=mins.iloc[1:,]

    #apply the TrueStrike back to the data and recalc the shots

    d_input=pd.merge(d_input, mins[["Seconds","TrueStrike"]], on="Seconds", how="left")
    d_input["TrueStrike"]=d_input["TrueStrike"].fillna(0)

    d_input["Strike2"]=d_input["TrueStrike"].shift(-2)
    d_input["Start"]=np.where(d_input.Strike2==1,d_input.Seconds,np.NaN)


    d_input["Strike3"]=d_input["TrueStrike"].shift(2)
    d_input["End"]=np.where(d_input.Strike3==1,d_input.Seconds,np.NaN)

    d_input["Movement2"]=(d_input['Strike2'] - d_input['Strike3'].shift(1, fill_value=0)).cumsum()

    d_input["Movement2"]=np.where(d_input["Movement2"]==0,0,1)

    # #restrict to shots again
    Movements=d_input[d_input.Movement2==1]
    Movements["TDiff"]=Movements.SampleTimeFine_base.diff()
    Movements["NewMovement"]=np.where(Movements.TDiff != 16667, 1,np.NaN)
    Movements["MovementCount"]=Movements.NewMovement.cumsum().fillna(method="ffill")
    
#     check that all lengths are equal
    
    return Movements #d_input, Movements, mins

def clean_up_fakes(movemnts, eval_fin3, points, df_all, shots_wide):
    move_mins = movemnts.groupby("MovementCount")["Acc_X"].min().reset_index()
    move_mins2 = pd.merge(move_mins, movemnts[["MovementCount", "Acc_X", "Seconds"]], how="left",
                        on=["MovementCount", "Acc_X"])
    move_mins3 = pd.merge(move_mins2, points[["Seconds", "ShotCount", "PointCount"]], how="left", on="Seconds")
    move_mins3["TDiff"] = move_mins3.Seconds.diff()
    # # move_mins3["FakeID"] = np.where((move_mins3.ShotCount != "") & (move_mins3.TDiff < 0.45) & (move_mins3.Acc_X >= -50), 1, 0 )
    move_mins3["FakeID"] = np.where((move_mins3.ShotCount.notnull()) & (move_mins3.TDiff < 0.45), 1, 0)

    eval_fin4 = pd.merge(eval_fin3, move_mins3[["Seconds", "FakeID", "Acc_X"]], on="Seconds", how="left")
    eval_fin4["TDiff"] = eval_fin4.Seconds.diff()
    eval_fin4["Time_Fake_Short_Working"] = np.where(eval_fin4.TDiff < 2, 1, 0)

    eval_fin4["FakeID_Adj"] = np.where((eval_fin4.FakeID == 1) & (eval_fin4.Acc_X > -50) & (eval_fin4.ShotInGame > 1),
                                    1, 0)
    eval_fin4["WeakGS_Fake"] = np.where(
        (eval_fin4.Acc_X > -50) & (eval_fin4.ShotInGame > 1) & (eval_fin4.RealShot.isin(["FH", "BH"])), 1, 0)
    eval_fin4["Time_Fake_Short"] = np.where(
        (eval_fin4.Time_Fake_Short_Working == 1) & (eval_fin4.Time_Fake_Short_Working.shift(-1)), 1, 0)
    eval_fin4["Time_Fake_Long"] = np.where(
        (eval_fin4.ShotInGame > 1) & (eval_fin4.TDiff > 4.5) & (eval_fin4.Acc_X > -50), 1, 0)
    eval_fin4["Volley_Fake"] = np.where(
        (eval_fin4.RealShot == "Volley") & (eval_fin4.TDiff > 3) & (eval_fin4.ShotInGame > 1), 1, 0)
    eval_fin4["Fake_Label"] = eval_fin4.iloc[:, -5:].sum(axis=1)

    eval_fin5 = eval_fin4[eval_fin4.Fake_Label == 0]
    eval_fin5["TDiff2"] = eval_fin5.Seconds.diff()
    eval_fin5["Fake_Label2"] = np.where(
        (eval_fin5.Acc_X > -100) & ((eval_fin5.TDiff2 < 2) | ((eval_fin5.TDiff2 > 5.5) & (eval_fin5.ShotInGame > 1))),
        1, 0)

    eval_fin6 = eval_fin5[eval_fin5.Fake_Label2 == 0]
    eval_fin6["ShotsClean"] = 1
    eval_fin6["ShotCount"] = eval_fin6.ShotsClean.cumsum()

    # adding in RealShotCorrection for weaker 2nd Serve
    eval_fin6["PreRealShot"] = eval_fin6.RealShot
    eval_fin6["TDiff"] = eval_fin6.Seconds.diff()
    eval_fin6["RealShot"] = np.where(
        (eval_fin6.TDiff > 6) & (eval_fin6.TDiff < 20) & (eval_fin6.PreRealShot.shift(1) == "Serve") &
        (eval_fin6.PreRealShot.isin(["OH", "FH"])), "Serve", eval_fin6.PreRealShot)

    points_mid = pd.merge(points.drop(["ShotCount"], axis=1), eval_fin6[["Seconds", "ShotsClean", "ShotCount"]],
                        on="Seconds", how="left")
    points_mid2 = points_mid[points_mid.ShotsClean.notnull()].drop(["NewPoint", "PointCount"], axis=1)

    points_mid2["NewPoint"] = np.where(points_mid2.Seconds.diff() > 5.5, 1, 0)
    points_mid2["PointCount"] = points_mid2.NewPoint.cumsum()
    points_mid2["PointCount"] = points_mid2["PointCount"] + 1

    # from the points, take new point =1 ie the start points - get the serve yes no information

    points_mid2["NewPoint"][0] = 1
    game_pre2 = points_mid2[points_mid2.NewPoint == 1]

    game_pre2 = pd.merge(game_pre2, shots_wide[["Key", "preds", "Serve"]], how="left", left_on="ShotCount",
                        right_on="Key")

    game_pre2["Serve_min1"] = game_pre2.Serve.shift(-1)
    game_pre2["Serve_min2"] = game_pre2.Serve.shift(-2)
    game_pre2["Serve_min3"] = game_pre2.Serve.shift(-3)
    game_pre2["Serve_min4"] = game_pre2.Serve.shift(-4)
    game_pre2["Serve_plus1"] = game_pre2.Serve.shift(1)
    game_pre2["Serve_plus2"] = game_pre2.Serve.shift(2)
    game_pre2["Serve_plus3"] = game_pre2.Serve.shift(3)
    game_pre2["Serve_plus4"] = game_pre2.Serve.shift(4)

    game_pre2["NewGame_Basic"] = np.where(game_pre2.Serve != game_pre2.Serve_plus1, 1, 0)
    game_pre2["NewGame_Basic_sum"] = game_pre2["NewGame_Basic"].cumsum()

    # add in a number of points to a game allowed
    ptspergame2 = game_pre2.groupby("NewGame_Basic_sum")["Serve"].count().reset_index().rename(
        columns={"Serve": "PtsPerGame"})

    game_pre2 = pd.merge(game_pre2, ptspergame2, how="left", on="NewGame_Basic_sum")
    game_pre2["Drop"] = np.where(game_pre2.PtsPerGame < 4, 1, 0)
    game_pre2["Drop_refine"] = np.where(
        (game_pre2.Serve == game_pre2.Serve_min4) | (game_pre2.Serve == game_pre2.Serve_min3) | (
                    game_pre2.Serve == game_pre2.Serve_min2), 0, game_pre2.Drop)
    game_pre2["NewGame_Post"] = np.where(game_pre2.Drop_refine == 1, 0, game_pre2.NewGame_Basic)
    game_pre2["NewGame_Post_sum"] = game_pre2["NewGame_Post"].cumsum()

    game_pre2["NewGame_simple"] = np.where(game_pre2.Drop == 1, 0, game_pre2.NewGame_Basic)
    game_pre2["NewGame_simple_sum"] = game_pre2["NewGame_simple"].cumsum()

    ngs2 = game_pre2.groupby("NewGame_Post_sum")["Seconds"].min().reset_index()

    # create the same RE data using the original method.  Save it and test it in jupyter see where my data gets me

    # add Seconds back to this data - would need the truestrike time... will be around somewhere

    # generate a list of game starts - compare this to the truth
    ng2 = ngs2["Seconds"].to_list()

    match2 = create_match(points_mid2, df_all, ng2)

    points_2_2 = create_points_part2(points_mid2, match2)
    # new variant of point start end based on first shot
    points_start_end_3 = points_2_2.groupby(["GameCount", "PointInGame"])["Seconds"].agg(
        {"min", "max", "count"}).reset_index().rename(
        columns={"min": "minimum", "max": "maximum", "count": "NumShotsinPt"})
    adjustmentperiod = 0  # audio is actually ahead this time
    # adding in 1.5 pre first shot, highly unlikey to say something here
    points_start_end_3["min_adj"] = points_start_end_3.minimum - adjustmentperiod - 1.5
    points_start_end_3["max_adj"] = points_start_end_3["min_adj"]
    points_start_end_3["keep_end"] = points_start_end_3.min_adj.shift(-1)
    points_start_end_3["keep_end2"] = np.where(points_start_end_3.keep_end.isnull(),
                                            df_all.Seconds.max() - adjustmentperiod,
                                            points_start_end_3.keep_end)

    # removing this element and adding in Serve info from eval_fin6
    # #adding if point starts with Serve or Not from Gamepre
    # #merge Serve & to points_2 on shotcount
    # points_3_2 = pd.merge(points_2_2, game_pre2[["ShotCount", "preds","Serve","NewGame_simple"]], how="left", on = "ShotCount")
    # #restrict to first point in game so can merge
    # points_3_2 = points_3_2.drop_duplicates(["GameCount","PointInGame"], keep ="first")

    # #merge that information to points_start_end based on gameCount and POintInGame
    # points_start_end_4 = pd.merge(points_start_end_3, points_3_2[["GameCount","PointInGame","preds","Serve","NewGame_simple"]],
    #         how = "left", on = ["GameCount","PointInGame"]).rename(columns={"Serve":"FirstIsServe","preds":"PredIsServe"})

    # creating the version where the data has no text inputs - would be overwritten if text output is created
    points_start_end_4 = pd.merge(points_start_end_3, eval_fin6[["TimeTrueStrike", "PreRealShot", "RealShot"]],
                                left_on="minimum", right_on="TimeTrueStrike", how="left")
    points_start_end_4["FirstIsServe"] = np.where(points_start_end_4.RealShot == "Serve", 1, 0)
    points_start_end_4["ChangedShot"] = np.where(points_start_end_4.RealShot != points_start_end_4.PreRealShot, 1, 0)
    points_start_end_4.drop(["RealShot", "TimeTrueStrike"], axis=1, inplace=True)

    # merge the right Game Counts & ShotInGames to Shots & export
    eval_fin7 = eval_fin6.drop(["GameCount", "PointInGame", "GamePoint", "ShotInGame"], axis=1)
    eval_fin8 = pd.merge(eval_fin7, points_2_2[["Seconds", "GameCount", "PointInGame", "ShotInGame"]], how="left",
                        on="Seconds")
    eval_fin8["GamePoint"] = eval_fin8.GameCount.astype("str") + "_" + eval_fin8.PointInGame.astype("str")

    return eval_fin8, points_start_end_4