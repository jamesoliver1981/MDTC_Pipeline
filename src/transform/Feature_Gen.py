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