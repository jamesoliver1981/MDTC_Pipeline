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