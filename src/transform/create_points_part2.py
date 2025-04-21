def create_points_part2(points, match):
    points = pd.merge(points, match[["Seconds", "GameCount","NewGame"]], how = "left" , on ="Seconds")
    startpoints = points.groupby("GameCount")["PointCount"].min().reset_index().rename(columns={"PointCount":"StartPoint"})
    points = pd.merge(points, startpoints, on = "GameCount", how = "left")
    points["PointInGame"] = points.PointCount - points.StartPoint + 1

    startshot = points.groupby("PointCount")["ShotCount"].min().reset_index().rename(columns={"ShotCount":"StartShot"})
    points = pd.merge(points, startshot, on = "PointCount", how ="left")
    points["ShotInGame"] = points.ShotCount - points.StartShot + 1
    
    return points