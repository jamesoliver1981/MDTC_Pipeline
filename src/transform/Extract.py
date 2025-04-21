import glob
import pandas as pd
import zipfile
import datetime as dt
import os
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

def dezip(zip_path, base_dir="data/tmp"):
    """
    Extracts a .zip file into a temporary local directory inside the project.

    Args:
        zip_path (str): Path to the .zip file.
        base_dir (str): Base directory for extraction (default is 'data/tmp').

    Returns:
        str: Path to the extracted contents.
    """
    zip_filename = os.path.basename(zip_path)
    zip_name = zip_filename.replace(".zip", "")
    extract_path = os.path.join(base_dir, zip_name, "Extract")

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path, zip_name

def create_out(pfad):
    
    meta = pd.read_json(glob.glob(f"{pfad}/*Main_Container*.json")[0])
    
    if meta.shape[0] == 0:
        meta_fin = pd.DataFrame({'id':[0],'StartTime':[0],'StopTime':[0],
           'currentFrequency':[0], 'duration':[0], 'method':[0] })
    else:
        meta1 = meta.iloc[0,:]

        Start = meta.sessionStartDate.values[0]
        Stop = meta.sessionStopDate.values[0]

        Start1 = dt.datetime(2001, 1, 1, 0, 0, 0) + dt.timedelta(seconds = Start)
        Stop1 = dt.datetime(2001, 1, 1, 0, 0, 0) + dt.timedelta(seconds = Stop)
        meta1["StartTime"] = Start1.strftime('%Y-%m-%d : %H-%M-%S')
        meta1["StartTime2"] = Start1.strftime('%Y-%m-%d_%H-%M-%S')
        meta1["StopTime"] = Stop1.strftime('%Y-%m-%d : %H-%M-%S')
        meta1 = meta1.to_frame().transpose()
        meta_fin = meta1[['id','StartTime','StopTime',
               'currentFrequency', 'duration', 'method' ,"StartTime2"]]
        meta_fin["model"] = meta.model.values[0]
    
    #adding in the meta_Player data
    meta_Player = pd.read_json(glob.glob(f"{pfad}/metadata.json")[0])
    
    rightOrLeft = meta_Player.profile.dominantHand
    bornYear = meta_Player.profile.birthYear
    gender = meta_Player.profile.gender
    rating_lev = meta_Player.profile.rating
    rating_typ = meta_Player.profile.ratingType

    matchResult = meta_Player.metadata.result
    matchLevel = meta_Player.metadata.opponentLevel
    matchSurface = meta_Player.metadata.surface
    matchType = meta_Player.metadata.type

    meta_id = meta_fin["id"].values[0]
    
    files = glob.glob(f"{pfad}/*SensorReadings*.json")
    ds = pd.DataFrame()

    #adding error handling
    if len(files) == 0:
        missingdata = 1
    else:
        missingdata = 0
        for i in range(len(files)):
        #     print(files[i][-9:])
            d = pd.read_json(files[i])
            ds = pd.concat([ds, d] , axis = 0)

        #error handling if won or loss or secondserve or endofgame not in data
        if "endOfGame"  not in ds.columns:
            ds["endOfGame"] = 0
        if "won"  not in ds.columns:
            ds["won"] = 0
        if "lost"  not in ds.columns:
            ds["lost"] = 0
        if "secondServe"  not in ds.columns:
            ds["secondServe"] = 0
        
        ds = ds[[#"timeStamp",
            "timeInterval", "gyroX", "gyroY", "gyroZ", "accX", "accY", "accZ",#"attX", "attY", "attZ",
            "heartRate"
                , "endOfGame",
                "won", "lost","secondServe"
                ]]
        ds = ds.sort_values("timeInterval", ascending = True)
        ds["Diff"] = ds.timeInterval.diff()
        ds["Seconds"] = ds.Diff.cumsum().fillna(0)
        ds = ds.reset_index().rename(columns={"index": "timeStamp"})
        maxtime = ds["Seconds"].max()
    return (
        ds, 
        meta_fin, 
        missingdata, 
        rightOrLeft, 
        bornYear, 
        gender, 
        rating_lev, 
        rating_typ, 
        matchResult, 
        matchLevel, 
        matchSurface, 
        matchType, 
        meta_id, 
        maxtime
        )

def convert_data(ds, hand):
    df = ds.copy()
    if hand == "LEFT":
        df["Acc_X"] = round(df.accX,4) * 9.8065
        df["Acc_Y"] = round(df.accY,4) * -9.8065 * -1
        df["Acc_Z"] = round(df.accZ,4) * 9.8065
        df["Gyr_X"] = round(df.gyroX,2) * -57.2958 *-1
        df["Gyr_Y"] = round(df.gyroY,2) * 57.2958
        df["Gyr_Z"] = round(df.gyroZ,2) * -57.2958 *-1
        
    else:
        df["Acc_X"] = round(df.accX,4) * 9.8065
        df["Acc_Y"] = round(df.accY,4) * -9.8065
        df["Acc_Z"] = round(df.accZ,4) * 9.8065
        df["Gyr_X"] = round(df.gyroX,2) * -57.2958
        df["Gyr_Y"] = round(df.gyroY,2) * 57.2958
        df["Gyr_Z"] = round(df.gyroZ,2) * -57.2958
        
    return df

def create_smooth(df, start, fin):
    smooth = pd.DataFrame({"Diff": [0.016667] * (fin - start) * 60})
    smooth["ShotTime"] = smooth.cumsum() + start

    smooth = pd.merge_asof(smooth, df[["Seconds",'Acc_X', 'Acc_Y', 'Acc_Z',
       'Gyr_X', 'Gyr_Y', 'Gyr_Z' ]].rename(columns = { 'Acc_X':'Acc_X_pre', 'Acc_Y':'Acc_Y_pre', 'Acc_Z': 'Acc_Z_pre',
       'Gyr_X': 'Gyr_X_pre', 'Gyr_Y': 'Gyr_Y_pre', 'Gyr_Z': 'Gyr_Z_pre'}), left_on = "ShotTime", right_on = "Seconds", 
                       direction = "backward")
    smooth = pd.merge_asof(smooth, df[["Seconds",'Acc_X', 'Acc_Y', 'Acc_Z',
       'Gyr_X', 'Gyr_Y', 'Gyr_Z' ]].rename(columns = { 'Acc_X':'Acc_X_pst', 'Acc_Y':'Acc_Y_pst', 'Acc_Z': 'Acc_Z_pst',
       'Gyr_X': 'Gyr_X_pst', 'Gyr_Y': 'Gyr_Y_pst', 'Gyr_Z': 'Gyr_Z_pst'}), left_on = "ShotTime", right_on = "Seconds", 
                       direction = "forward")
    #create the new values - difference between pre & post per %
    smooth["Recal"] = np.where((smooth.ShotTime - smooth.Seconds_x) <= (smooth.Seconds_y - smooth.ShotTime), (smooth.ShotTime - smooth.Seconds_x), 
                               (smooth.Seconds_y - smooth.ShotTime) )
    smooth["Recalpct"] = (smooth.Recal / (smooth.Seconds_y - smooth.Seconds_x))

    cols = ['Acc_X', 'Acc_Y', 'Acc_Z',
       'Gyr_X', 'Gyr_Y', 'Gyr_Z']
    for i in range(len(cols)):
        smooth[cols[i]] = np.where(smooth.ShotTime == smooth.Seconds_x, smooth[f"{cols[i]}_pre"] ,
                                   np.where((smooth.ShotTime - smooth.Seconds_x) <= (smooth.Seconds_y - smooth.ShotTime),
                                   ( smooth[f"{cols[i]}_pre"] + smooth.Recalpct * (smooth[f"{cols[i]}_pst"] - smooth[f"{cols[i]}_pre"]) ),
                                   ( smooth[f"{cols[i]}_pst"] - smooth.Recalpct * (smooth[f"{cols[i]}_pst"] - smooth[f"{cols[i]}_pre"]) )))
    
    smooth = smooth.reset_index().rename(columns={"index": "PacketCounter"})
    return smooth


