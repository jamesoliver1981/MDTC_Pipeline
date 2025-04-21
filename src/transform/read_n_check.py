def read_n_check(path, skipnum):
    
    df = pd.read_csv(path, skiprows = skipnum)
    if (df.SampleTimeFine[1] - df.SampleTimeFine[0]) == 16667:
        print("60Hz")
#         df["FirstCut"] = df.PacketCounter*16667/1000000
#         gap =16667
    else :
        print(df.SampleTimeFine[1] - df.SampleTimeFine[0])
        print("converting")
        if df.shape[0] % 2 ==1:
            df = df.iloc[:-1,:]
        new_index =[]
        for i in range(0,int(np.ceil(len(df)/2))):
            new_index.append(i)
            new_index.append(i)
        df["new_index"] = new_index
        df = df.groupby("new_index").mean()
        df["SampleTimeFine"] = df.SampleTimeFine +1
        df["PacketCounter"] = df.PacketCounter//2
#         df["FirstCut"] = df.PacketCounter*16666/1000000
#         gap = 16666
    
    df["FirstCut"] = df.PacketCounter*16667/1000000  
    print("Minutes of Recording: {:}".format(df.FirstCut.max()/60))
    df["Validate"] = df.SampleTimeFine.diff()
    print(df.Validate.value_counts())
          
    return df#, gap