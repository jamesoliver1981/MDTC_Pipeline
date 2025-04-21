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