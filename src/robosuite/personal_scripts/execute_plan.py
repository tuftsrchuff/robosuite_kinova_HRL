

if __name__ == "__main__":
    print("Executing plan...")
    #First call planner and return the plan
    #Plan - ['MOVE D1 D2 PEG2', 'MOVE D2 D3 PEG3', 'MOVE D1 PEG2 D2', 'MOVE D3 D4 PEG2', 'MOVE D1 D2 D4', 'MOVE D2 PEG3 D3', 'MOVE D1 D4 D2', 'MOVE D4 PEG1 PEG3', 'MOVE D1 D2 D4', 'MOVE D2 D3 PEG1', 'MOVE D1 D4 D2', 'MOVE D3 PEG2 D4', 'MOVE D1 D2 PEG2', 'MOVE D2 PEG1 D3', 'MOVE D1 PEG2 D2']

    
    #How is move decomposed into reach-pick, reachdrop, pick and drop?
    

    #Then create simple executor that just executes full policy
        #What would the pre and post conditions be?
            #Specified in the plan itself
        #Check pre and post conditions
    

    #Append the observation with the goal

    #Execute full policy for plan step

    #Optimal if you could watch this all happen
        #Success/failure message