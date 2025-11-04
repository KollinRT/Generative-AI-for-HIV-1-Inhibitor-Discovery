07/02/25
ADD IN CORRUPTION FUNCTION AND SCALE UP A LOT?

Got training and validation per steps working

07/06/25
- Test to see that corruption works? [MASK] token is ...

07/07/25
- Print tests to see that [MASK] works still...
- get train code working with parquet files...

code is the correct two then filtered to less than 64 tokens

`model_4_warmup_FP_pre.parquet` was the df 1 and 3


07/21/25  
- Implement for resume training the grab the last row of the `.csv` file and get the global steps and then start from there.
  - Under `train_for_pretrain.py` file...
- Implement the search parquet for novelty aspect. MUCH LARGER....
  - For `benchmarkGen.py`
- Need to look into past files and I have a script somewhere at the end of my Jupyter notebook that has drug filtering criteria that I can add to filter these molecules for a further helper!
- Remember before serious generation that I still need to `finetune` on the molecules of interest!

07/24/25 
- Got the thingy done. Simulations be done...
  - Get the mini benchmarking and sorting and resuming from soonest steps pretty easily? I have a script midway sketched out.
- We need to get the `finetune` done before doing the docking. I have the docking for the first done.
  - Need to find the 1/2 classification model to list if it is rve or the other one... protease or integrase to know which site to dock to!

08/01/25
- Need to get the drug filtering criteria solidified...
  - Can do post-hoc with the RDKit API.. This doesn't have to be explicitly conjoined... But it can be?
- With this, also solidify the classification that I have working.
- REMEMBER THAT I NEED TO FINETUNE THE MODEL FIRST!
  - Before explicitly testing for the druglikeness


08/05/25 
- Try doing `...or_8th...` with 0.35 dropout?
- Update finetune code to utilize AMP and follow similar code format. The same batch size and shit should work...?


11/03/25
- Rerun the `layers_unfrozen=1` for the whole thing
- Run the `layers_unfrozen=6` for the whole thing.
- Then I can compare them.... print all three for the best model... Do I plot the 1lay, 3lay, 6lay all in a 3x2? So 6 figures per page and each does the thing? This shows the importance of layer freezing that I can justify to my committee.
- Figure out if `dropout` is important to include in the hyperparams.yaml file thingy....
- Figure out where the visualization script is!