import pandas as pd
import numpy as np

class OracleTracker:
    def __init__(self,
                 oracle_limit: int):
        self.oracle_limit = oracle_limit
        self.oracle_calls = 0
        # track sampling as a function of oracle calls
        self.oracle_tracker = pd.DataFrame({'epoch': [],
                                            'oracle_calls': [],
                                            'reward': [],
                                            'smiles': []})

    def update_oracle_tracker(self, epoch: int, reward: np.array, smiles: np.array):
        df = pd.DataFrame({"epoch": np.full_like(smiles, epoch), 
                           "oracle_calls": np.full_like(smiles, self.oracle_calls),
                           "reward": reward, 
                           "smiles": smiles})
        
        self.oracle_tracker = pd.concat([self.oracle_tracker, df])

    def epoch_updates(self, num_valid_smiles: int, epoch: int, reward: np.array, smiles: np.array):
        """
        this method performs 2 updates on every epoch:
        1. Increments the number of oracle calls so far
        2. Updates the Oracle Tracker that tracks the generative sampling as a function of oracle calls
        """
        self.oracle_calls += num_valid_smiles
        # track generated SMILES + reward as a function of oracle calls - used to assess sample efficiency
        self.update_oracle_tracker(epoch=epoch,
                                   reward=reward,
                                   smiles=smiles)

    def budget_exceeded(self):
        if self.oracle_calls >= self.oracle_limit:
            print(f'----- Reached oracle limit of {self.oracle_limit} -----')
            # write out generative sampling as a function of oracle calls
            self.write_out_oracle_tracker()
            return True
        return False

    def write_out_oracle_tracker(self):
        self.oracle_tracker.to_csv('oracle_tracker.csv')
