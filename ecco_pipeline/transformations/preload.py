import os
import pickle
import xarray as xr

class Factors():
    '''
    Class for loading and storing all mapping factors.
    '''
    def __init__(self) -> None:
        pass
    
    def load_factors(self, factors_path):
        factors_name = os.path.basename(factors_path)
        with open(factors_path, "rb") as f:
            factors = pickle.load(f)
        setattr(self, factors_name, factors)
        
    def set_factors(self, factors_path):
        factors_name = os.path.basename(factors_path)
        if not getattr(self, factors_name, None):
            self.load_factors(factors_path)
        return getattr(self, factors_name)
    
    
class Grids():
    '''
    Class for loading and storing all mapping factors.
    '''
    def __init__(self) -> None:
        pass
    
    def load_grid(self, grid_ds):
        setattr(self, grid_ds.name, grid_ds)
        
    def set_grid(self, grid_path):
        grid_ds = xr.open_dataset(grid_path)
        if not getattr(self, grid_ds.name, None):
            self.load_grid(grid_ds)