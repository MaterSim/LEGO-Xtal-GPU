import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from time import time
from pyxtal.db import database_topology

if __name__ == "__main__":
    t0 = time()

    db_name = 'torch_2_100'
    db = database_topology(f'{db_name}.db', log_file=f'{db_name}.log')



    attribute = 'mace_energy'
    N_lowE = 0
    N_lowE_cubic = 0
    with open(f'{db_name}_metric.txt', 'a+') as f:
        for row in db.db.select():
            if hasattr(row, attribute):
                eng = getattr(row, attribute)
                if -9.4 < eng < -8.8:
                    N_lowE += 1
                    if row.space_group_number >= 195:
                        N_lowE_cubic += 1
        f.write(f'N_lowE_all:      {N_lowE:12d}\n')
        f.write(f'N_lowE_cubic:    {N_lowE_cubic:12d}\n')