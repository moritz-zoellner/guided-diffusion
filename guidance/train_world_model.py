import h5py
import re

def get_demo_keys(path):
    with h5py.File(path, 'r') as f:
        keys = list(f['data'].keys())
    # Numerical sort: 'demo_9' comes before 'demo_10'
    return sorted(keys, key=lambda x: int(re.search(r'\d+', x).group()))

def collect_trajectories(path1, path2, path3, machine_percent=0.5):
    """
    path1, path2: Human (Expert)
    path3: Machine Generated (Mixed)
    """
    # 1. Get all Human Demos
    human_demos = [(path1, k) for k in get_demo_keys(path1)]
    human_demos += [(path2, k) for k in get_demo_keys(path2)]
    num_human = len(human_demos)
    
    # 2. Calculate how many MG demos we need to hit the target %
    # total = human / (1 - machine_percent) -> mg_count = total - human
    total_needed = int(num_human / (1 - machine_percent))
    mg_count_needed = total_needed - num_human
    
    # 3. Get the LATEST Machine Demos (Groups 11-13)
    mg_all_keys = get_demo_keys(path3)
    # Take the last X entries
    mg_keys_selected = mg_all_keys[-mg_count_needed:]
    mg_demos = [(path3, k) for k in mg_keys_selected]
    
    all_demos = human_demos + mg_demos
    
    print(f"Dataset Summary:")
    print(f" - Human Demos: {num_human}")
    print(f" - Machine Demos (Selected from end): {len(mg_demos)}")
    print(f" - Total Trajectories: {len(all_demos)}")
    
    return all_demos

path1 = "./data/can/mh/low_dim_v15.hdf5"
path2 = "./data/can/ph/low_dim_v15.hdf5"
path3 = "./data/can/mg/low_dim_sparse_v15.hdf5"

selected_demos = collect_trajectories(path1, path2, path3, machine_percent=0.5)