def process(ema):
    channels = ['TD', 'TB', 'TT', 'LI', 'UL', 'LL']
    idx = {
        'TD': (0, 1),
        'TB': (2, 3),
        'TT': (4, 5),
        'LI': (6, 7),
        'UL': (8, 9),
        'LL': (10, 11),
    }
    traj = {name: ema[:, idx[name]] for name in channels} # (T, 2) each
    return traj