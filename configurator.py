"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    # Skip special flags that are handled elsewhere (--clean-memory, --memory-path=..., --copy-clean-from=..., --create-clean-to=..., --visualizations-off, --profile, --show-brain, --brain-always)
    if arg == '--clean-memory':
        continue  # Handled in train_hdeq.py
    if arg == '--visualizations-off':
        continue  # Handled in train_hdeq.py
    if arg == '--profile':
        continue  # Handled in train_hdeq.py
    if arg == '--show-brain':
        continue  # Handled in graph_memory_system.py - shows DEQ debug view (sparse)
    if arg == '--brain-always':
        continue  # Handled in graph_memory_system.py - shows every DEQ iteration (verbose)
    if arg.startswith('--memory-path='):
        continue  # Handled in train_hdeq.py
    if arg.startswith('--copy-clean-from='):
        continue  # Handled in train_hdeq.py
    if arg.startswith('--create-clean-to='):
        continue  # Handled in train_hdeq.py
    
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--'), f"Flag {arg} must be either a config file or --key=value format"
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
