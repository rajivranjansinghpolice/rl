#!/usr/bin/env python3

# Run without interactive mode to avoid stdin issues

import sys
sys.path.append('/home/romeo/projects/rl')

from rl_gridworld import main

if __name__ == "__main__":
    # Override arguments to avoid interactive input  
    original_argv = sys.argv
    
    # Run with minimal arguments to test everything is working properly
    sys.argv = ['rl_gridworld.py', '--size', '4', '--gamma', '0.99', '--theta', '1e-6']
    
    try:
        main()
    except SystemExit:
        pass  # Expected when program exits normally
    finally:
        sys.argv = original_argv