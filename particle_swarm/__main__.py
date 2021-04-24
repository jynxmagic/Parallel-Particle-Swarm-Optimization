"""Program entry point."""
import time
from pprint import pprint

from particle_swarm.pso import run

if __name__ == "__main__":
    now = time.time()
    pprint(run())
    then = time.time()

    tt = then - now
    print("time taken: ", tt)
