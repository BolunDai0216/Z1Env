import time

import numpy as np

from Z1Env.z1_env import Z1Sim


def main(): 
    env = Z1Sim(render_mode="human")
    info = env.reset()

    Tf = 100000

    for i in range(100000):
        











if __name__ == "__main__":
    main()