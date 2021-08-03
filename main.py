import os
import argparse
import json

def main():
    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='Modified QG model')
    parser.add_argument("namelist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist

    main_simualtion(namelist)

    return

def main_simualtion(namelist):
    import Simulation
    Simulation = Simulation.Simulation(namelist)
    Simulation.initialize(namelist)
    Simulation.run(namelist)
    print('The simulation has completed.')

    return

if __name__ == "__main__":
    main()


