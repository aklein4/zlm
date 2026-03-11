

import os
import json
import utils.constants as constants


def main():
    
    for checkpoint in os.listdir(os.path.join(constants.LOCAL_DATA_PATH, "evaluation_results")):
        for step in os.listdir(os.path.join(constants.LOCAL_DATA_PATH, "evaluation_results", checkpoint)):
            
            path = os.path.join(constants.LOCAL_DATA_PATH, "evaluation_results", checkpoint, step)
            print(f"Checkpoint: {checkpoint}/{step}")

            for f in os.listdir(path):
                if f.endswith(".json"):
                    
                    try:
                        with open(os.path.join(path, f)) as file:
                            data = json.load(file)
                            print(f"    {f}: {data['accuracy']:.2f} ({data['seen']})")
                    
                    except Exception as e:
                        pass

if __name__ == "__main__":
    main()
