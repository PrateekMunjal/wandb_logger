import wandb
import argparse
import json
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--json_fpath', help='Path to the json file', required=True)
    parser.add_argument('--entity', help='Entity for wandb run', default="m42")
    parser.add_argument('--project', help='Project name for wandb run', default="CXR_FM")
    parser.add_argument('--exp_name', help='Experiment name for wandb run', default="CXR_FM")
    parser.add_argument('--id', help='Run id for wandb run', default='', type=str)
    args = vars(parser.parse_args())
    print(args)

    log_data = {}
    with open(args['json_fpath'], 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            key = data['iteration']
            log_data[key] = data

    print("Log data: ", log_data[0])

    start_iteration = 0
    if len(args['id']) == 0:
        wandb.init(entity=args['entity'], project=args['project'], name = args['exp_name'])
    else:
        wandb.init(entity=args['entity'], project=args['project'], name = args['exp_name'], id=args['id'], resume="must")
        # when we resume a run we don't need to update metrics for previous training iterations
        start_iteration = wandb.run.starting_step

    all_keys = list(log_data.keys())

    all_keys = [x for x in all_keys if x > start_iteration]

    last_key = all_keys[-1] if len(all_keys) > 0 else None
    for temp_key in tqdm(all_keys[:-1]):
        wandb.log(log_data[temp_key], commit=False, step=temp_key)
    
    if last_key:
        wandb.log(log_data[last_key], step=last_key)

    print("="*50)
    print("PLEASE NOTE THE BELOW RUN ID TO UPDATE")
    print(wandb.run.id)
    print("="*50)


        