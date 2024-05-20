import wandb
import argparse
import json
from tqdm import tqdm
import yaml,os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--experiment_directory', help='Path to the experiment dir', required=True)
    parser.add_argument('--entity', help='Entity for wandb run', default="m42")
    parser.add_argument('--project', help='Project name for wandb run', default="dinov2_cxr")
    # parser.add_argument('--exp_name', help='Experiment name for wandb run', default="CXR_FM")
    parser.add_argument('--id', help='Run id for wandb run', default='', type=str)
    args = vars(parser.parse_args())
    print(args)
    
    yaml_file = os.path.join(args['experiment_directory'], 'config.yaml')
    json_fpath = os.path.join(args['experiment_directory'], 'training_metrics.json')
    exp_name = args['experiment_directory'].rstrip('/').split('/')[-1]
    print("Experiment name: ", exp_name)
    
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    log_data = {}
    with open(json_fpath, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            key = data['iteration']
            log_data[key] = data

    print("Log data: ", log_data[0])
    if not args['id']:
        if os.path.isfile(os.path.join(args['experiment_directory'], '.wandb_expid')):
            with open(os.path.join(args['experiment_directory'], '.wandb_expid'), 'r') as file:
                args['id'] = file.read().strip()
        else:
            args['id'] = None


    start_iteration = 0
    if not args['id']:
        wandb.init(entity=args['entity'], project=args['project'], name =exp_name, config=config)
    else:
        wandb.init(entity=args['entity'], project=args['project'], name = exp_name, id=args['id'], resume="must")
        # when we resume a run we don't need to update metrics for previous training iterations
        start_iteration = wandb.run.starting_step

    all_keys = list(log_data.keys())

    all_keys = [x for x in all_keys if x > start_iteration]

    last_key = all_keys[-1] if len(all_keys) > 0 else None
    for temp_key in tqdm(all_keys[:-1]):
        wandb.log(log_data[temp_key], commit=False, step=temp_key)
    
    if last_key:
        wandb.log(log_data[last_key], step=last_key)

    if not args['id']:
        with open(os.path.join(args['experiment_directory'], '.wandb_expid'), 'w') as file:
            file.write(wandb.run.id)

    print("="*50)
    print("PLEASE NOTE THE BELOW RUN ID TO UPDATE")
    print(wandb.run.id)
    print("="*50)
    print("Also saved in {}".format(os.path.join(args['experiment_directory'], '.wandb_expid')))


        