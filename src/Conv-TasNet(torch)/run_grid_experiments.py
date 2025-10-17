import copy
from train import CONFIG, main

experiments = [
    # Run 1: emphasize perceptual
    {"name": "grid_run_perc_high", "lambda_perc": 2.0, "lambda_sdr": 1.5, "lambda_stft": 1.5},
    # Run 2: moderate perceptual
    {"name": "grid_run_perc_med", "lambda_perc": 1.5, "lambda_sdr": 1.5, "lambda_stft": 1.5},
    # Run 3: control
    {"name": "grid_run_control", "lambda_perc": 1.5, "lambda_sdr": 2.0, "lambda_stft": 1.5},
]

if __name__ == '__main__':
    for exp in experiments:
        cfg = copy.deepcopy(CONFIG)
        cfg['epochs'] = 3
        cfg['batch_size'] = 4
        cfg['steps_per_epoch'] = 100
        cfg['use_torch_compile'] = False
        cfg['lambda_perc'] = exp['lambda_perc']
        cfg['lambda_sdr'] = exp['lambda_sdr']
        cfg['lambda_stft'] = exp['lambda_stft']
        cfg['output_root'] = f"convtasnet_grid_{exp['name']}"
        print('='*80)
        print(f"Starting experiment: {exp['name']} -> perc={cfg['lambda_perc']}, sdr={cfg['lambda_sdr']}")
        main(cfg)
        print(f"Finished experiment: {exp['name']}")
