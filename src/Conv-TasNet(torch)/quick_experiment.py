from train import CONFIG, main
import copy

if __name__ == '__main__':
    cfg = copy.deepcopy(CONFIG)
    cfg['epochs'] = 3
    cfg['save_interval'] = 1
    cfg['output_root'] = 'convtasnet_quick_experiment'
    # make training small
    cfg['batch_size'] = 4
    cfg['steps_per_epoch'] = 200
    # initialize skip gates to 0.0 for this quick test
    cfg['skip_gates_init'] = True
    cfg['skip_gates_init_value'] = 0.0
    print('Running quick experiment with config:')
    print({k:cfg[k] for k in ['epochs','batch_size','steps_per_epoch','output_root']})
    main(cfg)
