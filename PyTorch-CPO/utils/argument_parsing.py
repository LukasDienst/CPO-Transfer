import argparse

def parse_all_arguments():
        
    parser = argparse.ArgumentParser()
    
    # basic agruments 
    parser.add_argument('--algo-name', default="CPO", metavar='G',
                        help='algorithm name')
    parser.add_argument('--env-name', default="HalfCheetah-v3", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--exp-name', default="default", metavar='G',
                        help='experiment name, specify with folder or folder1/folder2')
    parser.add_argument('--exp-num', default="1", metavar='G',
                        help='experiment number for today (default: 1)')
    parser.add_argument('--max-iter-num', type=int, default=4000, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    
    ## arguments for using a pre-trained model and for correct TensorBoard logs
    parser.add_argument('--model-path', metavar='G',
                        help='path of pre-trained model')
    parser.add_argument('--only-write-in-own-path', action='store_true', default=False,
                        help='if True, then SummaryWriter only created in exp-name/part-exp-num, not also in exp-name')
    parser.add_argument('--prev-iter-num', type=int, default=0, metavar='N',
                        help='number of iterations from previously trained models used in model-path')
    parser.add_argument('--prev-best-reward', type=float, default=0, metavar='G',
                        help='best eval avg reward from previously trained models used in model-path')
    parser.add_argument('--prev-best-iter', type=int, default=-1, metavar='N',
                        help='iter with best eval avg reward from prev. trained models used in model-path')
    parser.add_argument('--prev-best-std', type=float, default=-1, metavar='G',
                        help='std from iter with eval avg reward from previously trained models used in model-path')
    parser.add_argument('--prev-violations', type=int, default=0, metavar='N',
                        help='number of iters with constraint_value > d_k in prev. trained models used in model-path')
    parser.add_argument('--prev-cv-sum', type=float, default=0, metavar='G',
                        help='cumulated sum of all constraint values from previously trained models used in model-path')
    parser.add_argument('--prev-infeasibles', type=int, default=0, metavar='N',
                        help='number of iters with infeasible optimization in prev. trained models used in model-path')
    parser.add_argument('--log-best-reward-in-last-iter', action='store_true', default=False,
                        help='add a point to rewards graph in TensorBoard summary if it is the last planned iter in this run')
    parser.add_argument('--no-constraint-mode', action='store_true', default=False,
                        help='with this mode, the constraint_value is stored in TensorBoard summary (but not d_k), then set to 0')

    ## rendering
    parser.add_argument('--render-int', type=int, default=0, metavar='N', ## 20 / 50 useful
                        help='interval in which the environment is rendered two times in a row (default: 0 = no rendering)')
    
    # learning rates and regularizations
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization of value function (default: 1e-3)')
    #parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='G', ## not used
    #                    help='gae (default: 3e-4 / 1e-3)')
    
    # GPU index, multi-threading and seeding
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    parser.add_argument('--num-threads', type=int, default=5, metavar='N',
                        help='number of threads for agent (default: 4 / 3)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    
    # batch sizes
    parser.add_argument('--min-batch-size', type=int, default=2000, metavar='N',
                        help='minimal batch size per CPO update (default: 2048 / 2000)')
    parser.add_argument('--max-batch-size', type=int, default=2000, metavar='N',
                        help='maximum batch size per CPO update (default: 2000)')
    parser.add_argument('--eval-batch-size', type=int, default=20000, metavar='N',
                        help='batch size per PPO update (default: 20000)')
    
    # logging and saving models
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10 / 1)')
    #parser.add_argument('--save-model-interval', type=int, default=50, metavar='N', ## not used
    #                    help='interval between saving model (default: 0 / 50, means don't save)')
    parser.add_argument('--save-intermediate-model', type=int, default=10, metavar='N',
                        help="intermediate model saving interval (default: 0 / 10, means don't save)")
    parser.add_argument('--acceptable-reward-std-percentage', type=float, default=5, metavar='G',
                        help='below which percentage of new best reward every model with std below that and new best reward should be saved')
    parser.add_argument('--include-std-for-model-saving', action='store_true', default=False,
                        help='do not consider the std of eval env avg reward whilst saving new best model')
    
    ## arguments for HalfCheetah-v3
    parser.add_argument('--forward-reward-weight', type=float, default=5.0, metavar='G',
                        help='weight for forward share in reward (default: 1.0 / 5.0)')
    
    ## arguments for applying noise
    parser.add_argument('--apply-noise-to-states', action='store_true', default=False,
                        help='apply noise to state_var used for action sampling')
    parser.add_argument('--noise-std-states', type=float, default=0, metavar='G',
                        help='normal distribution std for state noise generation (default: 0.05)')
    parser.add_argument('--apply-noise-to-actions', action='store_true', default=False,
                        help='apply noise to sampled actions')
    parser.add_argument('--noise-std-actions', type=float, default=0, metavar='G',
                        help='normal distribution std for action noise generation (default: 0.05)')
    
    """
    if parser.parse_args().algo_name == "TRPO": ## usage removed
        parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
        parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                        help='damping (default: 1e-2)')
        parser.parse_args()
    """

    ## else: CPO is used (default)
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
    parser.add_argument('--max-constraint', type=float, default=8e-2, metavar='G',
                    help='max constraint value (default: 1e-2 / 1e-3)')
    parser.add_argument('--annealing-factor', type=float, default=-0.00092376930293579, metavar='G', ## -5e-4 / 1-0.5^(1/halflife)
                    help='annealing factor of constraint (default: 1e-2 / 1e-6)')
    parser.add_argument('--anneal', default=True,
                    help='Should the constraint be annealed or not (default: True)')
    #parser.add_argument('--grad-norm', default=False, ## not used
    #                help='Should the norm of policy gradient be taken (default: False)') 
    parser.add_argument('--use-line-search', default=True,
                    help='use line search for policy update (default: True)')
    parser.add_argument('--cost-factor', type=float, default=0.02, metavar='G',
                        help='factor multiplied with 1 or 0 when constraint violated or satisfied')
    
    return parser.parse_args()

