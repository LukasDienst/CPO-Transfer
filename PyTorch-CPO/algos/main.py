import os
import sys
import pickle
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../gym')))
import gym

from utils import *
from models.continuous_policy import Policy
from models.critic import Value
from models.discrete_policy import DiscretePolicy
from algos.trpo import trpo_step
from algos.cpo import cpo_step
from core.common import estimate_advantages, estimate_constraint_value
from core.agent import Agent
CUDA_LAUNCH_BLOCKING=1

# Summarizing and plotting using TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Returns the current local date
from datetime import date
today = date.today()
print("Today date is: ", today)

# Parse arguments 
args = parse_all_arguments()

# Use of GPU or CPU
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)
else:
    print('using cpu')

"""environment"""
if not args.env_name == "HalfCheetah-v3":
    env = gym.make(args.env_name)
else: ## this code is mainly focussed on this environment
    env = gym.make("HalfCheetah-v3", forward_reward_weight=args.forward_reward_weight)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter((state_dim,), clip=5)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""create all the paths to save learned models/data"""
save_info_obj = save_info(assets_dir(), args.exp_num, args.exp_name, args.env_name) # model saving object
save_info_obj.create_all_paths() # create all paths

"""set up TensorBoard summary"""
## new TensorBoard summary for this exp-num
writer = SummaryWriter(os.path.join(assets_dir(), save_info_obj.saving_path, 'runs/'))
if not args.only_write_in_own_path: ## TensorBoard summary for whole exp
    writer2 = SummaryWriter(os.path.join(assets_dir(), save_info_obj.exp_path, 'runs/'))

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, render_interval=args.render_int, num_threads=args.num_threads,
    apply_noise_to_states=args.apply_noise_to_states, noise_std_states=args.noise_std_states,
    apply_noise_to_actions=args.apply_noise_to_actions, noise_std_actions=args.noise_std_actions)

"""help functions for costs"""
def too_close_to_180_deg(x, allowed_dist_to_180_deg = 0.5*np.pi):
    x_mod = abs(x) % (2*np.pi)
    if x_mod < 0 or x_mod > 2*np.pi:
        print("--- modulo of root angle not in [0, 2*pi] ---")
    else:
        return abs(x_mod - np.pi) < allowed_dist_to_180_deg

cost_help = np.vectorize(lambda z: 1 if too_close_to_180_deg(z) else 0)

"""define constraint cost function"""    
def constraint_cost(state, action):
    if args.algo_name == "CPO":
        root_angles = state[:,1] ## in rad
        costs = tensor(cost_help(root_angles)*args.cost_factor, dtype=dtype).to(device)
    elif args.algo_name == "TRPO":
        costs = tensor(0.01 * np.ones(state.shape[0]), dtype=dtype).to(device)
    return costs

def update_params(batch, d_k=0):
    states = torch.from_numpy(np.stack(batch.state)[:args.max_batch_size]).to(dtype).to(device) #[:args.batch_size]
    actions = torch.from_numpy(np.stack(batch.action)[:args.max_batch_size]).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)[:args.max_batch_size]).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)[:args.max_batch_size]).to(dtype).to(device)

    with torch.no_grad():
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
    
    if args.algo_name == "CPO":
        costs = constraint_cost(states, actions)
        cost_advantages, _ = estimate_advantages(costs, masks, values, args.gamma, args.tau, device)
        constraint_value = estimate_constraint_value(costs, masks, args.gamma, device)
        constraint_value = constraint_value[0]

    """perform update"""
    if args.algo_name == "CPO":
        if args.no_constraint_mode: ## constraint_value set to 0 for updates, but still has its old value for TensorBoard summaries (opt_was_infeasible always false)
            v_loss, p_loss, cost_loss, opt_was_infeasible = cpo_step(args.env_name, policy_net, value_net, states, actions, returns, advantages, cost_advantages, constraint_value*0, d_k, args.max_kl, args.damping, args.l2_reg, args.use_line_search)
        else:
            v_loss, p_loss, cost_loss, opt_was_infeasible = cpo_step(args.env_name, policy_net, value_net, states, actions, returns, advantages, cost_advantages, constraint_value, d_k, args.max_kl, args.damping, args.l2_reg, args.use_line_search)
    elif args.algo_name == "TRPO":
        v_loss, p_loss = trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)
        cost_loss = 0
        
    return v_loss, p_loss, cost_loss, constraint_value, opt_was_infeasible


def main_loop():
    # variables and lists for recording losses
    v_loss = 0
    p_loss = 0
    cost_loss = 0
    v_loss_list = []
    p_loss_list = []
    cost_loss_list = []
    
    # lists for dumping plotting data for agent
    rewards_std = []
    env_avg_reward = []
    num_of_steps = []
    num_of_episodes = []
    total_num_episodes = []
    total_num_steps = []
    tne = 0 #cumulative number of episodes
    tns = 0 #cumulative number of steps
    
    # lists for dumping plotting data for mean agent
    eval_avg_reward = []
    eval_avg_reward_std = []
    
    if args.algo_name == "CPO":
        # define initial d_k
        d_k = args.max_constraint
        # define annealing factor
        if args.anneal == True:
            e_k = args.annealing_factor
        else:
            e_k = 0            
    
    # for saving the best model
    best_avg_reward = args.prev_best_reward
    iter_of_best_reward = args.prev_best_iter
    std_of_best_reward = args.prev_best_std

    # for using correct iter num if model reused
    prev_iter_num = args.prev_iter_num

    # for TensorBoard summaries
    constraint_violation_count = args.prev_violations ## counting iters where constraint_value was > d_k
    constraint_value_sum = 0
    if not args.only_write_in_own_path:
        cum_constraint_value_sum = args.prev_cv_sum
    infeasible_count = args.prev_infeasibles ## counting iters where optimization problem was infeasible

    for i_iter in range(prev_iter_num, prev_iter_num+args.max_iter_num):
        t_start = time.time()

        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, i_iter)
        
        t0 = time.time()
        if args.algo_name == "CPO":
            v_loss, p_loss, cost_loss, constraint_value, opt_was_infeasible = update_params(batch, d_k)

            if constraint_value > d_k:
                constraint_violation_count += 1
            constraint_value_sum += constraint_value
            if not args.only_write_in_own_path:
                cum_constraint_value_sum += constraint_value
            if opt_was_infeasible:
                infeasible_count += 1
        elif args.algo_name == "TRPO":
            v_loss, p_loss, cost_loss, _, _ = update_params(batch)
        t1 = time.time()
        
        # update lists for saving
        v_loss_list.append(v_loss)
        p_loss_list.append(p_loss)
        cost_loss_list.append(cost_loss)
        rewards_std.append(log['std_reward']) 
        env_avg_reward.append(log['env_avg_reward'])
        num_of_steps.append(log['num_steps'])
        num_of_episodes.append(log['num_episodes'])
        tne = tne + log['num_episodes']
        tns = tns + log['num_steps']
        total_num_episodes.append(tne)
        total_num_steps.append(tns)

        # evaluate the current policy
        running_state.fix = True  #Fix the running state
        agent.num_threads = 20
        if args.env_name == "CartPole-v0" or args.env_name == "CartPole-v1" or args.env_name == "MountainCar-v0":
            agent.mean_action = False
        else:
            agent.mean_action = True
        seed = np.random.randint(1,1000)
        agent.env.seed(seed)
        _, eval_log = agent.collect_samples(args.eval_batch_size, i_iter)
        running_state.fix = False
        agent.num_threads = args.num_threads
        agent.mean_action = False
        
        # update eval lists
        eval_avg_reward.append(eval_log['env_avg_reward'])
        eval_avg_reward_std.append(eval_log['std_reward'])
        
        t_end = time.time()

        # print learning data on screen     
        if i_iter % args.log_interval == 0:
            ##print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_avg {:.2f}\tTest_R_avg {:.2f}\tTest_R_std {:.2f}'.format( i_iter, log['sample_time'], t1-t0, log['env_avg_reward'], eval_log['env_avg_reward'], eval_log['std_reward']))              
            print('{} | t: {:.2f}, {:.2f}, {:.1f}; {:.1f} | tr_R = {:.1f} |||   eval:  R = {:.1f}   std = {:.4f}   |||  d_k = {:.4f}  |  constr_val = {:.4f}  | c = {:.3f} | vios = {}'.format(
                    i_iter, log['sample_time'], eval_log['sample_time'], t1-t0, t_end-t_start, log['env_avg_reward'], eval_log['env_avg_reward'], eval_log['std_reward'], d_k, constraint_value, constraint_value-d_k, constraint_violation_count))
        
        # save the best model
        new_best_model_reached = eval_log['env_avg_reward'] >= best_avg_reward if not args.include_std_for_model_saving else (
            (
                iter_of_best_reward == -1
                and eval_log['env_avg_reward'] > 0
            ) or (
                iter_of_best_reward > -1
                and eval_log['env_avg_reward'] >= best_avg_reward
                and (
                    eval_log['std_reward'] <= std_of_best_reward
                    or eval_log['std_reward'] <= eval_log['env_avg_reward']*args.acceptable_reward_std_percentage/100
                )
            )
        )
        
        if new_best_model_reached:
            print('Saving new best model !!!!')
            to_device(torch.device('cpu'), policy_net, value_net)
            save_info_obj.save_models(policy_net, value_net, running_state)
            to_device(device, policy_net, value_net)
            best_avg_reward = eval_log['env_avg_reward']
            std_of_best_reward = eval_log['std_reward']
            iter_of_best_reward = i_iter

        ## update TensorBoard summary for sub-exp (weird order because the colors weren't arranged greatly)
        writer.add_scalars('_constraints', {'Jc':constraint_value}, i_iter)
        writer.add_scalars('_constraints', {'d':d_k}, i_iter)
        writer.add_scalars('_constraints', {'mean_Jc':constraint_value_sum/(i_iter-prev_iter_num+1)},i_iter)

        writer.add_scalars('_rewards', {'2_log_reward':log['env_avg_reward']}, i_iter)
        if new_best_model_reached:
            writer.add_scalars('_rewards', {'1_best_avg_reward':eval_log['env_avg_reward']}, i_iter)
        else:
            if i_iter == 0: ## first iter of this run and of this TensorBoard summary graph
                if best_avg_reward > 0:
                    writer.add_scalars('_rewards', {'1_best_avg_reward':0}, -1) ## so that in every graph, 0 is the best reward in the beginning
                writer.add_scalars('_rewards', {'1_best_avg_reward':best_avg_reward}, i_iter)
            if args.log_best_reward_in_last_iter and i_iter == prev_iter_num+args.max_iter_num-1: ## last iter and marked to be logged
                writer.add_scalars('_rewards', {'1_best_avg_reward':best_avg_reward}, i_iter)
        writer.add_scalars('_rewards', {'3_eval_log_reward':eval_log['env_avg_reward']}, i_iter)

        writer.add_scalars('counts', {'cum_constraint_violations':constraint_violation_count}, i_iter)
        writer.add_scalars('losses', {'v_loss':v_loss}, i_iter)
        writer.add_scalars('losses', {'p_loss':p_loss}, i_iter)
        writer.add_scalars('losses', {'cost_loss':cost_loss}, i_iter)
        writer.add_scalars('counts', {'infeasibles':infeasible_count}, i_iter)

        if not args.only_write_in_own_path: # update TensorBoard summary for whole exp
            writer2.add_scalars('_constraints', {'Jc':constraint_value}, i_iter)
            writer2.add_scalars('_constraints', {'d':d_k}, i_iter)
            writer2.add_scalars('_constraints', {'mean_Jc':cum_constraint_value_sum/(i_iter+1)},i_iter)

            writer2.add_scalars('_rewards', {'2_log_reward':log['env_avg_reward']}, i_iter)
            if new_best_model_reached:
                writer2.add_scalars('_rewards', {'1_best_avg_reward':eval_log['env_avg_reward']}, i_iter)
            else:
                if i_iter == 0: ## first iter of this run and of this TensorBoard summary graph
                    writer2.add_scalars('_rewards', {'1_best_avg_reward':best_avg_reward}, i_iter)
                if args.log_best_reward_in_last_iter and i_iter == prev_iter_num+args.max_iter_num-1: ## last iter and marked to be logged
                    writer2.add_scalars('_rewards', {'1_best_avg_reward':best_avg_reward}, i_iter)
            writer2.add_scalars('_rewards', {'3_eval_log_reward':eval_log['env_avg_reward']}, i_iter)

            writer2.add_scalars('counts', {'cum_constraint_violations':constraint_violation_count}, i_iter)
            writer2.add_scalars('losses', {'v_loss':v_loss}, i_iter)
            writer2.add_scalars('losses', {'p_loss':p_loss}, i_iter)
            writer2.add_scalars('losses', {'cost_loss':cost_loss}, i_iter)
            writer2.add_scalars('counts', {'infeasibles':infeasible_count}, i_iter)

        # save some intermediate models to sample trajectories from
        if args.save_intermediate_model > 0 and (i_iter+1) % args.save_intermediate_model == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            save_info_obj.save_intermediate_models(policy_net, value_net, running_state, i_iter)
            to_device(device, policy_net, value_net)
        
        if args.algo_name == "CPO" and not args.no_constraint_mode:
            d_k = d_k + d_k*e_k  # max constraint annealing 
        
        """clean up gpu memory"""
        torch.cuda.empty_cache()
        
    # dump expert_avg_reward, num_of_steps, num_of_episodes
    avg_reward = None
    save_info_obj.dump_lists(avg_reward, num_of_steps, num_of_episodes, total_num_episodes, total_num_steps, rewards_std, env_avg_reward, v_loss_list, p_loss_list, eval_avg_reward, eval_avg_reward_std)
    
    if not args.only_write_in_own_path:
        print('Best eval reward = {:.6f} with std = {:.6f} in iter {}  |  constr. vios: {},  infeas.: {}  |  cv_sum = {:.6f},  cum_cv_sum = {:.6f}  |  last d_k = {:.6f}.'.format(
            best_avg_reward, std_of_best_reward, iter_of_best_reward, constraint_violation_count, infeasible_count, constraint_value_sum, cum_constraint_value_sum, d_k))
    else:
        print('Best eval reward = {:.6f} with std = {:.6f} in iter {}  |  constr. vios: {},  infeas.: {}  |  cv_sum = {:.6f}  |  last d_k = {:.6f}.'.format(
            best_avg_reward, std_of_best_reward, iter_of_best_reward, constraint_violation_count, infeasible_count, constraint_value_sum, d_k))
    return best_avg_reward, std_of_best_reward, iter_of_best_reward

main_loop()
