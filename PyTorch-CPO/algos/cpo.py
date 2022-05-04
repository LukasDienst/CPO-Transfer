import numpy as np
import torch
import scipy.optimize
from utils import *

class bcolors:
    MAGENTA = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BLACK = '\033[90m'
    DEFAULT = '\033[99m'


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size(), device=b.device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def cpo_step(env_name, policy_net, value_net, states, actions, returns, advantages, cost_advantages, constraint_value, d_k, max_kl, damping, l2_reg, use_line_search, use_fim=True):

    """update critic"""

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()
    
    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).detach().cpu().numpy(),
                                                            maxiter=25)
    v_loss,_ = get_value_loss(get_flat_params_from(value_net).detach().cpu().numpy())
    set_flat_params_to(value_net, tensor(flat_params))

    """update policy"""
    with torch.no_grad():
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """define the loss function for Objective"""
    def get_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
            return action_loss.mean()
        
    """define the loss function for Constraint"""
    def get_cost_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            cost_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
            return cost_loss.mean()      

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(v):
        M, mu, info = policy_net.get_fim(states)
        mu = mu.view(-1)
        filter_input_ids = set() if policy_net.is_disc_action else set([info['std_id']])

        t = ones(mu.size(), requires_grad=True, device=mu.device)
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, policy_net.parameters(), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        if not policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    """directly compute Hessian*vector from KL"""
    def Fvp_direct(v):
        kl = policy_net.get_kl(states)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping
    
    Fvp = Fvp_fim if use_fim else Fvp_direct
    
    # Obtain objective gradient and step direction
    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #g  
    grad_norm = False
    if grad_norm == True:
        loss_grad = loss_grad/torch.norm(loss_grad)
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10) #(H^-1)*g   
    if grad_norm == True:
        stepdir = stepdir/torch.norm(stepdir)
    
    # Obtain constraint gradient and step direction
    agent_data = torch.cat([states, actions], 1)
    cost_loss = get_cost_loss()
    cost_grads = torch.autograd.grad(cost_loss, policy_net.parameters(), allow_unused=True)
    cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]).detach() #b
    cost_loss_grad = cost_loss_grad/torch.norm(cost_loss_grad)
    cost_stepdir = conjugate_gradients(Fvp, -cost_loss_grad, 10) #(H^-1)*b
    cost_stepdir = cost_stepdir/torch.norm(cost_stepdir)
    
    # Define q, r, s, c
    q = -loss_grad.dot(stepdir) #g^T.H^-1.g
    r = loss_grad.dot(cost_stepdir) #g^T.H^-1.b
    s = -cost_loss_grad.dot(cost_stepdir) #b^T.H^-1.b

    d_k = tensor(d_k).to(constraint_value.dtype).to(constraint_value.device)
    cc = constraint_value - d_k # c would be positive for most part of the training

    ## check if optimization is feasible; if not, skip opt_lambda and opt_nu calculation
    opt_is_infeasible = ((cc**2)/s - max_kl) > 0 and cc>0

    if not opt_is_infeasible:
        ## find optimal lambda
        fun_a = lambda lamda: 1/(2*lamda)*((r**2)/s-q) + lamda/2*((cc**2)/s-max_kl) - r*cc/s
        fun_b = lambda lamda: -1/2*(q/lamda + lamda*max_kl)
        fun_lambda = lambda lamda: fun_a(lamda) if lamda*cc-r > 0 else fun_b(lamda)

        is_valid = lambda candidate: candidate >= 0 and not torch.isnan(candidate)

        def list_of_valid_candidates(candidates):
            list = []
            for candidate in candidates:
                if is_valid(candidate):
                    list.append(candidate)
            return list
        
        def list_of_valid_fun_values(candidates):
            list = []
            for candidate in candidates:
                list.append(fun_lambda(candidate))
            return list

        A = torch.sqrt((q - (r**2)/s)/(max_kl - (cc**2)/s))
        B = torch.sqrt(q/max_kl)

        valid_candidate_list = list_of_valid_candidates([A, B, r/cc, tensor(0)])
        valid_fun_value_list = list_of_valid_fun_values(valid_candidate_list)
        max_valid_fun_value = max(valid_fun_value_list)

        opt_lambda = tensor(np.inf) ## standard value, should be replaced in the following loop

        for candidate in valid_candidate_list:
            if max_valid_fun_value == fun_lambda(candidate):
                opt_lambda = candidate ## if several fun_values are the best: use the first best value
                break
        
        if opt_lambda <= 0 or torch.isnan(opt_lambda) or torch.isinf(opt_lambda):
            if opt_lambda == 0:
                print("opt_lambda would have been set to 0, but it was prevented as then opt_stepdir would contain Infs")
            opt_lambda = tensor(np.inf) # retain standard value
        
        if fun_lambda(tensor(np.inf)) > max_valid_fun_value: # this should only occurr when infeasible
            opt_lambda = tensor(np.inf)

        ## find optimal nu
        nu = (opt_lambda*cc - r)/s
        opt_nu = torch.max(nu, tensor(0))

        """ find optimal step direction """
        opt_stepdir = (stepdir - opt_nu*cost_stepdir)/opt_lambda
    else: ## optimization infeasible
        print(" – – – – – – –")
        """ find optimal step direction """
        opt_stepdir = -torch.sqrt(2*max_kl/s)*cost_stepdir
    
    if use_line_search: #find the maximum step length
        # perform line search
        prev_params = get_flat_params_from(policy_net)
        fullstep = opt_stepdir
        expected_improve = -loss_grad.dot(fullstep)
        success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
        set_flat_params_to(policy_net, new_params)
    else:
        # without line search
        prev_params = get_flat_params_from(policy_net)
        new_params = prev_params + opt_stepdir
        set_flat_params_to(policy_net, new_params)
    
    return v_loss, loss, cost_loss, opt_is_infeasible
