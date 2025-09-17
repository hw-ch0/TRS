import torch
from torch.optim.optimizer import Optimizer, required
import math
from torch import Tensor
from typing import List, Optional

__all__ = ['quant_SGD','quant_Adam']

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        lr_scales: List[Tensor],  # new
        EMA_transitions: List[Tensor],  # new
        state_steps: List[int],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        momentum_tr: float,
        alpha: float,
        baseline: bool,
        flag: bool):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        lr_scale = lr_scales[i] # new
        EMA_transition = EMA_transitions[i] # new
        step = state_steps[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf


        if baseline:
            lr_new = lr
        else:
            if momentum_tr == 0 or not hasattr(param,'transition'):
                lr_new = lr
            else:
                if step == 1:
                    lr_new = alpha
                else:
                    EMA_transition.mul_(momentum_tr).add_(param.transition.sum()/param.numel(), alpha=1-momentum_tr)
                    update = lr_scale*lr/(EMA_transition+1e-8)
                    lr_scale.mul_(momentum_tr).add_(update, alpha=1-momentum_tr)
                    lr_new = alpha * lr_scale

        setattr(param, "d_p", d_p.clone())

        param.add_(d_p * (-lr_new))
        
   

class quant_SGD(Optimizer):
    def __init__(
        self, 
        params, 
        lr=required, 
        momentum=0, 
        dampening=0,
        weight_decay=0, 
        nesterov=False, 
        momentum_tr=0, 
        alpha=0.1, 
        baseline=True
        ):
        self.baseline = baseline
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        momentum_tr=momentum_tr, alpha=alpha)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(quant_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(quant_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            lr_scales = [] 
            EMA_transitions = [] 
            state_steps = []  
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            momentum_tr = group['momentum_tr'] 
            alpha = group['alpha'] 
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                        'quant_SGD does not support sparse gradients, '
                        'please use a dense gradient or a sparse-aware optimizer.'
                    )

                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    ### new
                    if len(state) == 0:
                        state['step'] = 0
                        state['lr_scale'] = torch.ones_like(p.view(-1)[0], memory_format=torch.preserve_format)
                        state['EMA_transition'] = torch.ones_like(p.view(-1)[0], memory_format=torch.preserve_format)*lr # avoid inf for initial steps
                    ###
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
                    lr_scales.append(state['lr_scale']) # new
                    EMA_transitions.append(state['EMA_transition']) # new
                    state['step'] += 1
                    state_steps.append(state['step'])

            sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  lr_scales,
                  EMA_transitions,
                  state_steps,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov,
                  momentum_tr=momentum_tr,
                  alpha=alpha,
                  baseline=self.baseline)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         lr_scales: List[Tensor],  # new
         EMA_transitions: List[Tensor],  # new
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         momentum_tr: float,
         alpha: float,
         baseline: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        lr_scale = lr_scales[i] # new
        EMA_transition = EMA_transitions[i] # new

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        if baseline:
            lr_new = float(lr)
        else:
            if momentum_tr == 0 or not hasattr(param,'transition'):
                lr_new = lr
            else:
                if step == 1:
                    lr_new = alpha
                else:
                    EMA_transition.mul_(momentum_tr).add_(param.transition.sum()/param.numel(), alpha=1-momentum_tr)
                    update = lr_scale*lr/(EMA_transition+1e-8)
                    lr_scale.mul_(momentum_tr).add_(update, alpha=1-momentum_tr)
                    lr_new = alpha * lr_scale
        
        step_size = lr_new / bias_correction1

        # for tracking
        d_p = exp_avg / denom / bias_correction1
        setattr(param, "d_p", d_p.clone())

        param.addcdiv_(exp_avg, denom, value=-step_size)

# https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py
class quant_Adam(Optimizer):
    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8,
        weight_decay=0, 
        amsgrad=False, 
        momentum_tr=0, 
        alpha=1e-3, 
        baseline=True
        ):
        self.baseline=baseline
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, 
                        momentum_tr=momentum_tr, alpha=alpha)
        super(quant_Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(quant_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            lr_scales = [] 
            EMA_transitions = [] 
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['lr_scale'] = torch.ones_like(p.view(-1)[0], memory_format=torch.preserve_format)
                        # avoid inf for initial steps
                        state['EMA_transition'] = torch.ones_like(p.view(-1)[0], memory_format=torch.preserve_format)*group['lr'] 

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    lr_scales.append(state['lr_scale']) # new
                    EMA_transitions.append(state['EMA_transition']) # new

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam(params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lr_scales,
                EMA_transitions,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                momentum_tr=group['momentum_tr'],
                alpha=group['alpha'],
                baseline=self.baseline)
        return loss