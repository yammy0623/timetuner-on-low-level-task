# python3.8
"""Contains the noise schedule, model decorator and class for TimeTuner."""

import torch

from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torchvision.utils as tvu
import os
from datasets import get_dataset, data_transform, inverse_data_transform

__all__ = ['NoiseScheduleVP', 'model_wrapper', 'TimeTuner']

class NoiseScheduleVP(object):
    """Create a wrapper class for the forward SDE (VP type).

    NOTE: We recommend to use `schedule=discrete` for the discrete-time
    diffusion models, especially for high-resolution images.

    The forward SDE ensures that the condition distribution

        q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).

    Therefore, we implement the functions for computing `alpha_t` and
    `sigma_t`. For t in [0, T], we have:

        log_alpha_t = self.marginal_log_mean_coeff(t)
        sigma_t = self.marginal_std(t)

    We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and
    continuous-time DPMs (trained on t in [t_0, T]).

    ===========================================================================

    1. For discrete-time DPMs:

        For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the
        discrete steps to continuous time steps by: t_i = (i + 1) / N, e.g.,
        for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.

        Args:
            betas: A `torch.Tensor`. The beta array for the discrete-time DPM.
                (See the original DDPM paper for details)
            alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the
                discrete-time DPM. (See the original DDPM paper for details)

        Note that we always have alphas_cumprod = cumprod(1 - betas).
        Therefore, we only need to set one of `betas` and `alphas_cumprod`.

    2. For continuous-time DPMs:

        We support the linear VPSDE for the continuous time setting. The
        hyperparameters for the noise schedule are the default settings in Yang
        Song's ScoreSDE:

        Args:
            beta_min: A `float` number. The smallest beta for the linear
                schedule.
            beta_max: A `float` number. The largest beta for the linear
            schedule.
            T: A `float` number. The ending time of the forward process.

    ===========================================================================

    Args:
        schedule: A `str`. The noise schedule of the forward SDE. `discrete`
            for discrete-time DPMs, and `linear` for continuous-time DPMs.
    Returns:
        A wrapper object of the forward SDE (VP type).

    ===========================================================================

    Example:

    # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ...,
    # N - 1):
    >>> ns = NoiseScheduleVP('discrete', betas=betas)

    # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for
    # n = 0, 1, ..., N - 1):
    >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

    # For continuous-time DPMs (VPSDE), linear schedule:
    >>> ns = NoiseScheduleVP('linear',
                             continuous_beta_0=0.1,
                             continuous_beta_1=20.)
    """

    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Initializes the noise schedule with basic settings."""
        if schedule not in ['discrete', 'linear']:
            raise ValueError(f'Unsupported noise schedule {schedule}. The '
                             f'schedule needs to be `discrete` or `linear`!')

        self._schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
                self.beta = betas
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self._T = 1.
            self._log_alpha_array = log_alphas.reshape(1, -1).to(dtype=dtype)
            self._total_N = self._log_alpha_array.shape[1]
            t_array = torch.linspace(0., 1., self.total_N + 1)
            self._t_array = t_array[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self._T = 1.
            self._total_N = 1000
            self._beta_0 = continuous_beta_0
            self._beta_1 = continuous_beta_1

    @property
    def schedule(self):
        return self._schedule

    @property
    def T(self):
        return self._T

    @property
    def total_N(self):
        return self._total_N

    def marginal_log_mean_coeff(self, t):
        """Compute log(alpha_t) for given continuous-time label t in [0, T]."""
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)),
                                  self._t_array.to(t.device),
                                  self._log_alpha_array.to(t.device)
                                  ).reshape((-1))
        elif self.schedule == 'linear':
            return (-0.25 * t ** 2 * (self.beta_1 - self.beta_0) -
                    0.5 * t * self.beta_0)

    def marginal_alpha(self, t):
        """Compute alpha_t of a given continuous-time label t in [0, T]."""
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """Compute sigma_t of a given continuous-time label t in [0, T]."""
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))
    
    def comput_alpha_ddnm(self, t):
        beta = torch.cat([torch.zeros(1).to(self.beta.device), self.beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    def marginal_alpha_ddnm(self, t):
        return self.comput_alpha_ddnm(t.long())

def model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    guidance_type='uncond',
    guidance_scale=1.,
    cond_process_fn=None,
    classifier_fn=None,
    model_kwargs=None,
    classifier_kwargs=None
):
    """Create a wrapper function for the noise prediction model.

    TimeTuner needs to use the continuous-time DPMs, since the optimized
    timesteps may be addressed out of the discrete schedule. For DPMs trained
    on discrete-time labels, we need to firstly wrap the model function to a
    noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. `noise`: noise prediction model. (Trained by predicting noise).

        2. `x_start`: data prediction model. (Trained by predicting the data
            x_0 at time 0).

        3. `v`: velocity prediction model. (Trained by predicting the
            velocity).

        4. `score`: marginal score function. (Trained by denoising score
            matching). Note that the score function and the noise prediction
            model follows a simple relationship:

                noise(x_t, t) = -sigma_t * score(x_t, t)

    We support three types of guided sampling by DPMs by setting
    `guidance_type`:
        1. `uncond`: unconditional sampling by DPMs.

        2. `classifier`: classifier guidance sampling by DPMs and another
        classifier.

        3. `classifier-free`: classifier-free guidance sampling by conditional
        DPMs.

    The `t_input` is the time label of the model, which may be discrete-time
    labels (i.e. 0 to 999) or continuous-time labels (i.e. epsilon to T).

    ===========================================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
        guidance_type: A `str`. The type of the guidance for sampling.
        guidance_scale: A `float`. The scale for the guided sampling.
        cond_process_fn: A function to pre-process condition employed in LDM.
        classifier_fn: A classifier function. Only used for the classifier
            guidance.
        model_kwargs: A `dict`. A dict for the other inputs of the model
            function.
        classifier_kwargs: A `dict`. A dict for the other inputs of the
            classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the
            `continuous time as the inputs.
    """
    model_kwargs = model_kwargs or dict()
    classifier_kwargs = classifier_kwargs or dict()

    def get_model_input_time(t_continuous):
        """Convert the continuous-time `t_continuous` (in [epsilon, T]) to the
        model input time.

        For discrete-time DPMs, we convert `t_continuous` in
        [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N]. For continuous-time
        DPMs, we just use `t_continuous`.
        """
        # only use schedule type in our case
        if noise_schedule.schedule == 'discrete':
            out_t = (t_continuous - 1. / noise_schedule.total_N) * 1000
            return out_t
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        # print(t_input.item())
        # t_seq.append(t_input)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        
        # by DDRM
        if output.size(1) == 6:
            output = output[:, :3]       
        
        if model_type == 'noise':
            return output
        elif model_type == 'x_start':
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return ((x - expand_dims(alpha_t, x.dim()) * output) /
                    expand_dims(sigma_t, x.dim()))
        elif model_type == 'v':
            alpha_t = noise_schedule.marginal_alpha(t_continuous),
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return (expand_dims(alpha_t, x.dim()) * output +
                    expand_dims(sigma_t, x.dim()) * x)
        elif model_type == 'score':
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -expand_dims(sigma_t, x.dim()) * output

    def cond_grad_fn(x, t_input, cond):
        """Compute the gradient of the classifier."""
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in,
                                     t_input,
                                     cond,
                                     **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    
    # 這邊會看noise pred fn 要 output 什麼
    def model_fn(x,
                 t_continuous,
                #  t_seq,
                 condition=None,
                 unconditional_condition=None,):
        """The noise predicition model function for TimeTuner."""
        if guidance_type == 'uncond':
            return noise_pred_fn(x, t_continuous, **model_kwargs)
        elif guidance_type == 'classifier':
            assert condition is not None
            t_input = get_model_input_time(t_continuous)
            if cond_process_fn is not None:
                condition = cond_process_fn(condition)
            if guidance_scale == 0.:
                return noise_pred_fn(x,
                                     t_continuous,
                                     cond=condition,
                                     **model_kwargs)
            else:
                assert classifier_fn is not None
                cond_grad = cond_grad_fn(x,
                                         t_input,
                                         cond=condition,
                                         **classifier_kwargs)
                # from noise schedule
                # DDNM
                # sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                sigma_t = noise_schedule.marginal_std(t_continuous)
                # model predict
                noise = noise_pred_fn(x,
                                    t_continuous,
                                    cond=condition,
                                    **model_kwargs)
                return (noise -
                        guidance_scale * expand_dims(sigma_t, x.dim())
                        * cond_grad)
        elif guidance_type == 'classifier-free':
            assert condition is not None
            if cond_process_fn is not None:
                condition = cond_process_fn(condition)
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x,
                                     t_continuous,
                                     cond=condition,
                                     **model_kwargs)
            else:
                assert unconditional_condition is not None
                if cond_process_fn is not None:
                    unconditional_condition = cond_process_fn(
                        unconditional_condition)
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in,
                                                    t_in,
                                                    cond=c_in,
                                                    **model_kwargs).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)
        # elif guidance_type == "ddnm":
        #     t_input = get_model_input_time(t_continuous)
        #     return ddnm_fn(x, t_continuous, y, condition=condition)

    assert model_type in ['noise', 'x_start', 'v', 'score']
    assert guidance_type in ['uncond', 'classifier', 'classifier-free']
    return model_fn


class TimeTuner(object):
    """The class for TimeTuner.
    
    TimeTuner is used to both train optimized timesteps and sample with the
    optimized timesteps accordingly.
    """
    def __init__(
        self,
        model_fn,
        noise_schedule,
        device='cuda',
    ):
        """Construct a TimeTuner. 

        Args:
            model_fn: A noise prediction model function which accepts the
                continuous-time input (t in [epsilon, T]).
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        """
        self._model = model_fn
        self._noise_schedule = noise_schedule
        self.device = device

    @property
    def noise_schedule(self):
        return self._noise_schedule

    def noise_predition_fn(self, x, t, condition, uncond_condition, **kwargs):
        """Return the noise prediction model."""
        t = t.expand((x.shape[0]))
        # print(t)
        return self._model(x, t, condition, uncond_condition)

    # single step denoising
    def ddim_step_fn(self,
                     x,
                     t,
                     s,
                     t_ratio=1.,
                     eta=0.,
                     noise=None,
                     condition=None,
                     uncond_condition=None):
        eps = self.noise_predition_fn(x,
                                      t * t_ratio,
                                      condition,
                                      uncond_condition)

        alpha_t = expand_dims(self.noise_schedule.marginal_alpha(t), x.dim())
        alpha_t_prev = expand_dims(self.noise_schedule.marginal_alpha(s),
                                   x.dim())
        # print(f"x shape: {x.shape}")
        # print(f"alpha_t shape: {alpha_t.shape}")
        # print(f"eps shape: {eps.shape}")

        sigma = (
            eta *
            torch.sqrt((1 - alpha_t_prev ** 2) / (1 - alpha_t ** 2)) *
            torch.sqrt(1 - alpha_t ** 2 / alpha_t_prev ** 2))
        x0_pred = (x - (1 - alpha_t ** 2).sqrt() * eps) / alpha_t

        if noise is not None:
            assert noise.shape == x.shape
        else:
            noise = torch.randn_like(x)
        mean_pred = (x0_pred * alpha_t_prev +
                     torch.sqrt(1 - alpha_t_prev ** 2 - sigma ** 2) * eps)

        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (x.ndim - 1))))
        x = mean_pred + nonzero_mask * sigma * noise
        return x, x0_pred

    def get_timesteps(self, num_steps, timesteps):
        """Get the continuous timesteps."""
        if timesteps is None:
            assert num_steps is not None
            step = self.noise_schedule.total_N // num_steps
            timesteps = torch.arange(
                0, self.noise_schedule.total_N, step).flip(0) + 1
        else:
            if not isinstance(timesteps, torch.Tensor):
                timesteps = torch.tensor(timesteps)
        if self.noise_schedule.schedule == 'discrete':
            timesteps = timesteps / 1000. + 1. / self.noise_schedule.total_N
        timesteps_prev = torch.cat(
            [timesteps[1:], torch.tensor([1. / self.noise_schedule.total_N])],
            dim=0)
        return timesteps.to(self.device), timesteps_prev.to(self.device)

    # single image sampling
    @torch.no_grad()
    def ddim_sample(self,
                    x,
                    num_steps=None,
                    timesteps=None,
                    t_ratios=None,
                    eta=0.,
                    condition=None,
                    uncond_condition=None,
                    return_intermediates=False,
                    verbose=False):
        timesteps, timesteps_prev = self.get_timesteps(num_steps, timesteps)
        if t_ratios is None:
            t_ratios = torch.ones_like(timesteps)
        else:
            assert timesteps.shape == t_ratios.shape
            if not isinstance(t_ratios, torch.Tensor):
                t_ratios = torch.tensor(t_ratios)
            t_ratios = t_ratios.to(self.device)

        intermediates = {'x_t': [x], 'x0_pred': [x]}
        total_steps = timesteps.shape[0]

        if verbose:
            iterator = tqdm(zip(timesteps, timesteps_prev, t_ratios),
                            desc='DDIM Sampler',
                            total=total_steps)
        else:
            iterator = zip(timesteps, timesteps_prev, t_ratios)
        self.t_seq = []
        for t, t_prev, t_ratio in iterator:
            x, x0_pred = self.ddim_step_fn(x,
                                           t=t,
                                           s=t_prev,
                                           t_ratio=t_ratio,
                                           eta=eta,
                                           condition=condition,
                                           uncond_condition=uncond_condition)
            intermediates['x_t'].append(x)
            intermediates['x0_pred'].append(x0_pred)
        print(self.t_seq)
        if return_intermediates:
            return x, intermediates
        return x
    
    # single image sampling
    @torch.no_grad()
    def ddnm_sample(self,
                    x,
                    args,
                    config,
                    A_funcs,
                    y,
                    classes,
                    num_steps=None,
                    timesteps=None,
                    t_ratios=None,
                    eta=0.,
                    condition=None,
                    uncond_condition=None,
                    return_intermediates=False,
                    verbose=False):
        
        timesteps, timesteps_prev = self.get_timesteps(num_steps, timesteps)
        if t_ratios is None:
            t_ratios = torch.ones_like(timesteps)
        else:
            assert timesteps.shape == t_ratios.shape
            if not isinstance(t_ratios, torch.Tensor):
                t_ratios = torch.tensor(t_ratios)
            t_ratios = t_ratios.to(self.device)

        intermediates = {'x_t': [x], 'x0_pred': [x]}
        total_steps = timesteps.shape[0]

        if verbose:
            iterator = tqdm(zip(timesteps, timesteps_prev, t_ratios),
                            desc='DDIM Sampler',
                            total=total_steps)
        else:
            iterator = zip(timesteps, timesteps_prev, t_ratios)
        
        print(timesteps)
        tvu.save_image(
                inverse_data_transform(x),
                os.path.join("images", f"start_x.png")
            )
        for t, t_prev, t_ratio in iterator:
            x, x0_pred = self.ddnm_step_fn(x,
                                           A_funcs,
                                           y,
                                           t=t,
                                           s=t_prev,
                                           t_ratio=t_ratio,
                                           eta=eta,
                                           condition=condition,
                                           uncond_condition=uncond_condition)
            intermediates['x_t'].append(x)
            intermediates['x0_pred'].append(x0_pred)
            tvu.save_image(
                    inverse_data_transform(x0_pred),
                    os.path.join("images", f"x0_pred.png")
                )
            tvu.save_image(
                    inverse_data_transform(x),
                    os.path.join("images", f"x_t.png")
                )
        if return_intermediates:
            return x, intermediates
        return x0_pred
    
    def ddnm_step_fn(self,
                     x,
                     A_funcs,
                     y,
                     t,
                     s,
                     t_ratio=1.,
                     eta=0.,
                     noise=None,
                     condition=None,
                     uncond_condition=None):
        
        # alpha_t = expand_dims(self.noise_schedule.marginal_alpha(t), x.dim())
        # alpha_t_prev = expand_dims(self.noise_schedule.marginal_alpha(s),
        #                            x.dim())
        print("t = ", t)
        print("s  = ", s)
        alpha_t =  self.noise_schedule.marginal_alpha_ddnm(t)
        alpha_t_prev =  self.noise_schedule.marginal_alpha_ddnm(s)
        # print(f"x shape: {x.shape}")
        # if cls_fn == None:
        # eps = self.noise_predition_fn(x,
        #                             t * t_ratio,
        #                             condition,
        #                             uncond_condition)
        # else:
        #     classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
        #     et = model(xt, t, classes)
        #     et = et[:, :3]
        #     et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)
        eps = self.noise_predition_fn(x,
                            t * t_ratio,
                            condition,
                            uncond_condition)
        if eps.size(1) == 6:
            eps = eps[:, :3]
            
        # print(f"alpha_t shape: {alpha_t.shape}")
        # print(f"eps shape: {eps.shape}")

        # sigma = (
        #     eta *
        #     torch.sqrt((1 - alpha_t_prev ** 2) / (1 - alpha_t ** 2)) *
        #     torch.sqrt(1 - alpha_t ** 2 / alpha_t_prev ** 2))
        
        # denoise
        x0_pred = (x - (1 - alpha_t ** 2).sqrt() * eps) / alpha_t
        # add for ddnm
        x0_pred_hat = x0_pred - A_funcs.A_pinv(
            A_funcs.A(x0_pred.reshape(x0_pred.size(0), -1)) - y.reshape(y.size(0), -1)
        ).reshape(*x0_pred.size())

        c1 = (1 - alpha_t_prev).sqrt() * eta
        c2 = (1 - alpha_t_prev).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = alpha_t_prev.sqrt() * x0_pred_hat + c1 * torch.randn_like(x0_pred) + c2 * eps

        tvu.save_image(
                inverse_data_transform(x0_pred_hat),
                os.path.join("images", f"x0_final.png")
            )
        # # add noise 
        # if noise is not None:
        #     assert noise.shape == x.shape
        # else:
        #     noise = torch.randn_like(x)
        # mean_pred = (x0_pred * alpha_t_prev +
        #              torch.sqrt(1 - alpha_t_prev ** 2 - sigma ** 2) * eps)

        # # no noise when t == 0
        # nonzero_mask = ((t != 0).float().view(-1, *([1] * (x.ndim - 1))))
        # x = mean_pred + nonzero_mask * sigma * noise
        x = xt_next


        
        return x, x0_pred_hat

    # train data to obtain t_ratio
    def optimize_timesteps(self,
                           data_loader,
                           step_fn,
                           args,
                           config, 
                           encode_fn=None,
                           num_steps=None,
                           timesteps=None,
                           tune_type='sequential',
                           lr=2e-3,
                           total_iters=500,
                           verbose=False,
                           **kwargs):
        if tune_type not in ['sequential', 'parallel', 'ddnm']:
            raise ValueError(f'Unsupported tune type {tune_type}. The tune '
                             f'type needs to be `sequential` or `parallel`!')
        t_ratios = list()
        timesteps, timesteps_prev = self.get_timesteps(num_steps, timesteps)
        # print(timesteps)
        # print(timesteps_prev)
        deg = args.deg
        num_tuned_timesteps = len(timesteps) - 1

        A_funcs, sigma_y = get_A_funcs_sigmay(args, config)
        for idx in trange(num_tuned_timesteps):

            t_ratio = Variable(torch.ones(1)).cuda()
            t_ratio.requires_grad = True
            optimizer = Adam([t_ratio], lr=lr, betas=(0.9, 0.999))
            for cur_iter, data_dict in enumerate(data_loader):
                if cur_iter >= total_iters:
                    break
                # x = data_dict.get('image').to(self.device)
                x = data_dict[0].to(self.device)
                x_orig = data_transform(config, x)
                y = A_funcs.A(x_orig)
            
                b, hwc = y.size()
                if 'color' in deg:
                    hw = hwc / 1
                    h = w = int(hw ** 0.5)
                    y = y.reshape((b, 1, h, w))
                elif 'inp' in deg or 'cs' in deg:
                    pass
                else:
                    hw = hwc / 3
                    h = w = int(hw ** 0.5)
                    y = y.reshape((b, 3, h, w))
                    
                if args.add_noise: # for denoising test
                    y = get_gaussian_noisy_img(y, sigma_y) 
                
                y = y.reshape((b, hwc))
                if encode_fn is not None:
                    x = encode_fn(x)
                # c = data_dict.get('label', None)
                c = data_dict[1]
                if c is not None and isinstance(c, torch.Tensor):
                    c = c.to(self.device)
                t = timesteps[idx]
                t_prev = timesteps_prev[idx]
                noise = torch.randn_like(x)

                with torch.no_grad():
                    if tune_type == 'sequential':
                        T = torch.tensor(self.noise_schedule.T).cuda()
                        alpha_T = self.noise_schedule.marginal_alpha(T)
                        sigma_T = self.noise_schedule.marginal_std(T)
                        x_inter = x * alpha_T + noise * sigma_T
                        for s, s_prev, ratio in zip(timesteps[:idx],
                                                    timesteps_prev[:idx],
                                                    t_ratios[:idx]):
                            x_inter, _ = step_fn(x_inter,
                                                 s,
                                                 s_prev,
                                                 ratio,
                                                 condition=c,
                                                 **kwargs)
                        x_t = x_inter
                    elif tune_type == 'ddnm':
                        # step_fn = ddnm_fn

                        T = torch.tensor(self.noise_schedule.T).cuda()
                        alpha_T = self.noise_schedule.marginal_alpha_ddnm(T)
                        # alpha_T = self.noise_schedule.marginal_alpha(T)
                        # sigma_T = self.noise_schedule.marginal_std(T)
                        x_inter = x * alpha_T.sqrt() + noise * (1-alpha_T).sqrt()
                        x_inter = noise
                        tvu.save_image(
                            inverse_data_transform(x_inter),
                            os.path.join("images", f"start_x.png")
                        )
                        for s, s_prev, ratio in zip(timesteps[:idx],
                                                    timesteps_prev[:idx],
                                                    t_ratios[:idx]):
                            x_inter, x0_t= step_fn(x_inter,
                                                 A_funcs,
                                                 y,
                                                 s,
                                                 s_prev,
                                                 ratio,
                                                 condition=c,
                                                 **kwargs)
                            tvu.save_image(
                                inverse_data_transform(x0_t),
                                os.path.join("images", f"x0_t.png")
                            )
                        x_t = x_inter


                    else:
                        alpha_t = self.noise_schedule.marginal_alpha(t)
                        sigma_t = self.noise_schedule.marginal_std(t)
                        x_t = x * alpha_t + noise * sigma_t

                    eps_t = self.noise_predition_fn(x_t,
                                                    t,
                                                    condition=c,
                                                    uncond_condition=None,
                                                    **kwargs)
                x_t_prev, x0_t_prev = step_fn(x_t,
                                      A_funcs,
                                      y,
                                      t=t,
                                      s=t_prev,
                                      t_ratio=t_ratio,
                                      condition=c,
                                      **kwargs)
                tvu.save_image(
                        inverse_data_transform(x0_t_prev),
                        os.path.join("images", f"x0_t_prev.png")
                    )
                eps_t_prev = self.noise_predition_fn(x_t_prev,
                                                     t_prev,
                                                     condition=c,
                                                     uncond_condition=None,
                                                     **kwargs)
                print(eps_t_prev)
                loss = mean_flat((eps_t - eps_t_prev).square())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"eps_t: {eps_t.mean().item()}, eps_t_prev: {eps_t_prev.mean().item()}")

                if verbose:
                    msg = f'idx: {idx} / {num_tuned_timesteps}, '
                    msg += f'# iters: {cur_iter} / {total_iters}, '
                    msg += f'loss: {loss.item():.4f}, '
                    msg += f't_ratio: {t_ratio.item():.4f}'
                    print(msg)

            t_ratios.append(t_ratio.detach().cpu().item())
            print(t_ratios)


        return torch.tensor(t_ratios + [1.])


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2),
                       xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K),
                    torch.tensor(K - 2, device=x.device),
                    cand_start_idx)
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx),
                          start_idx + 2,
                          start_idx + 1)
    start_x = torch.gather(sorted_all_x,
                           dim=2,
                           index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x,
                         dim=2,
                         index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K),
                    torch.tensor(K - 2, device=x.device),
                    cand_start_idx)
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded,
                           dim=2,
                           index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded,
                         dim=2,
                         index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension
            is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


def mean_flat(tensor):
    return tensor.sum(dim=list(range(1, tensor.ndim))).mean(dim=0)

# function for ddnm

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def get_noisy_x(next_t, b, x0_t_hat):
    next_t = next_t.to('cuda')
    x0_t_hat = x0_t_hat.to('cuda')
    at_next = compute_alpha(b, next_t.long())
    xt_next = at_next.sqrt() * x0_t_hat + torch.randn_like(x0_t_hat) * (1 - at_next).sqrt()
    return xt_next


def get_A_funcs_sigmay(args, config):
    deg = args.deg
    A_funcs = None
    device = 'cuda'
    if deg == 'cs_walshhadamard':
        compress_by = round(1/args.deg_scale)
        from functions.svd_operators import WalshHadamardCS
        A_funcs = WalshHadamardCS(config.data.channels, config.data.image_size, compress_by,
                                torch.randperm(config.data.image_size ** 2, device=device), device)
    elif deg == 'cs_blockbased':
        cs_ratio = args.deg_scale
        from functions.svd_operators import CS
        A_funcs = CS(config.data.channels, config.data.image_size, cs_ratio, device)
    elif deg == 'inpainting':
        from functions.svd_operators import Inpainting
        loaded = np.load("exp/inp_masks/mask.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, device)
    elif deg == 'denoising':
        from functions.svd_operators import Denoising
        A_funcs = Denoising(config.data.channels, config.data.image_size, device)
    elif deg == 'colorization':
        from functions.svd_operators import Colorization
        A_funcs = Colorization(config.data.image_size, device)
    elif deg == 'sr_averagepooling':
        blur_by = int(args.deg_scale)
        from functions.svd_operators import SuperResolution
        A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
    elif deg == 'sr_bicubic':
        factor = int(args.deg_scale)
        from functions.svd_operators import SRConv
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        A_funcs = SRConv(kernel / kernel.sum(), \
                        config.data.channels,config.data.image_size, device, stride=factor)
    elif deg == 'deblur_uni':
        from functions.svd_operators import Deblurring
        A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(device), config.data.channels,
                            config.data.image_size, device)
    elif deg == 'deblur_gauss':
        from functions.svd_operators import Deblurring
        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
        A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
    elif deg == 'deblur_aniso':
        from functions.svd_operators import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                            config.data.image_size, device)
    else:
        raise ValueError("degradation type not supported")
    args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
    sigma_y = args.sigma_y

    return A_funcs, sigma_y

def get_y_from_x(A_funcs, deg, sigma_y, config, args):
    x_orig = data_transform(config, x_orig)

    y = A_funcs.A(x_orig)

    b, hwc = y.size()
    if 'color' in deg:
        hw = hwc / 1
        h = w = int(hw ** 0.5)
        y = y.reshape((b, 1, h, w))
    elif 'inp' in deg or 'cs' in deg:
        pass
    else:
        hw = hwc / 3
        h = w = int(hw ** 0.5)
        y = y.reshape((b, 3, h, w))
        
    if args.add_noise: # for denoising test
        y = get_gaussian_noisy_img(y, sigma_y) 
    
    y = y.reshape((b, hwc))

    Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, config.data.image_size,
                                        config.data.image_size)
    x = Apy

    if deg[:6] == 'deblur':
        Apy = y.view(y.shape[0], config.data.channels, config.data.image_size,
                            config.data.image_size)
    elif deg == 'colorization':
        Apy = y.view(y.shape[0], 1, config.data.image_size, config.data.image_size).repeat(1,3,1,1)
    elif deg == 'inpainting':
        Apy += A_funcs.A_pinv(A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level