from time_tuner import NoiseScheduleVP, model_wrapper, TimeTuner


# 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

## 2. Convert your discrete-time `model` to the continuous-time
## noise prediction model. Here is an example for a diffusion model
## `model` with the noise prediction type.
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    guidance_type='classifier',
    guidance_scale=guidance_scale,
    classifier_fn=classifier,
    model_kwargs=model_kwargs,
    classifier_kwargs=classifier_kwargs,
)

# 3. Define TimeTuner for optimizing, together with the DDIM sampler.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)
step_fn = time_tuner.ddim_step_fn
step_fn_kwargs = dict(eta=eta)
tune_type = 'sequential'

# 4. Optimize the preset timesteps with NFE = 10.
t_ratios = time_tuner.optimize_timesteps(data_loader=data_loader,
                                         step_fn=step_fn,
                                         num_steps=10,
                                         tune_type=tune_type,
                                         lr=lr,
                                         total_iters=total_iters,
                                         verbose=True,
                                         **step_fn_kwargs)