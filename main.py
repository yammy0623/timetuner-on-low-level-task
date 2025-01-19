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
    model_kwargs=model_kwargs,
    guidance_type='uncond',
)

# 3. Define TimeTuner for sampling.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)

# 4. Use corresponding pre-tuned `t_ratios` for sampling with NFE = 10.
x = time_tuner.ddim_sample(x=x_T,
                           num_steps=10
                           t_ratios=t_ratios,
                           eta=eta)