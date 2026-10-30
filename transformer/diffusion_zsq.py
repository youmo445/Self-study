import torch

def make_beta_schedule(num_steps, beta_begin, beta_end, device: torch.device|None=None):
    beta = torch.linspace(beta_begin, beta_end,num_steps, device=device)
    return beta

def precompute_diffusion_items(beta):
    alpha = 1 - beta
    alpha_bars = torch.cumprod(alpha, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
    return alpha, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars


def extract(
    values: torch.Tensor,
    t: torch.Tensor,
    x_shape: tuple[int, ...],
) -> torch.Tensor:
    index = values[t] # [B]
    index = index.reshape(-1, * ([1] * (len(x_shape) - 1)))
    return index

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bars = extract(sqrt_alpha_bars, t, x0.shape)
    sqrt_one_minus_alpha_bars = extract(sqrt_one_minus_alpha_bars, t, x0.shape)
    x_t = sqrt_alpha_bars * x0 + sqrt_one_minus_alpha_bars * noise
    return x_t, noise

def precompute_posterior_items(
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    alpha_bars_prev = torch.cat([torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]],dim=0)
    posterior_variance = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
    posterior_mean_coef_x0 = betas * torch.sqrt(alpha_bars_prev) / (1 - alpha_bars)
    posterior_mean_coef_xt = torch.sqrt(alphas) * (1 - alpha_bars_prev) / (1 - alpha_bars)
    return (
    alpha_bars_prev,
    posterior_variance,
    posterior_mean_coef_x0,
    posterior_mean_coef_xt,
)

def q_posterior_mean_variance(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    posterior_mean_coef_x0: torch.Tensor,
    posterior_mean_coef_xt: torch.Tensor,
    posterior_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coef_x0 = extract(posterior_mean_coef_x0, t, x0.shape)
    coef_xt = extract(posterior_mean_coef_xt, t, x_t.shape)
    variance = extract(posterior_variance, t, x0.shape)
    posterior_mean = coef_x0 * x0 + coef_xt * x_t
    return posterior_mean, variance

def predict_x0_from_noise(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    sqrt_alpha_bars = extract(sqrt_alpha_bars, t, x_t.shape)
    sqrt_one_minus_alpha_bars = extract(sqrt_one_minus_alpha_bars, t, x_t.shape)
    x0 = (x_t - sqrt_one_minus_alpha_bars * pred_noise) / sqrt_alpha_bars
    return x0

def p_sample_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
    posterior_mean_coef_x0: torch.Tensor,
    posterior_mean_coef_xt: torch.Tensor,
    posterior_variance: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_x0 = predict_x0_from_noise(x_t, t, pred_noise, sqrt_alpha_bars,sqrt_one_minus_alpha_bars)
    posterior_mean, variance = q_posterior_mean_variance(pred_x0, x_t, t, posterior_mean_coef_x0, posterior_mean_coef_xt, posterior_variance)
    if noise is None:
        noise = torch.randn_like(x_t)
    x_prev = posterior_mean + torch.sqrt(variance) * noise
    is_clean = (t == 0).reshape(-1, *([1] * (x_t.ndim - 1)))
    x_prev = torch.where(is_clean, posterior_mean, x_prev)
    return x_prev, pred_x0

def ddim_sample_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    pred_noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_x0 = predict_x0_from_noise(x_t, t, pred_noise, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
    is_clean = (t_prev==-1).reshape(-1, *([1] * (x_t.ndim-1)))
    sqrt_alpha_bars_prev = extract(sqrt_alpha_bars, t_prev, x_t.shape) 
    sqrt_one_minus_alpha_bars_prev = extract(sqrt_one_minus_alpha_bars, t_prev, x_t.shape)
    sqrt_alpha_bars_prev = torch.where(is_clean, torch.ones_like(sqrt_alpha_bars_prev), sqrt_alpha_bars_prev)
    sqrt_one_minus_alpha_bars_prev = torch.where(is_clean, torch.zeros_like(sqrt_one_minus_alpha_bars_prev), sqrt_one_minus_alpha_bars_prev)
    x_prev = sqrt_alpha_bars_prev * pred_x0 + sqrt_one_minus_alpha_bars_prev * pred_noise
    return x_prev, pred_x0

    
def diffusion_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor, 
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
    noise: torch.Tensor | None = None,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    x_t, noise = q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, noise)
    model_output = model(x_t, t, labels)
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v":
        target = velocity_target(x0, t, noise, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
    elif prediction_type =="x0":
        target = x0
    else:
        raise ValueError(f"Prediction type Not Support")
    loss = ((target - model_output) ** 2).mean()
    return loss

@torch.no_grad()
def p_sample_loop(
    model: torch.nn.Module,
    sample_shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    guidance_scale: float,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
    posterior_mean_coef_x0: torch.Tensor,
    posterior_mean_coef_xt: torch.Tensor,
    posterior_variance: torch.Tensor,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    device = next(model.parameters()).device
    labels = labels.to(device)
    num_steps = sqrt_alpha_bars.shape[0]
    x_t = torch.randn(sample_shape, device=device)
    for step in reversed(range(num_steps)):
        # 要构造一个[B]的t
        t = torch.full((sample_shape[0],), step, device=device, dtype=torch.long)
        pred_noise = predict_noise_cfg(model, x_t, t, labels, guidance_scale, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, prediction_type)
        x_prev, _ = p_sample_step(
            x_t,
            t,
            pred_noise,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars, 
            posterior_mean_coef_x0,
            posterior_mean_coef_xt,
            posterior_variance
            )
        x_t = x_prev
    return x_t

@torch.no_grad()
def ddim_sample_loop(
    model: torch.nn.Module,
    sample_shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    guidance_scale: float,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
    num_sample_steps: int,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    device = next(model.parameters()).device
    labels = labels.to(device)
    timestep = torch.linspace(sqrt_alpha_bars.shape[0]-1, 0, num_sample_steps).round().long()
    # 获得一个从最大值递减的序号序列
    x_t = torch.randn(sample_shape, device=device)
    for i in range(num_sample_steps):
        step = timestep[i].item()
        if i + 1 < num_sample_steps:
            # t_s 不是干净的
            prev_step = timestep[i+1].item()
        else: 
            prev_step = -1
        t = torch.full((sample_shape[0],), step, dtype = torch.long, device=device)
        t_prev = torch.full((sample_shape[0],), prev_step, dtype=torch.long, device=device)
        pred_noise = predict_noise_cfg(model, x_t, t, labels, guidance_scale, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, prediction_type)
        x_prev, _ = ddim_sample_step(
            x_t,
            t,
            t_prev,
            pred_noise,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
            )
        x_t = x_prev
    return x_t

def predict_noise_cfg(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    guidance_scale: float,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    model_output_cond = model(x_t, t, labels)
    null_labels = torch.full_like(labels, model.label_embedding.null_class_id)
    model_output_uncond = model(x_t, t, null_labels)
    guided_model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
    pred_noise, _ = model_output_to_noise_and_x0(x_t, t, guided_model_output, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, prediction_type)

    return pred_noise

def velocity_target(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    a_t = extract(sqrt_alpha_bars, t, x0.shape)
    b_t = extract(sqrt_one_minus_alpha_bars, t, x0.shape)
    v_target = - b_t * x0 + a_t * noise
    return v_target

def predict_x0_from_v(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_v: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    a_t = extract(sqrt_alpha_bars, t, x_t.shape)
    b_t = extract(sqrt_one_minus_alpha_bars, t, x_t.shape)
    pred_x_0 = a_t * x_t - b_t * pred_v
    return pred_x_0

def predict_noise_from_v(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_v: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    a_t = extract(sqrt_alpha_bars, t, x_t.shape)
    b_t = extract(sqrt_one_minus_alpha_bars, t, x_t.shape)
    pred_noise = b_t * x_t + a_t * pred_v
    return pred_noise

def predict_noise_from_x0(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_x0: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    a_t = extract(sqrt_alpha_bars, t, x_t.shape)
    b_t = extract(sqrt_one_minus_alpha_bars, t, x_t.shape)
    pred_noise = (x_t - a_t * pred_x0) / b_t
    return pred_noise

def model_output_to_noise_and_x0(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_output: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
    prediction_type: str = "epsilon",
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction_type == "epsilon":
        pred_noise = model_output
        pred_x0 = predict_x0_from_noise(x_t, t, pred_noise, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
    elif prediction_type == "v":
        pred_noise = predict_noise_from_v(x_t, t, model_output, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
        pred_x0 = predict_x0_from_v(x_t, t, model_output, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
    elif prediction_type == "x0":
        pred_x0 = model_output
        pred_noise = predict_noise_from_x0(x_t, t, pred_x0, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
    else:
        raise ValueError(f"Prediction Type Not Support!")
    return pred_noise, pred_x0