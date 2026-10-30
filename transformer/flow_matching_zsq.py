import torch


def flow_interpolate(
    x_data: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x_data)
    t = t.reshape(-1, *([1] * (x_data.ndim - 1)))
    x_t = t * x_data + (1 - t) * noise
    target_v = x_data - noise
    return x_t, target_v

def flow_matching_loss(
    model: torch.nn.Module,
    x_data: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    x_t, target_v = flow_interpolate(x_data, t, noise)
    model_t = t * 1000.0
    pred_v = model(x_t, model_t, labels)
    loss = ((pred_v - target_v) ** 2).mean()
    return loss

@torch.no_grad()
def flow_euler_sample(
    model: torch.nn.Module,
    sample_shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    labels = labels.to(device)
    x_t = torch.randn(sample_shape, device=device)
    dt = 1 / num_steps
    for step in range(num_steps):
        t_value = step / num_steps
        t = torch.full((sample_shape[0],), t_value, device=device, dtype=x_t.dtype)
        pred_v = predict_velocity_cfg(model, x_t, t, labels, guidance_scale)
        x_t = x_t + pred_v * dt
    return x_t

def predict_velocity_cfg(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    model_t = t * 1000.0
    v_cond = model(x_t, model_t, labels)
    null_labels = torch.full_like(labels,model.label_embedding.null_class_id)
    v_uncond = model(x_t, model_t, null_labels)
    v = v_uncond + guidance_scale * (v_cond - v_uncond)
    return v