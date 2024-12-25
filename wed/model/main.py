import pyrallis
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from wed.model.config import RunConfig, Scheduler_Type
from wed.model.utils.enums_utils import model_type_to_size

@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)

def inversion_callback(pipe, step, timestep, callback_kwargs):
    return callback_kwargs

def inference_callback(pipe, step, timestep, callback_kwargs):
    return callback_kwargs

def run(init_image: Image, cfg: RunConfig, pipe_inversion, pipe_inference, latents = None, edit_prompt = None, edit_cfg = 1.0, noise = None):
    # pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))

    if latents is None and cfg.scheduler_type == Scheduler_Type.EULER or cfg.scheduler_type == Scheduler_Type.LCM or cfg.scheduler_type == Scheduler_Type.DDPM:
        g_cpu = torch.Generator().manual_seed(7865)
        img_size = model_type_to_size(cfg.model_type)
        VQAE_SCALE = 8
        latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
        noise = []
        for i in range(cfg.num_inversion_steps):
            noise.append(randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=g_cpu))
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
        pipe_inversion.scheduler_inference.set_noise_list(noise)
    
    if latents is not None and cfg.scheduler_type == Scheduler_Type.EULER or cfg.scheduler_type == Scheduler_Type.LCM or cfg.scheduler_type == Scheduler_Type.DDPM:
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
        pipe_inversion.scheduler_inference.set_noise_list(noise)
    

    pipe_inversion.cfg = cfg
    pipe_inference.cfg = cfg
    all_latents = None

    if latents is None:
        print("Inverting...")
        if cfg.save_gpu_mem:
            pipe_inference.to("cpu")
            pipe_inversion.to("cuda")
        res = pipe_inversion(prompt = cfg.prompt,
                        num_inversion_steps = cfg.num_inversion_steps,
                        num_inference_steps = cfg.num_inference_steps,
                        image = init_image,
                        guidance_scale = cfg.guidance_scale,
                        opt_iters = cfg.opt_iters,
                        opt_lr = cfg.opt_lr,
                        callback_on_step_end = inversion_callback,
                        strength = cfg.inversion_max_step,
                        denoising_start = 1.0-cfg.inversion_max_step,
                        opt_loss_kl_lambda = cfg.loss_kl_lambda,
                        num_aprox_steps = cfg.num_aprox_steps)
        latents = res[0][0]
        all_latents = res[1]
    
    inv_latent = latents.clone()

    if cfg.do_reconstruction:
        print("Generating...")
        edit_prompt = cfg.prompt if edit_prompt is None else edit_prompt
        guidance_scale = edit_cfg
        if cfg.save_gpu_mem:
            pipe_inversion.to("cpu")
            pipe_inference.to("cuda")
        img = pipe_inference(prompt = edit_prompt,
                            num_inference_steps = cfg.num_inference_steps,
                            negative_prompt = cfg.prompt,
                            callback_on_step_end = inference_callback,
                            image = latents,
                            strength = cfg.inversion_max_step,
                            denoising_start = 1.0-cfg.inversion_max_step,
                            guidance_scale = guidance_scale).images[0]
    else:
        img = pipe_inference(image = inv_latent,
                            prompt = scal.prompt_fw,
                            denoising_start=0.0,
                            num_inference_steps = config.num_inference_steps,
                            guidance_scale = 1.0,
                            omega=3, # omega=3 for "009698.jpg", omega=5 for "Arknight.jpg"
                            gamma=3, # gamma=3 for "009698.jpg", gamma=3 for "Arknight.jpg"
                            inv_latents=all_latents,
                            prompt_embeds_ref=other_kwargs[0],
                            added_cond_kwargs_ref=other_kwargs[1],
                            edit_threshold = edit_threshold,
                            edit_guidance_scale = edit_guidance_scale,
                            reverse_editing_direction = reverse_editing_direction,
                            t_exit = int(scal.t_exit), # t_exit=15 for "009698.jpg", t_exit=25 for "Arknight.jpg"
                            ).images[0]
                    
    return img, inv_latent, noise, all_latents

if __name__ == "__main__":
    main()
