from PIL import Image
import torch

from wed.model.eunms import Model_Type, Scheduler_Type, Gradient_Averaging_Type, Epsilon_Update_Type
from wed.model.utils.enums_utils import model_type_to_size, get_pipes
from wed.model.config import RunConfig
from wed.model.main import run as run_model
from wed.schemas.genI_schemas import base64_to_image #, TextInput


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
image_size = model_type_to_size(Model_Type.SDXL_Turbo)
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

# cache_size = 10
# prev_configs = [None for i in range(cache_size)]
# prev_inv_latents = [None for i in range(cache_size)]
# prev_images = [None for i in range(cache_size)]
# prev_noises = [None for i in range(cache_size)]


def crdeimg(
        scal,
        edit_cfg: float,
        inersion_strength: float,
        avg_gradients: bool,
        first_step_range_start: int,
        first_step_range_end: int,
        rest_step_range_start: int,
        rest_step_range_end: int,
        lambda_ac: float,
        lambda_kl: float):

        global prev_configs, prev_inv_latents, prev_images, prev_noises
        noise_correction = True if scal.nc == "1" else False
        update_epsilon_type = Epsilon_Update_Type.OPTIMIZE if noise_correction else Epsilon_Update_Type.NONE
        avg_gradients_type = Gradient_Averaging_Type.ON_END if avg_gradients else Gradient_Averaging_Type.NONE

        first_step_range = (first_step_range_start, first_step_range_end)
        rest_step_range = (rest_step_range_start, rest_step_range_end)
        input_image = base64_to_image(scal.img_base64)
        original_shape = input_image.size
        

        config = RunConfig(model_type = model_type,
                    num_inference_steps = int(scal.numts),
                    num_inversion_steps = int(scal.numts), 
                    guidance_scale = 0.0,
                    max_num_aprox_steps_first_step = first_step_range_end+1,
                    num_aprox_steps = int(scal.numrs),
                    inversion_max_step = inersion_strength,
                    gradient_averaging_type = avg_gradients_type,
                    gradient_averaging_first_step_range = first_step_range,
                    gradient_averaging_step_range = rest_step_range,
                    scheduler_type = scheduler_type,
                    num_reg_steps = 4,
                    num_ac_rolls = 5,
                    lambda_ac = lambda_ac,
                    lambda_kl = lambda_kl,
                    update_epsilon_type = update_epsilon_type,
                    save_gpu_mem=False,
                    do_reconstruction = True)
        config.prompt = scal.prompt

        inv_latent = None
        noise_list = None
        # for i in range(cache_size):
        #     if prev_configs[i] is not None and prev_configs[i] == config and prev_images[i] == input_image:
        #         print(f"Using cache for config #{i}")
        #         inv_latent = prev_inv_latents[i]
        #         noise_list = prev_noises[i]
        #         prev_configs.pop(i)
        #         prev_inv_latents.pop(i)
        #         prev_images.pop(i)
        #         prev_noises.pop(i)
        #         break

        original_image = input_image.resize(image_size)

        res_image, inv_latent, noise, all_latents = run_model(original_image,
                                    config,
                                    latents=inv_latent,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    edit_prompt=scal.prompt_fw,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg)

        # prev_configs.append(config)
        # prev_inv_latents.append(inv_latent)
        # prev_images.append(input_image)
        # prev_noises.append(noise)
        
        # if len(prev_configs) > cache_size:
        #     print("Popping cache")
        #     prev_configs.pop(0)
        #     prev_inv_latents.pop(0)
        #     prev_images.pop(0)
        #     prev_noises.pop(0)

        return res_image.resize((image_size[0], int(image_size[0] * original_shape[1] / original_shape[0]))) 

def genI():
    return 0