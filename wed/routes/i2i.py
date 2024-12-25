import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fastapi import APIRouter
from wed.schemas.genI_schemas import tai_rps, TextInput, image_to_base64
from wed.model.call import crdeimg


router = APIRouter()

@router.post('/edit')
def predict(scal : tai_rps):
    response = crdeimg(scal,
                        1.0, #edit_cfg
                        1.0, #inersion_strength
                        True, #avg_gradients
                        0, #first_step_range_start
                        5, #first_step_range_end
                        8, #rest_step_range_start
                        10, #rest_step_range_end
                        20.0, #lambda_ac
                        0.055, #lambda_kl
                    )

    return image_to_base64(response)

@router.post('/decrease')
def predictcx(scal : tai_rps):
    response = crdeimg(scal,
                        1.0, #edit_cfg
                        1.0, #inersion_strength
                        True, #avg_gradients
                        0, #first_step_range_start
                        5, #first_step_range_end
                        8, #rest_step_range_start
                        10, #rest_step_range_end
                        20.0, #lambda_ac
                        0.055, #lambda_kl
                    )
    # image = Image.open('/kaggle/input/dsgssdd/images.jpg')   
    return image_to_base64(response)
    # return 0

@router.post('/test')
def predictcx(txt: TextInput):
    # fix bug
    return {'txt': txt.txt}
