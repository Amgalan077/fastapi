import PIL
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from utils.model_func import class_id_to_label, load_model, transorm_image


app = FastAPI()

@app.get('/')
def return_info():
    return 'Hello!!!'


@app.post('/classify')
def classify(file: UploadFile = File(...)):
    image = PIL.Image.open(file.file)
    print(image)
    adapted_image = transorm_image(image)
    model = load_model()
    pred_index = model(adapted_image.unsqueeze(0)).argmax().item()
    result = jsonable_encoder(
        {
            'prediction': class_id_to_label(pred_index)
        }
    )

    return JSONResponse(content=result)