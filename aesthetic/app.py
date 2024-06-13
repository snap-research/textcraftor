import os
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
from PIL import Image, ImageFile
# import gradio as gr


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def load_models():
    model = MLP(768)

    device = "cuda"

    s = torch.load("aesthetic/sac+logos+ava1-l14-linearMSE.pth",
                   map_location='cpu')

    model.load_state_dict(s)
    model.to(device)
    model.half()
    model.eval()
    model.requires_grad_(False)

    model2, preprocess = clip.load("ViT-L/14", device=device)
    model2.eval()
    model2.requires_grad_(False)

    model_dict = {}
    model_dict['classifier'] = model
    model_dict['clip_model'] = model2
    model_dict['clip_preprocess'] = preprocess
    model_dict['device'] = device

    return model_dict


def predict(image, model_dict):
    image_input = model_dict['clip_preprocess'](image).unsqueeze(0).to(model_dict['device'])
    with torch.no_grad():
        image_features = model_dict['clip_model'].encode_image(image_input)
        if model_dict['device'] == 'cuda':
            im_emb_arr = normalized(image_features.detach().cpu().numpy())
            im_emb = torch.from_numpy(im_emb_arr).to(model_dict['device']).type(torch.cuda.FloatTensor)
        else:
            im_emb_arr = normalized(image_features.detach().numpy())
            im_emb = torch.from_numpy(im_emb_arr).to(model_dict['device']).type(torch.FloatTensor)

        prediction = model_dict['classifier'](im_emb)
    score = prediction.item()

    return {'aesthetic score': score}


if __name__ == '__main__':
    print('\tinit models')

    global model_dict

    model_dict = load_models()

    inputs = [gr.inputs.Image(type='pil', label='Image')]

    outputs = gr.outputs.JSON()

    title = 'image aesthetic predictor'

    examples = ['example1.jpg', 'example2.jpg', 'example3.jpg']

    description = """
    # Image Aesthetic Predictor Demo
    This model (Image Aesthetic Predictor) is trained by LAION Team. See [https://github.com/christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
    1. This model is desgined by adding five MLP layers on top of (frozen) CLIP ViT-L/14 and only the MLP layers are fine-tuned with a lot of images by a regression loss term such as MSE and MAE.
    2. Output is bounded from 0 to 10. The higher the better.
    """

    article = "<p style='text-align: center'><a href='https://laion.ai/blog/laion-aesthetics/'>LAION aesthetics blog post</a></p>"

    with gr.Blocks() as demo:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type='pil', label='Input image')
                submit_button = gr.Button('Submit')
            json_output = gr.JSON(label='Output')
        submit_button.click(predict, inputs=image_input, outputs=json_output)
        gr.Examples(examples=examples, inputs=image_input)
        gr.HTML(article)
    demo.launch()
