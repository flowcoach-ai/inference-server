import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File

import torch

from io import BytesIO
from mlp import MLP

INPUT_DIM = 102  # X is 102-dimensional
HIDDEN_DIM = INPUT_DIM * 4  # Center-most latent space vector will have length of 408
NUM_CLASSES = 16  # 16 classes -- aka: 16 unique instructions

app = FastAPI(
    title="FlowCoach",
    description="Realtime adjustments to your athletic practice",
    version="0.1")
ray.init(address="auto")
serve.start(detached=True)


@serve.deployment
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        model = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
        checkpoint = torch.load('model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.count = 0
        self.model = model

    def classify(self, features):
        input_tensor = torch.cat([features])
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            return {"instruction": int(torch.argmax(output_tensor[0]))}

    @app.get("/")
    def get(self):
        return "Welcome to the PyTorch model server."

    @app.post("/next_instruction")
    async def next_instruction(self, features):
        return self.classify(features)


ModelServer.deploy()