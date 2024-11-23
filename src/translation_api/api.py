import time

import click
import torch
from flask import Flask, jsonify, request

from src.model.two_tower_model import TwoTowerModel

app = Flask(__name__)


@click.command()
@click.option(
    "--model_path",
    default="models/two_tower_model.pth",
    help="Path to the model file.",
)
@click.option("--port", type=int, default=5003)
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--input_dimension", type=int, default=384)
def main(**kwargs):

    global model
    # Load the model
    print("Loading TwoTowerModel...")
    model = TwoTowerModel(
        kwargs["input_dimension"], kwargs["input_dimension"]
    )  # Assuming 384 is the embedding dimension
    model.load_state_dict(torch.load(kwargs["model_path"], weights_only=True))
    model.eval()
    print("Model loaded.")

    app.run(host=kwargs["host"], port=kwargs["port"])


@app.route("/translate/user", methods=["POST"])
def translate_user():
    start_time = time.time()
    data = request.json
    user_embedding = torch.tensor(data.get("embedding"), dtype=torch.float32)
    with torch.no_grad():
        user_translation = (
            model.user_tower(user_embedding.unsqueeze(0)).squeeze().numpy()
        )
    response = jsonify({"translation": user_translation.tolist()})
    elapsed_time = time.time() - start_time
    print(f"User translation response time: {elapsed_time:.4f} seconds")
    return response


@app.route("/translate/item", methods=["POST"])
def translate_item():
    start_time = time.time()
    data = request.json
    item_embedding = torch.tensor(data.get("embedding"), dtype=torch.float32)
    with torch.no_grad():
        item_translation = (
            model.item_tower(item_embedding.unsqueeze(0)).squeeze().numpy()
        )
    response = jsonify({"translation": item_translation.tolist()})
    elapsed_time = time.time() - start_time
    print(f"Item translation response time: {elapsed_time:.4f} seconds")
    return response


if __name__ == "__main__":
    main()
