from transformers import BertConfig, BertModel
from transformers import BertModel
import torch

# Building the config
config = BertConfig()
# Building the model from the config
model = BertModel(config)
print(config)


config = BertConfig()
model = BertModel(config)
# Model is randomly initialized!
model = BertModel.from_pretrained("bert-base-cased")
model.save_pretrained("directory_on_my_computer")
sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)


