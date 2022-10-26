
from concurrent.futures import thread
from threading import Lock, Thread
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# import json
# with open("config.json") as json_file:
#     config = json.load(json_file)


class Model:
    def __init__(self):
        self.model = TFAutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.lock = Lock()

    def predict(self, prompt):
        with self.lock:
            input_ids = self.tokenizer(prompt, return_tensors="tf").input_ids
            gen_tokens = self.model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
            return self.tokenizer.batch_decode(gen_tokens)[0]

# use for DI
model = Model()
def get_model():
    return model
