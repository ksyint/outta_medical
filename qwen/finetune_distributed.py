import torch
import json
import datetime
import os

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial

from util.vision_util import process_vision_info
from util.logutil import init_logger, get_logger

from accelerate import Accelerator, DeepSpeedPlugin
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=3,   
                    
    gradient_accumulation_steps=2,  
    zero3_save_16bit_model=True 
                                
)


accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)



device = accelerator.device
output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'

if accelerator.is_local_main_process:
    os.makedirs(output_dir, exist_ok=True)
    init_logger(output_dir)
    logger = get_logger()


class ToyDataSet(Dataset): 
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def find_assistant_content_sublist_indexes(l):


    start_indexes = []
    end_indexes = []


    for i in range(len(l) - 2):
       
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
           
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2)  
                    break  

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device):
    
    
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list) 
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

def write_chat_template(processor, output_dir):
    
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")

def train():

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="bfloat16", attn_implementation="flash_attention_2"
    )


    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

    train_loader = DataLoader(
        ToyDataSet("train_data/data.json"),
        batch_size=1,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    model.train()
    epochs = 10
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    for epoch in range(epochs):
        steps = 0
        for batch in train_loader:
            steps += 1
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.is_local_main_process:
                    logger.info(f"Batch {steps} of epoch {epoch + 1}/{epochs}, training loss : {loss.item()}")

   
    accelerator.wait_for_everyone()
    
    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_local_main_process:
        processor.save_pretrained(output_dir)
        write_chat_template(processor, output_dir)

if __name__ == "__main__":
    train()

