from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

model = Qwen2VLForConditionalGeneration.from_pretrained(
     "AdaptLLM/biomed-Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained( "AdaptLLM/biomed-Qwen2-VL-2B-Instruct")
dataset = load_dataset("hongrui/mimic_chest_xray_v_1")

train_dataset=dataset["train"]

main_list=[]
num=0
for factor in train_dataset:

    dict={}
    dict["img_path"]="train_data/1.jpeg"

    
    factor["image"].save("temporal.jpg")

    category=factor["text"]
    gt=factor["report"]

    

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "temporal.jpg",
                },
                {"type": "text", "text": "Describe this image. Also, analyze which diseases are in the image."},

            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


    reject_sen=output_text
    accept_sen=gt

    dict["reject"]=str(reject_sen)
    dict["accept"]=accept_sen
    dict["reference"]=factor["text"]
    main_list.append(dict)


    import json 
    
   

    import os 
    os.remove("temporal.jpg")
    num+=1
    print(num/len(train_dataset)*100)
    print("Done")


with open("train_annotations.json","w") as target:
    json.dump(main_list,target,indent=4)

