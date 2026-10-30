import base64
from io import BytesIO


def convert_messages(messages):
    """Decode the evaluator's JPEG frames for the Qwen processor."""
    from PIL import Image
    converted = []
    
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append(dict(item))
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                prefix = "data:image/jpeg;base64,"
                if not url.startswith(prefix):
                    raise ValueError("Direct inference expects embedded JPEG frames")
                with Image.open(BytesIO(base64.b64decode(url[len(prefix):], validate=True))) as image:
                    content.append({"type": "image", "image": image.convert("RGB")})
            else:
                raise ValueError("Unsupported prompt content type")
        
        converted.append({"role": message["role"], "content": content})
    
    return converted


def generate_batch(model, processor, messages, max_new_tokens):
    import torch
    
    inputs = processor.apply_chat_template(
        [convert_messages(conversation) for conversation in messages],
        tokenize=True, add_generation_prompt=True, padding=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    return processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
