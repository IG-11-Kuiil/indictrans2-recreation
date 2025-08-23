import os, sys, torch
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
for m in list(sys.modules):
    if m.startswith("transformers") or m.startswith("torchvision"):
        del sys.modules[m]

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "ai4bharat/indictrans2-en-indic-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device).eval()

ip = IndicProcessor(inference=True)

def translate(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", max_len=256, beams=5):
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, use_cache=True, max_length=max_len, num_beams=beams)
    outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
    outs = ip.postprocess_batch(outs, lang=tgt_lang)
    return outs

print(translate(["Hello, how are you?"], "eng_Latn", "hin_Deva")[0])
