import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection

os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock

LANG_EMB_OBS_KEY = "lang_emb"
TOKENIZER_NAME = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
lang_emb_model = None
tz = None


def _load_language_model():
    global lang_emb_model, tz
    if lang_emb_model is None or tz is None:
        cache_dir = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
        lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            TOKENIZER_NAME,
            cache_dir=cache_dir,
        ).eval()
        tz = AutoTokenizer.from_pretrained(TOKENIZER_NAME, TOKENIZERS_PARALLELISM=True)


def get_lang_emb(lang):
    if lang is None:
        return None

    _load_language_model()
    
    tokens = tz(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    lang_emb = lang_emb_model(**tokens)['text_embeds'].detach()[0]

    return lang_emb

def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)
