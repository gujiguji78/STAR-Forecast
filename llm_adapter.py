import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMAdapter(nn.Module):
    """
    只负责：从本地加载LLM + 根据config生成解释文本
    不负责训练/微调（你后续要LoRA再加）
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.llm_cfg = cfg.get("llm", {})
        self.enabled = bool(self.llm_cfg.get("enabled", False))

        self.tokenizer = None
        self.model = None

        if not self.enabled:
            return

        model_path = self.llm_cfg.get("model_path")
        if not model_path:
            raise ValueError("llm.enabled=true 但 config里没有 llm.model_path")

        local_files_only = bool(self.llm_cfg.get("local_files_only", True))
        trust_remote_code = bool(self.llm_cfg.get("trust_remote_code", True))

        # 先用 fp16/bf16 自动
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, override_gen: dict | None = None) -> str:
        """
        override_gen：允许API里临时覆盖 generation 参数
        """
        if not self.enabled or self.model is None:
            return ""

        gen_cfg = dict(self.llm_cfg.get("generation", {}))
        if override_gen:
            gen_cfg.update({k: v for k, v in override_gen.items() if v is not None})

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # 放到模型所在设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 200)),
            temperature=float(gen_cfg.get("temperature", 0.3)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # 只截取生成部分（尽量避免重复prompt）
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
