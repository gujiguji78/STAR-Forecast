import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from models.ensemble_model import TimeSeriesLLM
from training.data_loader import create_loaders

def finetune_llm_real(config_path="configs/config.yaml",
                      base_ckpt="checkpoints/pretrained_model.pth",
                      out_ckpt="checkpoints/finetuned_with_llm.pth"):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    text_col = cfg["data"].get("text_col", "").strip()
    if not text_col:
        raise RuntimeError("你没有配置 data.text_col（真实文本列），因此不允许进行“LLM微调”（避免任何模拟/生成）。")

    # 启用LLM，要求本地模型路径
    cfg["llm"]["enable"] = True
    if not cfg["llm"].get("model_id_or_path", "").strip():
        raise RuntimeError("请先配置 llm.model_id_or_path 为本地离线模型目录（如 Qwen2.5-1.5B-Instruct）。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesLLM(cfg).to(device)

    ckpt = torch.load(base_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # 冻结预测网络，只训练LLM的LoRA参数
    for name, p in model.named_parameters():
        p.requires_grad = False
        if "llm_adapter" in name and "lora" in name.lower():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("没有可训练的LoRA参数：请确认 peft 已安装且 llm.use_lora=true。")

    opt = optim.AdamW(params, lr=float(cfg["training"].get("llm_lr", 1e-4)))

    train_loader, _ = create_loaders(
        cfg["data"]["train_csvs"],
        cfg["data"]["test_csvs"],
        value_col=cfg["data"]["value_col"],
        seq_len=int(cfg["seq_len"]),
        horizon=int(cfg["prediction_horizon"]),
        batch_size=int(cfg["training"].get("llm_batch_size", 2)),
        normalize=cfg["data"].get("normalize", "zscore"),
        text_col=text_col,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    llm = model.llm_adapter.llm
    tok = model.llm_adapter.tokenizer
    instruction = cfg["llm"].get("instruction_prompt", "请分析该时间序列。")

    epochs = int(cfg["training"].get("llm_epochs", 3))
    model.train()

    for ep in range(1, epochs + 1):
        total = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"LLM Finetune Epoch {ep}/{epochs}")
        for x, y, text in pbar:
            x, y = x.to(device), y.to(device)
            # text 是真实文本（不生成）
            texts = list(text)

            with torch.no_grad():
                out = model(x, return_features=True)
                fused = out["fused_features"]
                forecast = out["forecast"]

            prompts = []
            for i in range(len(texts)):
                pred = forecast[i].detach().cpu().tolist()
                pred_txt = ", ".join([f"{v:.4f}" for v in pred[: min(24, len(pred))]])
                # 真实监督：目标文本来自CSV
                prompts.append(f"{instruction}\n预测摘要：[{pred_txt}]\n请输出分析：{texts[i]}")

            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(llm.device)
            out2 = llm(**enc, labels=enc["input_ids"])
            loss = out2.loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=float(loss.item()))

        print(f"[LLM Finetune Epoch {ep}] avg_loss={total/max(1,steps):.6f}")

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out_ckpt)
    print(f"✓ 已保存微调后模型: {out_ckpt}")
    print("请把 config.yaml 的 api_model_ckpt 改成该文件，并 llm.enable=true，然后启动API测试解释输出。")

if __name__ == "__main__":
    finetune_llm_real()
