from peft import get_peft_model, LoraConfig, TaskType

lora_configs = {}

lora_configs['llama2'] = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.SEQ_CLS
)

lora_configs['roberta8'] = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.SEQ_CLS
)

lora_configs['roberta32'] = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.SEQ_CLS
)

lora_configs['roberta64'] = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.SEQ_CLS
)

lora_configs['roberta128'] = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.SEQ_CLS
)