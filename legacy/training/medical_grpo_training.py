# -*- coding: utf-8 -*-
"""
Medical GRPO training entrypoint.

This variant adapts the existing GRPO flow from math/reasoning tasks to
medical triage and safety alignment. It expects structured JSON/JSONL rows
with at least:
- question
- triage_label
- red_flags
- unsafe_actions
- preferred_actions
- must_mentions
"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    tokenizer_name_or_path: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    train_samples: Optional[int] = field(default=-1)
    eval_samples: Optional[int] = field(default=-1)
    eval_split_ratio: Optional[float] = field(default=0.1)
    preprocessing_num_workers: Optional[int] = field(default=4)
    qlora: bool = field(default=False)
    reward_components: Optional[str] = field(
        # Simplified to 3 components based on ablation evidence:
        # - safety_penalty removed: unsafe_rate is 0% even without it, and it
        #   double-penalises undertriage already covered by triage_reward.
        # - mention_bonus removed: ablation shows negligible impact (<0.5%).
        # Both remain available in the registry for experimental use.
        default='triage,redflag,actionability'
    )
    triage_weight: float = field(default=1.2)        # up from 1.0: primary clinical signal
    redflag_weight: float = field(default=1.0)       # unchanged
    actionability_weight: float = field(default=0.8) # up from 0.7: compensates removed safety_penalty
    safety_penalty_weight: float = field(default=1.0)   # kept for opt-in use
    mention_bonus_weight: float = field(default=0.3)    # reduced; opt-in only


SYSTEM_PROMPT = (
    "你是一个谨慎的医疗助手。你的目标不是直接下诊断，而是基于症状识别危险程度，"
    "给出安全、清晰、可执行的下一步建议。高风险情况要明确提示立即急诊或呼叫急救；"
    "用药问题要避免危险建议；如果信息不足，可以说明需要线下评估。"
)


TRIAGE_PATTERNS = {
    # home_observe: home-management words only; do NOT include clinic/urgency words
    "home_observe": [
        r"先在家", r"居家观察", r"居家处理", r"居家对症", r"在家休息", r"在家处理",
        r"先休息", r"先对症", r"先观察.*在家", r"可以在家",
    ],
    # outpatient: clinic-scheduling words; intentionally excludes 尽快就医/尽快去医院
    # (those belong to urgent, ensuring a clear tier boundary)
    "outpatient": [
        r"门诊", r"预约.*(门诊|医生|就诊)", r"线下就诊", r"挂号", r"看.*医生",
        r"就诊.*门诊",
    ],
    # urgent: explicit same-day / soon evaluation; NOT negated forms
    "urgent": [
        r"尽快就医", r"尽快去医院", r"尽快到医院", r"尽快评估", r"今天就医",
        r"尽快线下", r"尽快就诊", r"今天去医院",
    ],
    # emergency: acute-care / 120 words
    "emergency": [
        r"急诊", r"立即就医", r"立刻就医", r"马上就医", r"拨打120",
        r"呼叫急救", r"立即去医院", r"叫救护车",
    ],
}

# ---------------------------------------------------------------------------
# Text preprocessing for triage inference
# ---------------------------------------------------------------------------

# Conditional markers that introduce escalation clauses.
# e.g. "先在家…如果加重，尽快就医" — the "尽快就医" is conditional, not primary.
_CONDITIONAL_RE = re.compile(
    r'如果|若(?!干)|一旦|万一|但如果|但若|但.*加重|加重时|病情加重|症状加重|若.*出现|当.*出现'
)

# Negated emergency/urgent phrases to strip before pattern matching.
# e.g. "不必去急诊" should not trigger emergency patterns.
_NEGATED_EMERGENCY_RE = re.compile(
    r'(?:不必|不用|无需|不需要|不建议|不要|并非|不是|没有必要)\S{0,6}'
    r'(?:急诊|120|立即就医|立刻就医|马上就医|呼叫急救|叫救护车)'
)


def _preprocess_for_triage(text: str) -> str:
    """Prepare response text for triage-level detection.

    Two transformations applied in order:
    1. Strip conditional escalation clauses — keeps only the primary
       recommendation before phrases like "如果/若/一旦/病情加重时".
    2. Remove negated emergency phrases — prevents "不必去急诊" from
       falsely triggering an emergency pattern.
    """
    # Step 1: truncate at the first conditional marker
    m = _CONDITIONAL_RE.search(text)
    if m:
        text = text[:m.start()]

    # Step 2: erase negated emergency/urgent expressions
    text = _NEGATED_EMERGENCY_RE.sub('', text)

    return text


RED_FLAG_PATTERNS = {
    "persistent_cough_over_3_weeks": [r"咳嗽.*三周", r"持续.*咳嗽"],
    "low_grade_fever": [r"低烧", r"低热"],
    "night_sweats": [r"夜间盗汗", r"盗汗"],
    "chest_pain": [r"胸痛", r"胸口.*痛", r"压榨样"],
    "cold_sweat": [r"冷汗", r"冒冷汗"],
    "pain_over_20_minutes": [r"20分钟", r"二十分钟"],
    "new_or_persistent_headache": [r"头痛", r"头疼"],
    "pregnancy_bleeding": [r"怀孕.*出血", r"阴道流血", r"阴道出血"],
    "abdominal_pain_during_pregnancy": [r"腹痛"],
    "persistent_high_fever": [r"高烧", r"39\.5", r"39度"],
    "dehydration_signs": [r"脱水", r"口干", r"尿少", r"头晕"],
    "anticoagulant_related_bleeding": [r"华法林", r"出血", r"牙龈.*出血"],
    "functional_impairment": [r"影响工作", r"影响生活"],
    "possible_psychological_distress": [r"情绪低落", r"失眠", r"心慌"],
    "one_sided_weakness": [r"一侧.*无力", r"肢体无力"],
    "slurred_speech": [r"说话.*含糊", r"言语.*不清"],
    "sudden_neurologic_deficit": [r"突然"],
    "drug_allergy": [r"过敏", r"皮疹"],
    "lip_swelling": [r"嘴唇.*肿"],
}

UNSAFE_PATTERNS = {
    "home_observation_only": [r"先观察看看", r"再观察几天", r"不用去医院"],
    "ignore_symptoms": [r"不用管", r"不需要处理"],
    "wait_until_tomorrow": [r"明天再说", r"明天再去"],
    "self_medication_only": [r"自己吃点药就行"],
    "immediate_antibiotics_without_evaluation": [r"直接吃抗生素", r"自己先吃头孢"],
    "double_dose_antipyretics": [r"多吃一片退烧药", r"双倍退烧药"],
    "automatic_er_for_all_fever": [r"只要发烧就去急诊"],
    "adult_dose_antipyretics": [r"按成人剂量"],
    "antidiarrheal_overuse_without_assessment": [r"先大量止泻药"],
    "ignore_dehydration": [r"口干没事", r"不用补液"],
    "stop_warfarin_without_medical_advice": [r"立刻停华法林", r"自己先停药"],
    "double_dose_antihypertensive": [r"补两片", r"双倍补服"],
    "self_prescribe_psychotropic_drugs": [r"自己买安眠药", r"自己买精神类药"],
    "sleep_it_off": [r"先睡一觉看看"],
    "continue_medication": [r"继续吃头孢"],
    "take_more_same_drug": [r"多吃一点头孢"],
}

PREFERRED_PATTERNS = {
    "visit_hospital_soon": [r"尽快就医", r"尽快去医院", r"尽快线下就诊"],
    "respiratory_evaluation": [r"呼吸科", r"肺部检查", r"进一步检查"],
    "rest_and_hydration": [r"休息", r"多喝水", r"补充水分"],
    "symptomatic_care": [r"对症", r"退热", r"含片", r"温水漱口"],
    "seek_care_if_worse": [r"加重.*就医", r"如果.*加重.*就医", r"呼吸困难.*就医"],
    "call_emergency": [r"拨打120", r"呼叫急救"],
    "go_to_er_immediately": [r"去急诊", r"立即就医", r"立刻就医", r"马上就医"],
    "schedule_outpatient_visit": [r"门诊", r"预约医生", r"线下就诊"],
    "seek_urgent_care_if_neurologic_symptoms": [r"肢体无力.*立即就医", r"呕吐.*立即就医", r"意识.*立即就医"],
    "seek_urgent_obstetric_assessment": [r"尽快就医", r"妇产科", r"产科急诊"],
    "home_monitoring": [r"先观察", r"在家观察"],
    "oral_fluids": [r"喝水", r"补液"],
    "correct_antipyretic_use": [r"按说明书", r"对乙酰氨基酚", r"布洛芬"],
    "oral_rehydration_solution": [r"口服补液盐", r"ORS"],
    "monitor_urine_and_dizziness": [r"尿量", r"头晕"],
    "contact_doctor_soon": [r"联系医生", r"尽快就医", r"尽快评估"],
    "do_not_double_dose": [r"不要补两片", r"不要双倍", r"不要一次吃两片"],
    "follow_prescription_or_contact_doctor_or_pharmacist": [r"咨询医生", r"咨询药师", r"按原处方"],
    "schedule_psych_or_primary_care_visit": [r"心理科", r"精神科", r"门诊评估", r"尽快就医"],
    "seek_immediate_help_if_self_harm_or_suicidal_ideation": [r"自伤", r"自杀", r"立即求助"],
    "stop_and_seek_medical_help_promptly": [r"停药", r"尽快就医", r"尽快评估"],
}

TRIAGE_ORDER = ["home_observe", "outpatient", "urgent", "emergency"]
TRIAGE_RANK = {name: idx for idx, name in enumerate(TRIAGE_ORDER)}

UNDERTRIAGE_PATTERNS = {
    "home_observe": TRIAGE_PATTERNS["home_observe"] + [r"观察一天", r"先在家休息", r"问题不大"],
    "outpatient": TRIAGE_PATTERNS["outpatient"] + [r"改天去门诊", r"抽空去门诊"],
}

HIGH_RISK_DELAY_PATTERNS = [
    r"先观察看看",
    r"再观察几天",
    r"不用去医院",
    r"明天再说",
    r"明天再去",
    r"先睡一觉看看",
    r"问题不大",
]

VAGUE_ACTION_PATTERNS = [
    r"多注意",
    r"看情况",
    r"先观察",
    r"必要时",
]



def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _completion_text(completion) -> str:
    return _normalize_text(completion[0]["content"])


def _matches_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def _infer_triage_level(text: str) -> Optional[str]:
    """Infer the primary triage level recommended in a model response.

    Uses preprocessed text (conditional clauses stripped, negated phrases
    removed) so that safety caveats like "如果加重，尽快就医" do not inflate
    the detected level above the model's actual primary recommendation.
    """
    clean = _preprocess_for_triage(text)
    matched = []
    for level, patterns in TRIAGE_PATTERNS.items():
        if _matches_any(clean, patterns):
            matched.append(level)
    if not matched:
        return None
    return max(matched, key=lambda x: TRIAGE_RANK[x])


def triage_reward(completions, triage_label, **kwargs):
    """Soft triage reward with asymmetric scoring.

    Scoring design rationale:
    - Exact match: full credit (1.0).
    - Under-triage (predicted lower than actual):
        Always penalised heavily because missing a high-acuity patient is
        clinically dangerous regardless of how far off we are.
        delta=-1 → -0.5;  delta≤-2 → -1.0
    - Over-triage for HIGH-acuity ground truth (urgent/emergency):
        Mild over-triage is acceptable (safety bias).
        delta=+1 → +0.5;  delta≥+2 → +0.2
    - Over-triage for LOW-acuity ground truth (home_observe/outpatient):
        Sending a routine patient to the ER wastes resources and trains
        the model to default to alarm mode.  Previously these got 0.45
        (positive reward), creating a systematic bias.
        delta=+1 → +0.2  (slight credit — better than nothing)
        delta≥+2 → -0.3  (penalty — clear over-escalation)
    - No triage signal detected:
        Penalise if the case was urgent/emergency (model missed critical cue);
        neutral for low-acuity cases where omitting a triage level is less
        dangerous.
    """
    rewards = []
    for completion, label in zip(completions, triage_label):
        text = _completion_text(completion)
        predicted = _infer_triage_level(text)

        gold_rank = TRIAGE_RANK[label]

        if predicted is None:
            # Model gave no detectable triage signal
            reward = -0.3 if gold_rank >= TRIAGE_RANK['urgent'] else -0.1
            rewards.append(reward)
            continue

        pred_rank = TRIAGE_RANK[predicted]
        delta = pred_rank - gold_rank

        if delta == 0:
            reward = 1.0
        elif delta < 0:
            # Under-triage: always penalise
            reward = -1.0 if abs(delta) >= 2 else -0.5
        else:
            # Over-triage: severity depends on ground-truth acuity
            is_low_acuity = gold_rank <= TRIAGE_RANK['outpatient']
            if is_low_acuity:
                # home_observe / outpatient: over-escalation is harmful
                reward = 0.2 if delta == 1 else -0.3
            else:
                # urgent / emergency: slight over-triage is acceptable
                reward = 0.5 if delta == 1 else 0.2

        rewards.append(reward)
    return rewards


def redflag_reward(completions, red_flags, **kwargs):
    rewards = []
    for completion, flags in zip(completions, red_flags):
        text = _completion_text(completion)
        if not flags:
            rewards.append(0.0)
            continue
        hit = 0
        for flag in flags:
            if _matches_any(text, RED_FLAG_PATTERNS.get(flag, [])):
                hit += 1
        rewards.append(hit / max(len(flags), 1))
    return rewards


def actionability_reward(completions, preferred_actions, **kwargs):
    rewards = []
    for completion, actions in zip(completions, preferred_actions):
        text = _completion_text(completion)
        if not actions:
            rewards.append(0.0)
            continue
        hit = 0
        for action in actions:
            if _matches_any(text, PREFERRED_PATTERNS.get(action, [])):
                hit += 1
        base = min(hit / max(len(actions), 1), 1.0)
        if hit == 0 and _matches_any(text, VAGUE_ACTION_PATTERNS):
            base -= 0.2
        rewards.append(max(base, -0.2))
    return rewards


def safety_penalty_reward(completions, unsafe_actions, triage_label, **kwargs):
    rewards = []
    for completion, actions, label in zip(completions, unsafe_actions, triage_label):
        text = _completion_text(completion)
        penalty = 0.0
        for action in actions:
            if _matches_any(text, UNSAFE_PATTERNS.get(action, [])):
                penalty -= 1.25

        predicted = _infer_triage_level(text)
        if label in {'urgent', 'emergency'}:
            if _matches_any(text, HIGH_RISK_DELAY_PATTERNS):
                penalty -= 0.8
            if predicted in {'home_observe', 'outpatient'}:
                penalty -= 0.8 if label == 'urgent' else 1.2
        elif label == 'outpatient':
            if predicted == 'home_observe' and _matches_any(text, UNDERTRIAGE_PATTERNS['home_observe']):
                penalty -= 0.35
        rewards.append(penalty)
    return rewards


def mention_bonus_reward(completions, must_mentions, **kwargs):
    rewards = []
    for completion, mentions in zip(completions, must_mentions):
        text = _completion_text(completion)
        if not mentions:
            rewards.append(0.0)
            continue
        hit = 0
        for mention in mentions:
            key = _normalize_text(mention)
            if key and key in text:
                hit += 1
        rewards.append(0.15 * hit / max(len(mentions), 1))
    return rewards


REWARD_FUNC_REGISTRY: Dict[str, Callable] = {
    'triage': triage_reward,
    'redflag': redflag_reward,
    'actionability': actionability_reward,
    'safety_penalty': safety_penalty_reward,
    'mention_bonus': mention_bonus_reward,
}


def _reward_weights(script_args: ScriptArguments) -> Dict[str, float]:
    return {
        'triage': script_args.triage_weight,
        'redflag': script_args.redflag_weight,
        'actionability': script_args.actionability_weight,
        'safety_penalty': script_args.safety_penalty_weight,
        'mention_bonus': script_args.mention_bonus_weight,
    }


def build_reward_funcs(script_args: ScriptArguments):
    names = [x.strip() for x in (script_args.reward_components or '').split(',') if x.strip()]
    if not names:
        raise ValueError('reward_components is empty; please provide at least one reward component')
    unknown = [x for x in names if x not in REWARD_FUNC_REGISTRY]
    if unknown:
        raise ValueError(f'Unknown reward component(s): {unknown}. Available: {sorted(REWARD_FUNC_REGISTRY)}')

    weights = _reward_weights(script_args)
    reward_funcs = []
    for name in names:
        base_fn = REWARD_FUNC_REGISTRY[name]
        weight = float(weights[name])

        def _make_weighted(fn, reward_name, reward_weight):
            def _wrapped(*args, **kwargs):
                values = fn(*args, **kwargs)
                if reward_weight == 1.0:
                    return values
                return [reward_weight * float(v) for v in values]
            _wrapped.__name__ = f'{reward_name}_reward'
            return _wrapped

        reward_funcs.append(_make_weighted(base_fn, name, weight))
    return reward_funcs


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        cls = bnb.nn.Linear4bit if int4 else bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if 'lm_head' in name or 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def grpo_train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    is_main_process = training_args.local_rank in [-1, 0]
    if is_main_process:
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not script_args.train_file_dir or not os.path.exists(script_args.train_file_dir):
        raise ValueError("medical_grpo_training.py requires --train_file_dir pointing to local JSON/JSONL files")

    dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    with training_args.main_process_first(desc="Dataset preparation"):
        dataset = dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['question']}
                ],
                'triage_label': x['triage_label'],
                'red_flags': x.get('red_flags', []),
                'unsafe_actions': x.get('unsafe_actions', []),
                'preferred_actions': x.get('preferred_actions', []),
                'must_mentions': x.get('must_mentions', []),
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing medical GRPO dataset" if is_main_process else None,
        )

    split = dataset.train_test_split(test_size=script_args.eval_split_ratio, seed=42)
    train_dataset = split['train']
    test_dataset = split['test']
    if script_args.eval_samples and script_args.eval_samples > 0:
        eval_count = min(len(test_dataset), script_args.eval_samples)
        test_dataset = test_dataset.shuffle(seed=42).select(range(eval_count))
    if is_main_process:
        logger.info(
            f"Prepared medical GRPO datasets: train={len(train_dataset)}, eval={len(test_dataset)}, "
            f"eval_split_ratio={script_args.eval_split_ratio}, eval_samples={script_args.eval_samples}"
        )

    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    quantization_config = None
    if model_args.load_in_4bit or model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    # DDP (torchrun) 下 device_map="auto" 与 FSDP/DDP 冲突，需置为 None
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
        device_map=None if is_ddp else "auto",
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if model_args.use_peft:
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    reward_funcs = build_reward_funcs(script_args)
    if is_main_process:
        logger.info(
            f"Medical GRPO reward setup: components={script_args.reward_components}, "
            f"weights={_reward_weights(script_args)}"
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )

    last_checkpoint = get_checkpoint(training_args)
    if is_main_process:
        logger.info(f'*** Starting medical GRPO training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.model.config.save_pretrained(training_args.output_dir)
        logger.info(f"Medical GRPO model saved to {training_args.output_dir}")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
