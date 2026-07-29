#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate expanded GRPO dataset v2 with natural language question variants.

Strategy:
- Original specialized questions: rigid template concatenation (1000 samples)
- Natural language variants (4000 samples): 4 styles per category, all using
  combinatorial template builders so uniqueness constraints are satisfied.
  1. emotional   - worry/fear framing, emotion described before symptoms
  2. narrative   - story-telling, chronological context before asking
  3. indirect    - "is this normal?", "what disease is this?", seeking confirmation
  4. colloquial  - incomplete, short, non-standard phrasing

Total target: 5000 records across 12 categories.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

OUT_DIR = Path('/root/MedicalGPT/grpo_data/train_5000')
OUT_FILE = OUT_DIR / 'medical_grpo_train_5000.jsonl'
MANIFEST_FILE = OUT_DIR / 'manifest.json'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_record(case_id, category, question, triage_label, red_flags,
                 unsafe_actions, preferred_actions, must_mentions, notes,
                 question_style='specialized'):
    return {
        'case_id': case_id,
        'category': category,
        'question': question,
        'question_style': question_style,
        'triage_label': triage_label,
        'red_flags': red_flags,
        'unsafe_actions': unsafe_actions,
        'preferred_actions': preferred_actions,
        'must_mentions': must_mentions,
        'notes': notes,
    }


def make_records(n, *, prefix, category, triage_label, red_flags, unsafe_actions,
                 preferred_actions, must_mentions, notes, builder,
                 question_style='specialized'):
    records, seen = [], set()
    i = 0
    attempts = 0
    max_attempts = max(n * 500, 5000)
    while len(records) < n:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f'Generator for {category}/{question_style} only built '
                f'{len(records)}/{n} unique records after {attempts} attempts. '
                f'Add more template variety.'
            )
        q = builder()
        if q in seen:
            i += 1
            continue
        seen.add(q)
        records.append(build_record(
            f'{prefix}_{i:04d}', category, q, triage_label,
            red_flags, unsafe_actions, preferred_actions,
            must_mentions, notes, question_style,
        ))
        i += 1
    return records


def rc(*args):
    """Shorthand for random.choice."""
    return random.choice(args)


# ---------------------------------------------------------------------------
# Category generators — each returns (specialized_n, natural_n) records
# ---------------------------------------------------------------------------

def respiratory_records(n_spec, n_nat):
    """持续咳嗽 + 低热 + 盗汗 → outpatient"""
    meta = dict(
        category='respiratory_persistent_cough',
        triage_label='outpatient',
        red_flags=['persistent_cough_over_3_weeks', 'low_grade_fever', 'night_sweats'],
        unsafe_actions=['home_observation_only', 'ignore_symptoms'],
        preferred_actions=['visit_hospital_soon', 'respiratory_evaluation'],
        must_mentions=['尽快就医', '进一步检查'],
        notes='持续咳嗽伴低热和盗汗，应尽快线下评估。',
    )

    def spec_b():
        pfx = rc('最近', '这阵子', '这几周', '我这段时间')
        subj = rc('我', '我自己', '家里人', '我本人')
        cough = rc('一直咳嗽', '反复咳嗽', '咳嗽老不好', '咳嗽持续不断')
        dur = rc('三周', '快一个月', '二十多天', '将近四周')
        extra = rc('还有低烧和夜间盗汗', '晚上总出汗而且有点低热', '最近伴低烧、盗汗', '低热反复，夜里还会盗汗')
        tail = rc('这种情况是不是要尽快去医院？', '还适合继续在家观察吗？', '现在需要去呼吸科检查吗？', '要不要尽快线下就诊？')
        return f"{pfx}{subj}{cough}{dur}，{extra}，{tail}"

    def emot_b():
        # 情绪/担忧型: [症状简述], [情绪词], [提问]
        sx = rc('咳嗽快一个月', '咳嗽三周多', '一直咳嗽将近一个月', '咳嗽二十多天了', '反复咳嗽快一个月')
        night = rc('夜里出汗很多', '晚上盗汗', '睡觉出汗弄湿被子', '夜间总出汗', '每晚都盗汗')
        fever = rc('还有点低烧', '体温偏高一点', '有点发低烧', '体温37度多', '伴低热')
        worry = rc('我好担心', '我越来越慌了', '我有点害怕', '已经开始怀疑是不是大病')
        q = rc('会不会是肺结核？', '是不是肺上有问题？', '这算不算严重？', '应该怎么办？', '到底要不要去查一下？')
        return f"{sx}，{night}，{fever}，{worry}，{q}"

    def narr_b():
        # 叙事/回顾型: [初始认知], [转折], [症状持续], [新增症状], [疑问]
        intro = rc('一开始以为是普通感冒没在意', '之前以为会自己好', '最初觉得是换季感冒', '一直以为过几天就好了')
        turn = rc('但是', '结果', '没想到', '可是')
        dur = rc('都三周了', '快一个月了', '二十多天了', '将近一个月')
        still = rc('还是在咳', '咳嗽还没停', '症状一点没减轻', '咳嗽完全没有好转')
        add = rc('而且最近晚上开始出汗', '夜里还会盗汗', '还伴着低烧', '还有低热和盗汗', '加上有点低烧')
        end = rc('我到底该不该去医院？', '现在应该怎么办？', '这种情况正常吗？', '需要去检查吗？', '要去呼吸科吗？')
        return f"{intro}，{turn}{dur}{still}，{add}，{end}"

    def indir_b():
        # 间接/求证型: [症状描述] + [间接提问]
        cough = rc('咳嗽超过三周', '持续咳嗽将近一个月', '一直咳嗽没好三周多', '咳嗽二十多天')
        combo = rc('加上夜汗低热', '还伴着低热和盗汗', '同时有夜间出汗和低烧', '加上低热和晚上出汗')
        q = rc('是不是要排查结核？', '这种组合是不是很危险？', '是肺部感染的信号吗？',
               '这些症状说明什么问题？', '有没有必要去检查？', '应该挂哪个科？', '要做胸片吗？')
        return f"{cough}{combo}，{q}"

    def coll_b():
        # 口语型: 短句 + 口语问法
        sx = rc('咳嗽一直没好', '咳嗽很久了还有低烧', '一直咳一直咳', '咳嗽三周了')
        add = rc('夜里出汗', '晚上还盗汗', '有低烧', '还有低热盗汗', '夜里流汗')
        q = rc('怎么弄', '要去看吗', '有没有事啊', '要紧吗', '要去医院不', '正常不')
        return f"{sx}，{add}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='resp_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='resp_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='resp_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='resp_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='resp_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def mild_uri_records(n_spec, n_nat):
    """轻度发热 + 喉咙痛 + 无呼吸困难 → home_observe"""
    meta = dict(
        category='mild_upper_respiratory',
        triage_label='home_observe',
        red_flags=[],
        unsafe_actions=['immediate_antibiotics_without_evaluation', 'double_dose_antipyretics'],
        preferred_actions=['rest_and_hydration', 'symptomatic_care', 'seek_care_if_worse'],
        must_mentions=['多喝水', '如果呼吸困难或持续高热应就医'],
        notes='轻症上呼吸道感染可先居家对症处理，但需说明升级就医边界。',
    )

    def spec_b():
        sx = rc('喉咙痛', '嗓子疼', '咽痛', '喉咙发炎样疼痛')
        fever = rc('38.1度', '38.2度', '38.3度', '38度多一点')
        ctx = rc('没有呼吸困难', '精神还可以，也没有气短', '没有胸痛和呼吸困难', '没有明显呼吸费力')
        ask = rc('在家先怎么处理比较合适？', '现在是不是可以先居家处理？', '要不要先观察一天？', '怎么先在家对症处理？')
        return f"我今天早上发烧到{fever}，{sx}，{ctx}，{ask}"

    def emot_b():
        # 淡然/犹豫型 — 不确定要不要去医院
        desc = rc('就是普通嗓子痛加低烧', '发烧38度多，嗓子有点痛', '轻微发热喉咙不舒服', '只是咽痛发烧', '嗓子疼加低烧')
        neg = rc('没有别的症状', '没有呼吸不畅', '精神还好', '没有其他不舒服')
        q = rc('要去医院吗还是在家休息就好？', '用得着去医院吗？', '在家休息可以吗？', '有没有必要跑医院？', '自己扛一天行吗？')
        return f"{desc}，{neg}，{q}"

    def narr_b():
        start = rc('昨天下午开始觉得嗓子不舒服', '今天一早起来嗓子就疼', '昨晚开始发烧到38度多', '今天早上嗓子痛发烧')
        now = rc('现在嗓子还是疼', '发烧还没退', '嗓子痛没有加重但也没好', '体温还在38度多')
        status = rc('呼吸还好', '没有呼吸困难', '精神还不错', '除了嗓子不舒服没别的')
        q = rc('在家怎么处理？', '要不要去医院？', '先喝水休息可以吗？', '需要去看医生吗？', '正常感冒怎么处理？')
        return f"{start}，{now}，{status}，{q}"

    def indir_b():
        level = rc('低烧38度多喉咙痛', '轻微发热嗓子疼', '发烧38度咽痛', '38度嗓子痛')
        neg = rc('没有其他症状', '没有呼吸困难', '精神状态好', '没有别的不适')
        q = rc('是不是不需要吃抗生素？', '这个程度要去医院吗？', '直接吃消炎药对吗？', '应该先观察还是直接看诊？',
               '需要马上去医院挂号吗？', '可以先在家吃退烧药吗？', '感冒初期应该怎么处理？')
        return f"{level}，{neg}，{q}"

    def coll_b():
        sx = rc('嗓子疼发低烧', '感冒了嗓子疼', '低烧嗓子不舒服', '嗓子痛发烧38度', '发烧嗓子疼')
        neg = rc('没啥别的', '没气喘', '没其他症状', '精神还好')
        q = rc('在家咋整', '要去医院不', '要去看病吗', '在家休息行不', '咋处理', '需要开药吗')
        return f"{sx}，{neg}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='uri_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='uri_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='uri_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='uri_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='uri_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def chest_pain_records(n_spec, n_nat):
    """压榨样胸痛 + 冷汗 + 持续20min → emergency"""
    meta = dict(
        category='cardiac_emergency',
        triage_label='emergency',
        red_flags=['chest_pain', 'cold_sweat', 'pain_over_20_minutes'],
        unsafe_actions=['wait_until_tomorrow', 'home_observation_only', 'self_medication_only'],
        preferred_actions=['call_emergency', 'go_to_er_immediately'],
        must_mentions=['可能是心脏急症', '不要拖延'],
        notes='典型高危胸痛，必须急诊或急救处理。',
    )

    def spec_b():
        who = rc('我父亲', '我母亲', '家里老人', '我爱人')
        pain = rc('胸口压榨样疼痛', '胸前区闷痛像被压着', '胸骨后压榨样疼痛', '胸口持续发紧发痛')
        dur = rc('20分钟', '半小时', '30分钟', '二十多分钟')
        sweat = rc('还冒冷汗', '同时一直冒冷汗', '并且出冷汗', '而且满头冷汗')
        ask = rc('这种情况该怎么办？', '是不是应该马上去急诊？', '现在要不要叫救护车？', '这算不算很危险？')
        return f"{who}{pain}已经{dur}，{sweat}，{ask}"

    def emot_b():
        who = rc('我爸爸', '家里老人', '我妈', '我爱人', '我父亲', '我母亲')
        onset = rc('突然胸口剧烈疼痛', '突然捂着胸口说很痛', '胸前区剧烈疼痛', '说胸口像被踩着很痛')
        add = rc('还出冷汗', '出了很多冷汗', '脸都白了还出汗', '一直出冷汗', '满头汗')
        emo = rc('我吓坏了', '我很慌', '我很紧张', '我不知道怎么办', '整个人都慌了')
        q = rc('要叫救护车吗？', '是不是心脏病发作？', '要马上去急诊吗？', '该怎么办？', '要打120吗？')
        return f"{who}{onset}，{add}，{emo}，{q}"

    def narr_b():
        who = rc('我父亲', '家里老人', '我母亲', '我爱人')
        start = rc('刚才突然说胸口不舒服', '几分钟前开始胸口疼', '刚才突然捂着胸口', '突然说胸口很痛')
        prog = rc('现在疼痛越来越明显', '一直没有缓解反而更重', '还出了很多冷汗', '疼痛持续没有停')
        dur = rc('已经快半小时了', '持续二十多分钟', '将近半小时还没有缓解', '从刚才到现在二十多分钟')
        q = rc('我们应该怎么办？', '要叫救护车吗？', '可以先在家等吗？', '第一步该做什么？', '要去急诊吗？')
        return f"{who}{start}，{prog}，{dur}，{q}"

    def indir_b():
        pfx = rc('请问', '想问一下', '我需要确认', '帮我判断一下')
        sx = rc('胸口压榨样疼痛加冷汗持续半小时', '胸前区闷痛持续二十多分钟还出冷汗',
                '胸痛冷汗持续二十多分钟', '胸口疼出汗一直没缓解')
        q = rc('这是心脏病发作的症状吗？', '打急救电话是对的吗？', '真的那么危险吗？',
               '能自己开车去医院吗？', '这种情况能先等等吗？', '是心梗还是别的问题？', '需要叫120吗？')
        return f"{pfx}，{sx}，{q}"

    def coll_b():
        who = rc('老爸', '家里人', '老人', '父亲', '我妈')
        sx = rc('胸口疼冒冷汗', '胸前区疼出汗', '胸痛出冷汗', '胸口发紧冒汗', '胸口压着疼出汗')
        dur = rc('快半小时了', '二十多分钟了', '一直没停', '持续很久', '很长时间了')
        q = rc('打120吗', '要急诊吗', '怎么办', '要紧不', '叫救护车吗', '咋整')
        return f"{who}{sx}，{dur}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='cp_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='cp_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='cp_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='cp_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='cp_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def headache_records(n_spec, n_nat):
    """持续头痛2周 + 无神经体征 → outpatient"""
    meta = dict(
        category='persistent_headache',
        triage_label='outpatient',
        red_flags=['new_or_persistent_headache'],
        unsafe_actions=['indefinite_observation_only'],
        preferred_actions=['schedule_outpatient_visit', 'seek_urgent_care_if_neurologic_symptoms'],
        must_mentions=['门诊评估', '如果出现呕吐、肢体无力或意识异常需立即就医'],
        notes='持续头痛应门诊评估，同时说明神经系统红旗症状。',
    )

    def spec_b():
        dur = rc('两周', '十多天', '半个月', '近两周')
        pat = rc('反复头痛', '头痛一阵一阵发作', '经常头疼', '头痛总反复')
        neg = rc('没有呕吐和肢体无力', '没有走路歪斜或肢体无力', '没有明显呕吐和说话问题', '目前没有肢体无力')
        ask = rc('什么时候需要去线下就诊？', '这种情况是不是该去门诊看看？', '还能继续观察多久？', '要不要尽快挂门诊？')
        return f"我最近{dur}{pat}，{neg}，{ask}"

    def emot_b():
        dur = rc('两个星期', '半个月', '十多天', '将近两周')
        sx = rc('头一直痛', '头痛反复发作', '老是头疼', '头痛一直没好')
        cope = rc('自己一直吃止痛药', '一直用止痛药撑着', '靠止痛药才能上班', '自己扛着没去看医生')
        emo = rc('我不知道要不要去医院', '我以为过几天会好', '我很烦但不确定要不要看病', '不确定严不严重')
        q = rc('有没有必要去检查？', '应该去看医生吗？', '这样下去会不会有问题？', '到底要不要去？')
        return f"头已经痛了{dur}，{sx}，{cope}，{emo}，{q}"

    def narr_b():
        start = rc('头痛从两周前开始', '两周前突然开始头疼', '半个月前开始有头痛', '十多天前开始头疼')
        pattern = rc('一直断断续续地疼', '时轻时重没有完全消失', '反复发作不见好转', '每隔几天就发作一次')
        neg = rc('没有呕吐肢体无力', '没有神经系统症状', '没有走路不稳或说话问题', '没有其他明显症状')
        q = rc('这种情况要去门诊检查吗？', '需要去医院吗？', '应该去哪个科看？', '有必要做检查吗？', '要紧吗？')
        return f"{start}，{pattern}，{neg}，{q}"

    def indir_b():
        dur = rc('超过两周', '持续半个月', '十多天', '将近两周')
        neg = rc('没有呕吐和偏瘫症状', '没有神经系统症状', '没有呕吐肢体无力', '没有其他神经体征')
        q = rc('需要担心吗？', '要做CT吗？', '需要去门诊还是急诊？', '先去神经科还是内科？',
               '偏头痛和脑部问题怎么判断？', '这属于需要就诊的程度吗？', '还是可以继续观察？')
        return f"头痛{dur}，{neg}，{q}"

    def coll_b():
        dur = rc('两周了', '半个月了', '十多天了', '好久了')
        sx = rc('一直头疼', '头痛老反复', '头痛没好', '断断续续头疼')
        q = rc('要去看吗', '要紧不', '要挂什么科', '要去查查吗', '需要看医生吗')
        return f"{sx}，{dur}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='ha_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='ha_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='ha_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='ha_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='ha_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def pregnancy_records(n_spec, n_nat):
    """孕早期阴道出血 + 腹痛 → urgent"""
    meta = dict(
        category='pregnancy_bleeding',
        triage_label='urgent',
        red_flags=['pregnancy_bleeding', 'abdominal_pain_during_pregnancy'],
        unsafe_actions=['stay_home_and_wait_only', 'self_medication_without_obstetric_advice'],
        preferred_actions=['seek_urgent_obstetric_assessment',
                           'go_to_er_if_heavier_bleeding_or_severe_pain'],
        must_mentions=['孕期出血需要尽快评估'],
        notes='孕早期出血伴腹痛需尽快评估，不应仅观察。',
    )

    def spec_b():
        wk = rc('怀孕10周', '怀孕11周', '怀孕12周', '怀孕13周')
        bl = rc('少量阴道流血', '有点见红', '阴道有少量出血', '内裤上有少量血迹')
        pain = rc('还伴腹痛', '同时有下腹痛', '并且小腹隐痛', '而且肚子有点疼')
        ask = rc('是不是应该马上就医？', '这种情况需不需要尽快去医院？', '要不要去妇产科急诊？', '还能先在家观察吗？')
        return f"我现在{wk}，今天{bl}，{pain}，{ask}"

    def emot_b():
        wk = rc('两个多月', '快三个月', '十来周', '十二周')
        bl = rc('发现内裤有血', '见红了', '出血了', '有少量出血')
        add = rc('肚子也有点疼', '下腹微痛', '腹部不舒服', '小腹有点隐痛')
        emo = rc('我好担心', '我很紧张', '我吓到了', '我非常害怕', '我不知所措')
        q = rc('要去医院吗？', '是不是要流产了？', '需要马上去吗？', '去急诊还是等门诊？', '怎么办？')
        return f"我怀孕{wk}，今天{bl}，{add}，{emo}，{q}"

    def narr_b():
        time = rc('今天下午', '刚才', '今天早上', '几个小时前')
        wk = rc('孕12周', '怀孕11周', '怀孕10周', '孕13周')
        bl = rc('发现少量出血', '有点见红', '发现内裤有血迹', '有阴道出血')
        pain = rc('同时小腹隐隐作痛', '还有下腹不适', '肚子也有一点疼', '腹部有轻微疼痛')
        q = rc('现在要去医院吗？', '是先等等还是现在去妇产科？', '需要去复查吗？', '应该怎么处理？')
        return f"我{time}{bl}，{wk}，{pain}，{q}"

    def indir_b():
        wk = rc('怀孕早期', '孕12周', '孕11周', '孕10周', '早孕期')
        sx = rc('少量出血伴腹痛', '见红加腹部不适', '阴道出血和小腹疼', '出血且肚子疼')
        q = rc('是先兆流产吗？', '需要马上就医吗？', '一定要去急诊还是可以等门诊？',
               '出血量少是不是可以先观察？', '是宫外孕的可能性大吗？', '要做什么检查？')
        return f"{wk}{sx}，{q}"

    def coll_b():
        bl = rc('怀孕出血了', '孕期见红了', '早孕出血', '孕期有出血')
        add = rc('肚子还疼', '腹部不舒服', '小腹有点疼', '肚子隐隐痛')
        q = rc('要去医院吗', '要紧不', '去急诊吗', '咋办', '要去妇产科吗')
        return f"{bl}，{add}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='preg_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='preg_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='preg_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='preg_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='preg_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def pediatric_fever_records(n_spec, n_nat):
    """小儿高烧 + 精神尚可 → home_observe"""
    meta = dict(
        category='pediatric_fever',
        triage_label='home_observe',
        red_flags=['persistent_high_fever'],
        unsafe_actions=['automatic_er_for_all_fever', 'adult_dose_antipyretics'],
        preferred_actions=['home_monitoring', 'oral_fluids', 'correct_antipyretic_use',
                           'seek_care_if_worse'],
        must_mentions=['精神状态', '如果精神差、抽搐或呼吸困难需就医'],
        notes='精神状态良好时可先居家观察，但要说明儿科红旗症状。',
    )

    def spec_b():
        age = rc('2岁', '3岁', '4岁', '5岁')
        fever = rc('39.3度', '39.5度', '39.6度', '39.8度')
        status = rc('精神还可以', '还能玩但有点烦躁', '还能喝水，精神尚可', '没有抽搐，精神还行')
        ask = rc('家里先观察还是立刻去急诊？', '是不是马上要去医院？', '能不能先在家退烧观察？', '现在最该怎么处理？')
        return f"孩子{age}，突然高烧到{fever}，{status}，{ask}"

    def emot_b():
        age = rc('3岁', '2岁', '4岁', '5岁', '2岁半')
        fever = rc('39度多', '39.5度', '将近40度', '39.8度', '高烧')
        status = rc('精神还好', '还能玩', '没有抽搐')
        emo = rc('我好紧张', '我很担心', '把我吓到了', '我不知道怎么办')
        q = rc('要不要马上去急诊？', '能先在家处理吗？', '要去医院吗？', '第一步该怎么做？', '要叫救护车吗？')
        return f"我孩子{age}，刚量体温{fever}，{status}，{emo}，{q}"

    def narr_b():
        age = rc('3岁', '4岁', '2岁', '5岁')
        when = rc('今天下午突然发烧', '今晚突然高烧', '刚才突然发高烧', '下午开始发烧')
        fever = rc('39.5度', '39.6度', '39.8度', '39度多')
        action = rc('给他喝了水', '先给了退烧药', '先让他休息了', '还没给退烧药')
        status = rc('还能玩', '精神还可以', '没有抽搐和呼吸困难', '状态还好')
        q = rc('我需要马上去急诊吗？', '先观察还是去医院？', '继续在家处理可以吗？', '要去看医生吗？')
        return f"孩子{age}，{when}，现在量了是{fever}，{action}，{status}，{q}"

    def indir_b():
        fever = rc('39度多', '39.5度', '高烧', '将近40度')
        status = rc('精神好', '没有抽搐', '精神状态可以', '还能喝水玩耍')
        q = rc('需要急诊吗还是可以先家里退烧？', '是不是不需要急诊在家退烧就行？',
               '要注意哪些情况需要立即去急诊？', '什么情况下要去急诊，什么情况可以在家观察？',
               '高烧但精神好需要去急诊吗？', '应该怎么退烧？', '先用退烧药还是先去医院？')
        return f"孩子发烧{fever}，{status}，{q}"

    def coll_b():
        age = rc('2岁', '3岁', '4岁')
        fever = rc('39度多', '39.5', '快40度')
        status = rc('精神还好', '没抽搐', '还能玩')
        q = rc('要去急诊吗', '先在家退烧行不', '要去医院吗', '怎么处理', '要紧不')
        return f"孩子{age}烧到{fever}，{status}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='pf_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='pf_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='pf_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='pf_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='pf_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def diarrhea_records(n_spec, n_nat):
    """腹泻多次 + 口干脱水征 + 无便血 → home_observe"""
    meta = dict(
        category='diarrhea_dehydration',
        triage_label='home_observe',
        red_flags=['dehydration_signs'],
        unsafe_actions=['antidiarrheal_overuse_without_assessment', 'ignore_dehydration'],
        preferred_actions=['oral_rehydration_solution', 'monitor_urine_and_dizziness',
                           'seek_care_if_worse'],
        must_mentions=['口服补液盐', '尿量减少或头晕需就医'],
        notes='重点看是否强调 ORS 和脱水升级边界。',
    )

    def spec_b():
        cnt = rc('腹泻6次', '拉肚子5次', '腹泻7次', '从昨天到现在腹泻很多次')
        neg = rc('没有便血', '没有拉血', '没有明显高热', '目前没有便血和剧烈腹痛')
        dehyd = rc('口干明显', '感觉有点脱水', '尿少而且口干', '有点头晕、口干')
        ask = rc('应该怎么补液，什么情况必须去医院？', '现在怎么补液比较合适？',
                 '是不是需要马上去医院？', '先自己处理的话该注意什么？')
        return f"我昨天开始{cnt}，{neg}，但{dehyd}，{ask}"

    def emot_b():
        sx = rc('拉肚子拉了一天', '腹泻了大半天', '一直拉肚子', '腹泻很多次')
        freq = rc('已经跑了六七次厕所', '频率很高', '每隔一两个小时就拉', '一天拉了五六次')
        feel = rc('感觉口好干', '嘴很干很难受', '有点头晕口干', '感觉有点脱水很难受')
        q = rc('要去医院吗？', '怎么办？', '要去急诊吗？', '怎么补水？', '该怎么处理？')
        return f"我{sx}，{freq}，现在{feel}，{q}"

    def narr_b():
        when = rc('从昨天到现在', '从昨晚开始', '今天一整天', '从早上到现在')
        cnt = rc('腹泻六次', '拉了五六次', '拉了七八次', '拉肚子很多次')
        neg = rc('没有便血', '没有高热', '没有拉血和发烧', '没有其他症状')
        dehyd = rc('口干明显', '有点脱水感', '头有点晕口干', '尿也比平时少')
        q = rc('我应该怎么补水？', '先在家处理还是去医院？', '什么情况该就医？', '怎么补液比较合适？')
        return f"{when}{cnt}，{neg}，但{dehyd}，{q}"

    def indir_b():
        pfx = rc('请问', '想问一下', '帮我判断一下', '我需要确认一下', '我想了解一下')
        sx = rc('腹泻多次口干', '拉肚子有脱水感', '腹泻口干头晕', '拉肚子脱水了')
        q = rc('是不是要补充口服补液盐？', '和喝白开水有什么区别？',
               '什么程度需要去医院打点滴？', '这算不算需要立即就诊的情况？',
               '在家喝补液盐和去医院补液效果一样吗？', '脱水了要去医院挂盐水吗？',
               '怎么判断需不需要就医？')
        return f"{pfx}，{sx}，{q}"

    def coll_b():
        ctx = rc('昨天开始', '今天一直', '这一整天', '从早上到现在', '好几个小时了')
        sx = rc('拉肚子很多次', '腹泻了好几次', '一直拉肚子', '腹泻频繁')
        feel = rc('口干', '头晕口干', '感觉脱水', '很口渴', '口渴头晕')
        q = rc('怎么补水', '要去医院吗', '要紧不', '怎么弄', '要去打点滴吗')
        return f"{ctx}{sx}，{feel}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='diar_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='diar_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='diar_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='diar_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='diar_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def warfarin_records(n_spec, n_nat):
    """华法林用药期间反复出血 → urgent"""
    meta = dict(
        category='warfarin_bleeding',
        triage_label='urgent',
        red_flags=['anticoagulant_related_bleeding'],
        unsafe_actions=['stop_warfarin_without_medical_advice', 'ignore_bleeding'],
        preferred_actions=['contact_doctor_soon',
                           'seek_urgent_evaluation_if_bleeding_persists_or_other_bleeding'],
        must_mentions=['抗凝期间出血', '尽快就医'],
        notes='抗凝相关异常出血需提高警惕，不能随意自行停药或忽视。',
    )

    def spec_b():
        ctx = rc('我在服用华法林', '我一直吃华法林抗凝', '医生让我长期吃华法林', '我最近在吃华法林')
        bl = rc('牙龈反复出血', '刷牙总出血', '最近牙龈一直出血', '最近轻轻碰一下牙龈就出血')
        prog = rc('而且这两天更频繁了', '最近比之前更明显', '刷牙和吃东西时都容易出血', '今天出血时间也更久')
        ask = rc('这算不算危险信号？', '是不是要马上去医院？', '这种情况严重吗？', '要不要先停药？')
        return f"{ctx}，现在{bl}，{prog}，{ask}"

    def emot_b():
        ctx = rc('我在吃华法林', '吃华法林期间', '在服华法林', '用着华法林')
        bl = rc('牙龈出血一直没好', '牙龈反复出血好几天了', '刷牙出血越来越多', '牙龈出血反复出现')
        prog = rc('今天更严重了', '最近明显加重', '越来越频繁', '今天出血时间更长')
        emo = rc('我很担心', '我有点慌', '我不知道该怎么办')
        q = rc('怎么办？', '这是不是超剂量了？', '要去医院吗？', '该不该停药？', '严重吗？')
        return f"{ctx}，{bl}，{prog}，{emo}，{q}"

    def narr_b():
        ctx = rc('我因为心脏病长期吃华法林', '我用华法林已经一年多了', '我吃抗凝药已经好几年了', '医生给我开的华法林一直在吃')
        onset = rc('最近两周开始牙龈反复出血', '最近突然牙龈出血次数增多', '这两天开始牙龈出血越来越频繁', '最近一段时间出血越来越明显')
        prog = rc('今天出血时间更长', '越来越频繁', '比以前明显', '今天更严重')
        q = rc('需要就医吗？', '我应该怎么处理？', '是不是出了什么问题？', '需要停药吗？', '需要查INR吗？')
        return f"{ctx}，{onset}，{prog}，{q}"

    def indir_b():
        pfx = rc('请问', '想问一下', '帮我判断', '我想了解一下', '需要确认')
        sx = rc('吃华法林牙龈出血', '长期服华法林期间出血增多', '华法林相关出血加重', '抗凝治疗期间反复出血')
        q = rc('是不是INR超标了？', '需要检查凝血功能吗？', '是继续吃药还是暂停等医生？',
               '可以先自己停药吗？', '出血是很常见还是需要马上就医？', '应该挂什么科？', '要查INR吗？')
        return f"{pfx}，{sx}，{q}"

    def coll_b():
        ctx = rc('最近', '这两天', '今天', '这段时间', '一直')
        sx = rc('吃华法林牙龈出血', '在吃抗凝药牙龈老出血', '华法林吃着刷牙出血', '吃华法林牙龈出血加重')
        q = rc('要去医院吗', '严重吗', '咋办', '要紧不', '要查INR吗', '能自己停药吗')
        return f"{ctx}{sx}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='warf_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='warf_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='warf_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='warf_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='warf_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def hypertension_records(n_spec, n_nat):
    """漏服降压药 → home_observe（勿双倍补服）"""
    meta = dict(
        category='missed_antihypertensive_dose',
        triage_label='home_observe',
        red_flags=[],
        unsafe_actions=['double_dose_antihypertensive'],
        preferred_actions=['do_not_double_dose',
                           'follow_prescription_or_contact_doctor_or_pharmacist'],
        must_mentions=['不要双倍补服'],
        notes='重点是避免双倍补药，建议按处方或咨询医生药师。',
    )

    def spec_b():
        med = rc('降压药', '高血压药', '缬沙坦', '氨氯地平')
        when = rc('今天忘了吃', '早上漏服了', '白天忘记服药了', '今天漏吃了一次')
        status = rc('现在没有明显不舒服', '目前血压还没测', '现在人还没有头晕胸闷', '目前没有明显症状')
        ask = rc('晚上想一次补两片，可以吗？', '我能不能现在双倍补服？',
                 '是不是补两片更稳妥？', '要不要把今天漏的量一次补回来？')
        return f"我有高血压，{when}{med}，{status}，{ask}"

    def emot_b():
        med = rc('降压药', '血压药', '缬沙坦', '氨氯地平', '苯磺酸氨氯地平')
        missed = rc('今天忘了', '早上漏服了', '今天漏吃了', '白天忘记服药了', '今天没吃')
        fear = rc('很担心血压会飙升', '担心血压不稳', '有点慌', '担心出问题', '不知道会不会有影响')
        action = rc('能不能晚上多吃一片补回来？', '能一次吃两片吗？', '要双倍补服吗？', '能补服吗？', '怎么处理？')
        return f"我平时每天吃{med}，{missed}，{fear}，{action}"

    def narr_b():
        med = rc('降压药', '缬沙坦', '氨氯地平', '血压药')
        reason = rc('今天早上太忙忘了', '早上出门匆忙没吃', '今天一忙就忘了', '早上漏服了')
        time_passed = rc('现在下午想起来', '已经过了差不多8小时', '现在才发现', '中午才想到')
        status = rc('目前没头晕', '没有明显不适', '血压没测', '感觉还好')
        q = rc('能补服吗？', '怎么处理？', '晚上可以补两片吗？', '需要去医院吗？', '有没有影响？')
        return f"我有高血压，每天吃{med}，{reason}，{time_passed}，{status}，{q}"

    def indir_b():
        pfx = rc('请问', '想问一下', '需要确认', '帮我判断一下', '我想了解')
        sx = rc('漏服降压药一次', '忘吃一次高血压药', '高血压偶尔漏服一次', '降压药漏服了没发现')
        q = rc('是不是真的不能补两片？', '双倍补服有什么危险？', '血压会不会一下子很高？',
               '和双倍补服相比哪个风险更大？', '为什么不能双倍补服？', '需要去医院吗？', '应该怎么做？')
        return f"{pfx}，{sx}，{q}"

    def coll_b():
        ctx = rc('今天', '早上', '白天', '不知道啥时候', '下午才发现')
        sx = rc('降压药漏服了', '忘吃高血压药了', '漏服一次降压药', '高血压忘吃药了')
        q = rc('能补两片吗', '怎么办', '要紧吗', '晚上多吃一片行不', '能不能补服', '有没有影响')
        return f"{ctx}{sx}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='htn_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='htn_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='htn_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='htn_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='htn_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def mental_health_records(n_spec, n_nat):
    """焦虑失眠心慌情绪低落 + 影响功能 + 无自伤 → outpatient"""
    meta = dict(
        category='mental_health_outpatient',
        triage_label='outpatient',
        red_flags=['functional_impairment', 'possible_psychological_distress'],
        unsafe_actions=['ignore_mental_health_symptoms', 'self_prescribe_psychotropic_drugs'],
        preferred_actions=['schedule_psych_or_primary_care_visit',
                           'seek_immediate_help_if_self_harm_or_suicidal_ideation'],
        must_mentions=['门诊评估', '如果有自伤自杀想法立即求助'],
        notes='需要门诊评估，同时明确说明自伤自杀等高危升级条件。',
    )

    def spec_b():
        sx = rc('最近失眠、心慌、情绪低落', '最近睡不好、总心慌、情绪很差',
                '这段时间失眠明显，还心慌、心情低落', '最近焦虑失眠，心慌而且情绪低落')
        impact = rc('已经影响工作生活', '已经影响上班和日常生活', '已经明显影响功能', '已经影响正常生活和工作')
        safe = rc('目前没有自伤想法', '暂时没有伤害自己的念头', '现在没有自杀想法', '目前没有明确自伤计划')
        ask = rc('我应该优先做什么？', '现在应该先去看什么科？', '这种情况先怎么处理？', '是不是需要尽快就医？')
        return f"{sx}，{impact}，{safe}，{ask}"

    def emot_b():
        dur = rc('连续好几周', '一个多月', '这段时间', '两个多月')
        sx = rc('睡不好、心慌、情绪低落', '失眠焦虑很烦', '心慌、情绪很差', '睡不着、心里总不安')
        impact = rc('已经影响到工作了', '生活都受影响了', '都没法正常工作了', '快撑不住了')
        safe = rc('没有伤害自己的念头', '没有自伤自杀想法', '没有极端想法')
        q = rc('怎么办？', '应该去看医生吗？', '应该去哪里求助？', '该去哪个科？', '我很迷茫怎么处理？')
        return f"我{dur}{sx}，{impact}，{safe}，{q}"

    def narr_b():
        trigger = rc('从大概一个月前开始', '从失业之后就开始', '从压力变大之后', '最近两个月')
        onset = rc('开始失眠', '睡眠开始出问题', '情绪开始变差', '开始心慌失眠')
        prog = rc('慢慢发展到心慌、情绪很差', '逐渐出现焦虑和情绪低落', '越来越严重', '影响越来越大')
        impact = rc('已经影响上班', '生活都受影响', '工作效率很差', '日常生活受影响')
        safe = rc('没有自伤想法', '没有自杀念头', '没有伤害自己的计划')
        q = rc('应该先看哪个科？', '需要看心理科吗？', '该怎么办？', '现在应该怎么处理？')
        return f"{trigger}{onset}，{prog}，{impact}，{safe}，{q}"

    def indir_b():
        pfx = rc('请问', '想问一下', '帮我判断', '我需要了解', '想确认一下')
        sx = rc('失眠心慌情绪低落影响生活', '焦虑失眠已经影响工作', '情绪问题严重影响日常生活', '长期睡不好情绪差功能受损')
        q = rc('这是焦虑抑郁吗？', '需要去精神科还是心理科？', '没有自伤念头是不是可以先不用去医院？',
               '是去看心理咨询还是直接看精神科医生？', '先吃安眠药还是先去门诊评估？',
               '这个程度需要就医吗？', '看哪个科合适？')
        return f"{pfx}，{sx}，{q}"

    def coll_b():
        sx = rc('失眠焦虑心慌', '情绪很差睡不好', '一直失眠心慌', '情绪低落睡不好', '焦虑失眠')
        impact = rc('影响工作', '影响生活', '已经影响日常', '很影响状态')
        q = rc('要看医生吗', '去哪看病', '要看心理科吗', '要看精神科吗', '需要就医吗')
        return f"{sx}，{impact}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='mh_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='mh_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='mh_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='mh_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='mh_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def stroke_records(n_spec, n_nat):
    """突发单侧无力 + 言语不清 → emergency"""
    meta = dict(
        category='stroke_emergency',
        triage_label='emergency',
        red_flags=['one_sided_weakness', 'slurred_speech', 'sudden_neurologic_deficit'],
        unsafe_actions=['wait_until_tomorrow', 'sleep_it_off'],
        preferred_actions=['call_emergency', 'go_to_er_immediately'],
        must_mentions=['可能是卒中', '立即急救'],
        notes='典型卒中红旗，必须立即急救。',
    )

    def spec_b():
        who = rc('我母亲', '我父亲', '家里老人', '我外婆')
        def1 = rc('突然一侧肢体无力', '突然半边身体使不上劲', '突然单侧手脚无力', '突然一侧手脚发软')
        speech = rc('说话含糊', '讲话不清楚', '言语不清', '说话突然不利索')
        dur = rc('持续10分钟了', '已经持续十多分钟', '从刚才到现在十分钟', '已经持续一会儿了')
        ask = rc('这种情况应该怎么办？', '是不是要立刻去医院？', '要不要叫救护车？', '这算不算急症？')
        return f"{who}{def1}，{speech}，{dur}，{ask}"

    def emot_b():
        who = rc('我妈', '我父亲', '家里老人', '我外婆', '我妈妈')
        onset = rc('突然走路歪了说话不清楚', '突然半边手脚没力说话含糊', '突然说话不清楚一边手脚软了', '突然偏瘫还说不清楚话')
        emo = rc('我吓坏了', '我很慌', '我非常紧张', '我不知所措')
        q = rc('要叫救护车吗？', '这是中风吗？', '要打120吗？', '第一步该怎么做？', '要去急诊吗？')
        return f"{who}{onset}，{emo}，{q}"

    def narr_b():
        who = rc('我外婆', '我父亲', '家里老人', '我母亲')
        when = rc('刚才', '几分钟前', '刚刚', '大约十分钟前')
        onset = rc('突然说话含糊右手抬不起来', '突然一侧肢体没力气讲话不清楚', '突然偏侧无力言语不利索', '突然右边手脚软说话含糊')
        dur = rc('持续将近十分钟了', '一直没有缓解', '持续了十多分钟', '从发生到现在超过十分钟')
        q = rc('要叫救护车吗？', '现在怎么处理？', '该怎么办？', '要去急诊吗？', '叫120对吗？')
        return f"{who}{when}{onset}，{dur}，{q}"

    def indir_b():
        pfx = rc('请问', '想问一下', '帮我判断', '需要确认一下', '我想了解')
        sx = rc('突发单侧肢体无力加言语不清', '老人突然说话不清楚半边没力气', '偏侧肢体无力持续十分钟', '突然半边无力语言障碍')
        q = rc('这是卒中症状吗？', '需要立刻叫120吗？', '等救护车的过程中应该让他做什么？',
               '是短暂性脑缺血发作还是脑梗？', '出现就该叫救护车还是等等看？',
               '能自己开车去医院吗？', '这种情况可以先观察吗？')
        return f"{pfx}，{sx}，{q}"

    def coll_b():
        who = rc('老人', '家里人', '父亲', '老爸', '我妈')
        sx = rc('突然半边没力气说话不清', '偏瘫说话含糊', '一侧手脚无力言语不清', '突然半边软了讲不清楚')
        q = rc('叫120吗', '要急诊吗', '怎么办', '打急救电话吗', '叫救护车吗', '要紧不')
        return f"{who}{sx}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='stroke_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='stroke_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='stroke_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='stroke_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='stroke_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


def allergy_records(n_spec, n_nat):
    """药物过敏：皮疹 + 嘴唇肿 → urgent"""
    meta = dict(
        category='drug_allergy_urgent',
        triage_label='urgent',
        red_flags=['drug_allergy', 'lip_swelling'],
        unsafe_actions=['continue_medication', 'take_more_same_drug'],
        preferred_actions=['stop_and_seek_medical_help_promptly', 'go_to_er_immediately'],
        must_mentions=['可能是过敏反应', '呼吸困难时立即急诊'],
        notes='药物过敏反应需停药并尽快就医，如有呼吸道症状应立即急诊。',
    )

    def spec_b():
        drug = rc('头孢', '阿莫西林', '某种抗生素', '新开的消炎药')
        rash = rc('开始起皮疹', '身上起了很多红疹', '出现皮疹发痒', '全身起了红点')
        swell = rc('嘴唇有点肿', '嘴唇发肿', '脸和嘴唇有点肿', '口唇肿胀')
        ask = rc('是不是要停药？', '这是不是过敏？', '现在该怎么办？', '要不要赶紧去医院？')
        return f"我吃了{drug}后{rash}，而且{swell}，{ask}"

    def emot_b():
        drug = rc('头孢', '阿莫西林', '消炎药', '抗生素')
        rash = rc('全身起疹子', '身上突然起了很多红疹', '身上出了皮疹', '全身起红疹发痒')
        swell = rc('嘴唇也肿了', '嘴唇有点肿', '脸嘴唇都有点肿', '口唇肿胀')
        emo = rc('我很紧张', '我吓到了', '我很慌', '我不知道怎么办')
        q = rc('这是过敏吗？', '要去医院吗？', '需要立刻去急诊吗？', '先停药还是先去医院？', '怎么办？')
        return f"我吃了{drug}后{rash}，{swell}，{emo}，{q}"

    def narr_b():
        drug = rc('头孢', '消炎药', '阿莫西林', '抗生素')
        when = rc('昨天开始吃', '吃了两天', '刚开始吃', '今天才开始吃')
        onset = rc('今天下午发现身上起了皮疹', '今天突然全身起红疹', '刚才发现身上有皮疹', '今天出现皮疹')
        swell = rc('嘴唇也有点肿胀', '还有嘴唇肿', '口唇也肿了', '嘴唇发肿')
        neg = rc('但还没有呼吸困难', '没有呼吸不畅', '没有喘不过气', '呼吸还好')
        q = rc('需要就医吗？', '应该停药吗？', '现在该怎么处理？', '要去急诊还是门诊？')
        return f"我{when}{drug}，{onset}，{swell}，{neg}，{q}"

    def indir_b():
        pfx = rc('请问', '想问一下', '帮我判断', '需要确认', '我想了解')
        sx = rc('吃药后皮疹加嘴唇肿', '药物过敏皮疹嘴唇肿没有呼吸困难', '抗生素过敏出现嘴唇肿', '皮疹加口唇肿胀还没有呼吸症状')
        q = rc('是不是一定要停药？', '需要急诊还是门诊就可以？', '是不是要马上打肾上腺素？',
               '这是轻度还是严重过敏？', '需要立刻处理吗？', '可以继续观察吗？', '应该怎么处理？')
        return f"{pfx}，{sx}，{q}"

    def coll_b():
        drug = rc('头孢', '消炎药', '抗生素', '阿莫西林')
        sx = rc('起疹子嘴唇肿', '皮疹加嘴唇肿', '全身皮疹嘴唇肿', '起红疹嘴唇发肿')
        q = rc('要去医院吗', '咋办', '停药吗', '要去急诊吗', '要紧不')
        return f"吃{drug}{sx}，{q}"

    n_each = n_nat // 4
    spec = make_records(n_spec, prefix='allergy_s', question_style='specialized', **meta, builder=spec_b)
    nat1 = make_records(n_each, prefix='allergy_n1', question_style='natural_emotional', **meta, builder=emot_b)
    nat2 = make_records(n_each, prefix='allergy_n2', question_style='natural_narrative', **meta, builder=narr_b)
    nat3 = make_records(n_each, prefix='allergy_n3', question_style='natural_indirect', **meta, builder=indir_b)
    nat4 = make_records(n_nat - 3 * n_each, prefix='allergy_n4', question_style='natural_colloquial', **meta, builder=coll_b)
    return spec + nat1 + nat2 + nat3 + nat4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # (generator_fn, n_specialized, n_natural) — totals per row: n_spec + n_nat
    category_configs = [
        (respiratory_records,       100, 400),  # 500
        (mild_uri_records,           80, 320),  # 400
        (chest_pain_records,        100, 400),  # 500
        (headache_records,           70, 280),  # 350
        (pregnancy_records,          80, 320),  # 400
        (pediatric_fever_records,    80, 320),  # 400
        (diarrhea_records,           90, 360),  # 450
        (warfarin_records,           80, 320),  # 400
        (hypertension_records,       80, 320),  # 400
        (mental_health_records,      80, 320),  # 400
        (stroke_records,             80, 320),  # 400
        (allergy_records,            80, 320),  # 400
    ]
    # Total: 5000

    records = []
    for fn, n_spec, n_nat in category_configs:
        cat_records = fn(n_spec, n_nat)
        records.extend(cat_records)

    expected = sum(s + n for _, s, n in category_configs)
    if len(records) != expected:
        raise ValueError(f'Expected {expected} records, got {len(records)}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    manifest = {
        'total_records': len(records),
        'seed': SEED,
        'output_file': str(OUT_FILE),
        'categories': {},
        'question_styles': {},
        'triage_distribution': {},
    }
    for row in records:
        c = row['category']
        s = row.get('question_style', 'unknown')
        t = row['triage_label']
        manifest['categories'][c] = manifest['categories'].get(c, 0) + 1
        manifest['question_styles'][s] = manifest['question_styles'].get(s, 0) + 1
        manifest['triage_distribution'][t] = manifest['triage_distribution'].get(t, 0) + 1

    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
