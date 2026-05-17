# Mini-Guandan RL

Self-play DMC for simplified Guandan, evolving through v4 → v6 → v7 → v8 as we
add rule fidelity and architectural improvements.

## Versions

| Version | Env | Model | Training scope | Status |
|---|---|---|---|---|
| v4 | 简化规则,无级牌/wildcard | MLP 4×1024 | 200k 单局 | done (winrates ~85% / 95%) |
| v5 | 同 v4 | 同 v4 | 200k 续训 + 多对手 / 搜索注入实验 | done; **搜索注入证伪**,纯续训最优 |
| v6 | 加级牌 + wildcard(2 张通用) | 同 v4 | 300k 单局 | done; **多局 vs rule_agent 100%** |
| v7 | 同 v6 + Match 驱动多局 | MLP 6×1536 + **信念头** | 600k rounds,v6 当 anchor | training |
| v8 | **真规则**:suit 维度 + 红心-only wildcard + **同花顺** | 同 v7 | 600k rounds,v7 当 anchor | code ready |

## Combos modelled

```
v4-v7:  pass / single / pair / triple / 3+2 / bomb / SEQ5 / PSEQ3 / TSEQ2
v8 也加:  FLUSH_SEQ5 (同花顺,作为 bomb-tier 牌型)
```

Bomb hierarchy in v8 (high→low):
1. 王炸 (4 jokers)
2. 6+张同点炸
3. 同花顺
4. 5张同点炸
5. 4张同点炸

## What's modelled

- 4 players, 2v2 teams (P0+P2 vs P1+P3), 108 cards
- 级牌 / wildcards (v6+):红心级牌作万能牌,可替换非王牌
- 多局 Match:进贡 / 还贡(末位向头名),升级(双下+3/单下+2/平+1)
- v7+:信念头 — 网络辅助预测对手手牌,被 MSE-loss 监督

## What's NOT modelled

- 变长顺子(只支持 SEQ5 / PSEQ3 / TSEQ2 固定长度)
- 抗贡(big-joker 拒贡)
- 同花顺含 wildcards(可后续加)
- 多局 reward shaping(目前 per-round ±1)
- 队友信号通讯(掼蛋的最关键策略层,需要 RNN/Transformer)

## Files

```
env.py            v6/v7 env (级牌 + 通用 wildcard)
env_v8.py         v8 env (suit-aware + 真 wildcard + 同花顺)
features.py       v6/v7 编码
features_v8.py    v8 编码
match.py          v6/v7 多局
match_v8.py       v8 多局
rule_agent.py     v6/v7 启发式 baseline
rule_agent_v8.py  v8 启发式 baseline
model.py          QNet(state_dim/action_dim/hidden/num_layers/belief_dim 参数化)
agent.py          replay buffer (含可选 belief 维度) + pick_action
vec_collect.py    v6 vec collector
vec_collect_v7.py v7 collector (Match 驱动 + belief 采集)
vec_collect_v8.py v8 collector
train_v3.py       v4/v5 训练脚本(简化 env)
train_v5.py       v5 实验(随机对手 + 搜索注入)
train_v6.py       v6 训练
train_v7.py       v7 训练(可选 anchor)
train_v8.py       v8 训练(可选 anchor)
eval.py           载入 ckpt 单局 eval
eval_match.py     多局 match eval
eval_search.py    v4-PIMC 推理时搜索(实验,证明搜索注入无效)
compare.py        多模型 vs random/rule + 互怼
search.py         PIMC search 实现(实验)
serve.py          v4 时代 HTTP 服务(老协议)
test_smoke.py     v6/v7 env 烟雾
test_smoke_v8.py  v8 env 烟雾
V8_DESIGN.md      v8 设计文档
INTEGRATION.md    v4 时代外部协议(过时,待更新)
```

## Run

```bash
# Smoke
python test_smoke.py       # v6/v7 env
python test_smoke_v8.py    # v8 env

# Train
python train_v6.py --out runs/v6 --total-episodes 300000
python train_v7.py --out runs/v7 --total-rounds 600000 --anchor-ckpt runs/v6/latest.pt
python train_v8.py --out runs/v8 --total-rounds 600000 --anchor-ckpt runs/v7/latest.pt

# Eval
python compare.py --ckpts runs/v6/latest.pt runs/v7/latest.pt --names v6 v7 --games 400
python eval_match.py --ckpt runs/v6/latest.pt --vs rule --matches 50
```

## Honest expectations

- 跟 rule_agent / random 这种弱基线打:v6+ 已饱和(多局 100%)
- 跟会算牌、会信号通讯的真人:**v8 大概率打不过**;
  - DouZero (斗地主 SOTA) 用了 ~5B episodes,我们 v8 才 600k 量级
  - 4 人 2v2 通讯博弈(桥牌 / 掼蛋)是 imperfect-info RL 里最难一类
- 这个项目可以拿来:learn RL pipeline、当对比基线、产生牌局数据,**不是要超越商业掼蛋 AI**

## Hardware

```
RTX A4000 (16GB) 单卡,Linux,Python 3.10+,torch 2.8+cu128。
单次训练 1-4 小时;v8 全套约 4-5 小时。
```
