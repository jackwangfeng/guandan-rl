# v8 设计:真规则掼蛋

> **Status:** 实现完成,已训练 600k rounds(2026-05-18)。终评:vs rule_agent 多局 100%,单局 vs random 88.8% / vs rule 90.2%。
> 实施时**接受简化**:同花顺不接受 wildcards;抗贡未建模;固定 SEQ5/PSEQ3/TSEQ2 长度。详见下方"风险"段。

不和 v6/v7 共用 `env.py`,新写 `env_v8.py` + `features_v8.py` + `model_v8.py`(可同架构但维度不同)+ `train_v8.py`。

## 核心改动:加 suit 维度

```
DECK shape: (NUM_TYPES=15, NUM_SUITS=4)
  非王牌 (rank 0..12): 每个 (rank, suit) 有 2 张(2 副)→ 13×4×2 = 104
  小王 (SJ=13): 2 张(suit 编 0)
  大王 (BJ=14): 2 张(suit 编 0)
  合计:108 张

SUIT 编码: SPADE=0, HEART=1, DIAMOND=2, CLUB=3
WILDCARDS: rank == level_rank AND suit == HEART
  每副牌正好 1 张红心级牌 → 一局正好 2 张 wildcards(同 v6,只是现在 suit-specific)
```

## 新增牌型

**同花顺**(`FLUSH_SEQ5`):5 张连续点数 + 同花色
- 起始 rank: 0..7(即 '3' 到 'T'),不可包含 '2' 或 王
- 同花色约束:5 张必须同一 suit(其中 suit==HEART 可用 wildcards 替代)
- **不接受非红心 wildcards**;红心同花顺可用 wildcards(因 wildcard 本身就是红心)
- 简化方案:**v8 第一版不允许 wildcards 进入同花顺**,规则清爽

**同花顺**视作炸弹等级,可以跨牌型压制(像普通炸一样)。

## 炸弹等级(从高到低)

1. 王炸(4 王)
2. **同花顺**(任意起点)
3. 6+ 张同点炸弹(by size,then by rank)
4. 5 张同点炸弹
5. 4 张同点炸弹

同级别比较:
- 同点炸:大 size 压小 size;同 size 压高 rank
- 同花顺 vs 同花顺:高起始 rank 压低

注意:6+ 张同点炸 **压过** 同花顺(质量胜数量);同花顺 **压过** 5 张同点炸。

## Move 元组扩展

```
(combo, rank, pair_rank, count, n_wild, suit)
```

新增 `suit` 字段:
- 非同花顺组合:`suit=0`(无意义)
- `FLUSH_SEQ5`:`suit ∈ {0,1,2,3}`,记录所用花色

## 手牌表示

```
hand[player]: (15, 4) int8 — 每个 (rank, suit) 的张数
wildcards[player]: int — 当前手中红心级牌数
```

为方便计算,提供 `rank_counts(hand) = hand.sum(axis=1)` 派生量。

## 状态特征

```
own_hand_rank_total: (15,)        # 求和 over suits
own_hand_per_suit:   (15, 4)      # 完整 per-suit
last_play_counts:    (15,)
last_play_combo:     onehot(NUM_COMBO_TYPES)  # 多一个 FLUSH_SEQ5
hand_sizes:          (3,)         # teammate + 2 opps
played_per_seat:     (15,) × 4    # 总数,不按 suit
level_rank_onehot:   (15,)
own_wildcards:       (1,)
wildcards_left:      (1,)
played_wildcards:    (4,)
last_play_suit:      onehot(4)    # 仅同花顺时有意义

总计: 15 + 60 + 15 + 10 + 3 + 60 + 15 + 1 + 1 + 4 + 4 = 188
```

## 动作特征

```
combo_onehot:        (NUM_COMBO_TYPES=10)
rank_onehot:         (16, +1 joker bomb)
pair_rank_onehot:    (16)
count:               (1, /8)
n_wild_onehot:       (3, values 0/1/2)
suit_onehot:         (4)

总计: 10 + 16 + 16 + 1 + 3 + 4 = 50
```

## 信念头

仍是 3 × 15 = 45 维(预测对手 rank 总数,不预测 suit 分布 —— 简化)。
后续可以加 suit-aware 信念,但收益边际,先不做。

## 训练计划

- 重训 v8 from scratch(STATE_DIM 变了)
- 配方同 v7:hidden=1536, layers=6, belief=45
- 训练目标 600k rounds,多局训练
- v7 当 anchor(`--anchor-ckpt runs/v7/latest.pt --anchor-p 0.20`)
- 预计 ~4-5 小时(动作空间变大,合法动作更多)

## 评估

- 单局 vs random/rule(对照)
- 多局 vs random/rule
- 直接对打:**v8 vs v7**, v8 vs v6 — 看真规则下能否压制简化规则模型

## 实现顺序

1. `env_v8.py`:DECK suit 化、deal、legal_moves(同点 + 同花顺 + 炸弹)、step、obs
2. `features_v8.py`:新状态/动作编码
3. `match_v8.py`:更新进贡/还贡逻辑(忽略 suit,只看 rank)
4. `rule_agent_v8.py`:升级,会用同花顺凑炸,会用 wildcards 补同花顺
5. `train_v8.py`:训练脚本,带 anchor 选项
6. `test_smoke_v8.py`:发牌守恒、wildcard 限定红心、同花顺枚举、炸弹等级

## 风险 / 未解决

- 同花顺的 wildcards 处理:第一版不允许,简洁;后续可以加(在红心-级牌存在时允许"用 1 张 wildcard 凑红心同花顺")
- 变长顺子(SEQ6, SEQ7 etc.):v8 不实现;掼蛋多数变体只允许 5 张顺子,这是主流
- 抗贡(big joker 拒贡):v8 不实现;影响小
- 炸弹等级里同花顺 vs 6 张炸的优先级:不同地区版本差异;选了"6+ 张炸压同花顺"这一版本
