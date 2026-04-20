# v3 Model Family Support — 新モデル追加ガイド

v3 エンジンは元々 Qwen3.5 hybrid (24 GDN + 8 FA) を前提に実装されて
いますが、2026-04-19 の汎用化層により **model family 自動判別 + 設定化**
で他モデル (Gemma 4、Qwen 3.6、Llama 3、Mistral 等) も動作します。

本書は新モデル追加の手順をまとめます。

---

## 1. 対応状態 (current)

| Family | Status | mlx_lm module | Hybrid | Thinking | Notes |
|:---|:---:|:---|:---:|:---:|:---|
| **qwen35** | ✅ Native | `mlx_lm.models.qwen3_5` | Yes | Yes | v3 開発モデル |
| qwen36 | Scaffolding | `mlx_lm.models.qwen3_6` (想定) | Yes | Yes | 要実装時確認 |
| qwen25 | Scaffolding | `mlx_lm.models.qwen2` | No | No | pure transformer |
| llama | Scaffolding | `mlx_lm.models.llama` | No | No | 3/3.1/3.2/3.3 |
| mistral | Scaffolding | `mlx_lm.models.mistral` | No | No | Mistral 系 |
| gemma | Scaffolding | `mlx_lm.models.gemma3` | No | No | Gemma 2/3/4 |

Scaffolding = コードパスは整っているが、実モデルでの検証は未実施。

---

## 2. 最小の設定例

### 2.1 Gemma 4 を main model にする場合 (想定)

`configs/engine.toml`:
```toml
[model]
family = "auto"                                              # または "gemma" 明示
main_path = "mlx-community/gemma-4-9b-it-4bit"               # HF id or local path
draft_path = "mlx-community/gemma-4-2b-it-4bit"              # RDMS 用 (optional)
draft_family_mismatch = "warn"
```

起動ログで:
```
model_family: detected gemma via path heuristic (gemma-4)
```

### 2.2 Qwen 2.5 に切り替え (pure transformer、speculative FAST PATH)

```toml
[model]
family = "qwen25"
main_path = "mlx-community/Qwen2.5-14B-Instruct-4bit"
draft_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

RDMS の speculative_generate が pure transformer fast path (offset rollback)
で動作、hybrid の save/restore overhead が不要なので +1.5-2.5x の期待。

---

## 3. Innovation の family 互換性

| Innovation | Qwen3.5 | Pure Transformer (Llama/Gemma/Qwen2.5) | Notes |
|:---|:---:|:---:|:---|
| N1 CTRSP | ✅ | ⚠️ | GDN state save 依存、Llama 系では不要/無効 |
| X4 Context Fold | ✅ | ✅ | generic |
| X2 DPC | ✅ | ✅ | precision swap、generic |
| N4 ALS | ✅ | ✅ | layer skip、generic |
| N3 SSEE | ✅ | ✅ | self-speculative、generic |
| X5R Compiled | ✅ | ✅ | mx.compile、generic |
| N6 PES | ✅ | ✅ | parallel expert、generic |
| N5 ERP | ✅ | ✅ | entropy routing、generic |
| RDMS | Hybrid SLOW | **FAST PATH** | pure で 1.5-2.5x、hybrid で 0.5-0.8x |
| Think-Aware Budget | ✅ | N/A | thinking 非対応モデルでは自動 skip |
| Loop-Escape Injection | ✅ | N/A | 同上 |

---

## 4. 新 family を追加する開発手順

### Step 1. `app/core/model_family.py` に enum + profile 追加

```python
class ModelFamily(str, Enum):
    ...
    CLAUDE = "claude"  # 仮の例

FAMILY_PROFILES[ModelFamily.CLAUDE] = FamilyProfile(
    family=ModelFamily.CLAUDE,
    is_hybrid=False,
    has_gdn=False,
    supports_thinking=False,
    mask_module="mlx_lm.models.claude",
    notes="Example",
)
```

### Step 2. `app/backend/mask_adapter.py` に分岐追加

```python
if family == ModelFamily.CLAUDE:
    return _pure_transformer_masks(mod, inner_model, hidden_states, cache)
```

### Step 3. `_PATH_HEURISTICS` または `_ARCH_TO_FAMILY` で自動判別対応

```python
_PATH_HEURISTICS.append(("claude", ModelFamily.CLAUDE))
```

### Step 4. config に family 明示 or auto-detect で動作

### Step 5. 必要なら `tests/test_core/test_model_family.py` に case 追加

---

## 5. Hybrid 特化 innovation の自動無効化

Pure transformer family が選択されたとき、以下は**自動で no-op** となります:

- GDN state save/restore (CTRSP): layer に `is_linear=True` が存在しないので skip
- Think-aware budget: `supports_thinking=False` で think_open/close = None
- SSM mask: `mask_adapter` が None を返し、forward で FA mask のみ使用

結果、コード変更なしで innovation が family 適合する挙動になります。

---

## 6. Draft / Main family mismatch

RDMS で draft と main が異なる family の場合の挙動は
`[model] draft_family_mismatch` で制御:

- `warn` (default): warning log を出し、mismatch のまま動作。速度・品質は保証外
- `refuse`: draft を不活化、main 単独 decode にフォールバック
- `ignore`: log も出さず進む (非推奨)

---

## 7. 推奨構成例

### 7.1 軽量運用 (Gemma 4 + Gemma 4 draft)

```toml
[model]
family = "gemma"
main_path = "mlx-community/gemma-4-9b-it-4bit"
draft_path = "mlx-community/gemma-4-2b-it-4bit"
```

### 7.2 Agent workload (Qwen 3.5 + 0.8B RDMS)

```toml
[model]
family = "qwen35"
main_path = "mlx-community/Qwen3.5-9B-MLX-4bit"
draft_path = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
```

### 7.3 Pure transformer で最大速度 (Qwen 2.5)

```toml
[model]
family = "qwen25"
main_path = "mlx-community/Qwen2.5-14B-Instruct-4bit"
draft_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
# pure transformer: RDMS FAST PATH で 1.5-2.5x 期待、hybrid overhead なし
```

---

## 8. 既知の制約

- mlx_lm の最新版に対応モデル module が存在することが前提。古い mlx_lm では動かない可能性あり
- 一部 innovation (N2 GGSA 等) は Qwen3.5 depth 32 に特化した数値が残っているので、
  異なる layer 数のモデルでは acceptance が落ちる可能性
- Gemma 4 の mlx_lm 対応状況は随時確認のこと。`mask_module` の module path を調整

---

## 9. 検証手順 (新モデル追加後)

1. `pytest tests/test_core/test_model_family.py -v` で unit tests pass
2. `pytest tests/ -q` で全体 regression なし
3. Server 起動 + `/health` で family が正しく検出されているか確認
4. `app/bench/full_suite_bench.py` で decode / needle / warm TTFT 計測
5. Opus rubric bench で品質測定 (別途 API 経由)

---

v3 generalization layer は Qwen3.5 挙動を完全に保持する設計です。
新 family 追加は **ゼロ回帰** を基本方針としてください。
