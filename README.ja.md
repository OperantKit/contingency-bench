# contingency-bench

:gb: [English README](README.md)

OperantKit 実験キットのためのベンチマーク・ランナー: タイミング精度、
LLM-as-subject 性能、マッチング法則パラメータ回復。

## 戦略 A: Python 専用移植

本パッケージはオリジナル Rust + Python 混成ベンチマーク・プロジェクトの
Python 専用移植である。戦略 A（移植計画に基づく）は LLM 行動特性
ランナーをここに残し、Rust hardware-in-the-loop (HIL) host_runner /
Teensy ファームウェアは別パッケージに延期する。そのハンドオフが
届くまで、Rust ワークスペースと Teensy ground-truth ファームウェアは
凍結 v0 アーカイブ内に残る（読み取り専用参照、本配布の一部ではない）。

## 公開 API

```python
from contingency_bench import (
    BaseLLMClient,
    APILLMClient,
    MockLLMClient,
    ConcurrentVISession,
    PolicyCompressionAgent,
    MatchingParameters,
    estimate_matching_parameters,
    estimate_beta_from_sensitivity,
    equate_training_steps,
    run_benchmark,
    run_condition,
)
```

CLI:

```bash
contingency-bench --mock --steps 50 --verbose
contingency-bench --mock --model claude-3-5-sonnet --model gpt-4o --steps 100
```

## インストール

```bash
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

ライブ API ベンチマーク用（Anthropic SDK、遅延読み込み）:

```bash
.venv/bin/python -m pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...
```

## テスト

```bash
.venv/bin/python -m pytest -q
```

## ローカル依存関係に関する注記

Q-learning エージェントと GML 推定器は、同じ monorepo の姉妹パッケージに
由来するが、まだ PyPI に公開されていない。`contingency-bench` を
スタンドアロン・パッケージとしてインストール可能に保つため、両モジュールは
ここに **インライン化** されている:

- `contingency_bench.policy_compression.PolicyCompressionAgent`
- `contingency_bench.matching.estimate_matching_parameters`
  （`session_analyzer.analytics.matching` のミラー）

upstream モジュールが公開されたら、インライン・コピーは再エクスポートに
置き換えるべき。公開シンボルは同じまま。

## LLM 依存関係に関する注記

v0 ドライバは `litellm` を介して複数モデル・ベンダをサポートしていた。本
移植は新リポジトリ標準の LLM 依存関係に合わせ、単一ベンダの `anthropic`
SDK を採用している。SDK は `contingency_bench.llm_client.APILLMClient.__init__`
で **遅延** インポートされる; モジュール自体のインポートで optional な
`[llm]` extra が必要になることはない。複数ベンダ・サポートの再導入は
フォローアップ。

## 将来の Rust ワークスペース

Rust HIL ハーネス (`host_runner` バイナリ、`protocol`、`scenarios`、
`metrics`) と Teensy 4.1 ground-truth ファームウェアは v0 アーカイブに
存在し、`contingency-rs` ハンドオフと並行して別パッケージに移植される
予定。これらは本 Python 専用リリースでは意図的にスコープ外。

## 参考文献

- Baum, W. M. (1974). On two types of deviation from the matching law: Bias
  and undermatching. *Journal of the Experimental Analysis of Behavior*, 22,
  231-242. https://doi.org/10.1901/jeab.1974.22-231
- Davison, M. (2021). What is the value of the generalized matching law?
  *Journal of the Experimental Analysis of Behavior*, 116, 114-123.
  https://doi.org/10.1002/jeab.694
- Johnson, J. E., Kangas, B. D., Bergman, J., & Rice, K. C. (2022).
  Comparison of fixed- and variable-ratio schedules of reinforcement on
  progressive-ratio responding. *Journal of the Experimental Analysis of
  Behavior*, 117, 320-330. https://doi.org/10.1002/jeab.755
- Lai, L., & Gershman, S. J. (2021). Policy compression: An information
  bottleneck in action selection. *Psychology of Learning and Motivation*, 74,
  195-232. https://doi.org/10.1016/bs.plm.2021.02.004
