# OpenAI gym CartPole-v1 Environment / Unity ML-Agents

Usage:
/ml-agents/mlagents/trainers/learn_gym.py ../../../config/trainer_config.yaml --run-id=firstRun  --train --no-graphics --env-args=opt

interface:
ml-agent-gym/environment.py

# ML-AgentsでCartPoleを学習する

gym学習環境はこちらで設定できます。
ml-agent-gym/environment.py
configファイルの設定によってSAC学習、PPO学習の２つで学習できます。

ブレインは１つに限定したインターフェースとなっています。
