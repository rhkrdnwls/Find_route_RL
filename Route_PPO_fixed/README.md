# PPO-based Grid Navigation

PPO(Proximal Policy Optimization)로 2D 그리드 환경에서 로봇이 장애물을 회피하며 시작점에서 목적지까지 경로를 탐색하도록 학습하는 프로젝트입니다.

## Environment

| 항목 | 값 |
|---|---|
| Grid Size | 10 × 10 |
| Start | (0, 0) |
| Goal | (9, 9) |
| Action Space | Up / Down / Left / Right |
| Max Steps | 100 |
| Obstacles | Random placement (BFS로 도달 불가능한 맵은 제외) |

## Observation Space

3개 채널로 분리된 (3, 10, 10) 텐서를 사용합니다. (초기엔 단일 그리드에 값을 0/1/2/3으로 구분해서 표현했으나, 채널 분리 방식으로 변경)

- Channel 0: Obstacle
- Channel 1: Robot
- Channel 2: Goal

MLP 입력 시 이를 Flatten하여 300차원 벡터로 사용합니다.

## Reward Design

```python
goal_change = old_goal_distance - new_goal_distance
reward += goal_weight * goal_change

obstacle_change = new_obs_distance - old_obs_distance
reward += obstacle_weight * obstacle_change
```

| 파라미터 | 값 |
|---|---|
| goal_weight | 0.05 |
| obstacle_weight | 0.005 |
| step_penalty | 0.01 |
| invalid_move_penalty | -0.3 |
| Goal Reward | +7.0 |
| Collision Reward | -4.0 |

## Model: Actor-Critic (MLP)

```
Observation (3×10×10) → Flatten → Linear(300→128) → ReLU
                       → Linear(128→128) → ReLU
                       ├─ Actor  → 4 Actions
                       └─ Critic → V(s)
```

10×10 크기의 작은 환경에서는 CNN을 실험해봐도 MLP 대비 뚜렷한 성능 향상이 없어, 최종적으로 MLP 구조를 유지했습니다.

## Training

Rollout 수집 → Shuffle → Mini-batch 분할 → PPO Update → 여러 Epoch 반복

| 파라미터 | 값 |
|---|---|
| Rollout Steps | 2048 |
| Mini-batch Size | 256 |
| Update Epochs | 4 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Epsilon | 0.2 |

GAE(Generalized Advantage Estimation)로 Advantage를 안정적으로 계산합니다.

```python
delta = rewards[t] + gamma * next_value * not_done - values[t].item()
gae = delta + gamma * gae_lambda * not_done * gae
```

### Validation & Model Saving

- 고정된 Validation Map(300개)으로 학습 중 일반화 성능 평가 (Eval Interval: 10,000 episodes)
- Validation Success Rate가 최고 기록을 갱신할 때마다 모델 저장 (`ppo_route_model.pth`)
- Early Stopping: `patience=10`, `min_delta=0.005`, `early_stop_start=300,000`

## Evaluation

경로 효율은 BFS 최단거리 대비 PPO가 실제로 이동한 경로 길이로 계산합니다.

```
Path Efficiency = BFS Shortest Distance / PPO Path Length
```

대표 Test 결과 (100 episodes):

| 지표 | 값 |
|---|---|
| Success | 83 |
| Success Rate | 83% |
| Average Steps | 18 |
| Path Efficiency | 100% |

초기 학습에서는 Timeout이 빈번했으며(예: Invalid Move 91회, BFS Distance 18 → Timeout), 학습이 진행되며 개선되었습니다.

## Usage

```bash
python Route_fixed_RL.py    # Linux: python3 Route_fixed_RL.py
```

모델 로드:

```python
model = ActorCritic()
model.load_state_dict(torch.load("ppo_route_model.pth", map_location="cpu"))
model.eval()
```

## Git Workflow

```bash
git pull origin main
# 작업 후
git add .
git commit -m "update project"
git push origin main
```

## Future Work

2D Grid PPO → Gazebo Simulation → LiDAR/Odometry/Goal Observation → Continuous Action PPO → ROS2 Integration → Sim-to-Real

## Tech Stack

Python, PyTorch, PPO, Actor-Critic, BFS, Pygame, ROS2 (예정), Gazebo (예정)