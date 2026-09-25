import random
import pygame
from collections import deque
import numpy as np

SEED = 42

import torch
import torch.nn as nn
import torch.optim as optim

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from torch.distributions import Categorical

MODEL_PATH = "ppo_route_model.pth"

class RouteFind:
    ACTIONS = {
        0: (-1, 0),  # 위
        1: (1, 0),   # 아래
        2: (0, -1),  # 왼쪽
        3: (0, 1),   # 오른쪽
    }

    def __init__(self, grid_size=10, n_obstacles=10, max_steps=100, seed=None):
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.robot_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.step_count = 0
        self.invalid_count = 0

    def reset(self):
        while True: # 맵은 계속 생성
            occupied = set()

            # 로봇 배치
            self.robot_pos = (0,0)
            occupied.add(self.robot_pos)

            # 목적지 배치
            self.goal_pos = (9,9)
            occupied.add(self.goal_pos)

            # 장애물 배치
            self.obstacles = []
            for _ in range(self.n_obstacles):   
                pos = self._random_empty_cell(occupied)
                occupied.add(pos)
                self.obstacles.append(pos)

            shortest_distance = self.get_shortest_distance() # 시작에서 끝까지 갈 수 있는지 확인

            if shortest_distance is None:
                # 갈 수 없다면 스킵하고 다시 생성
                continue

            break; # 아니라면 맵 사용

        self.step_count = 0
        self.invalid_count = 0

        return self._get_obs()
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    # 맨해튼 거리 코드

    def nearest_obstacle_distance(self, pos):
        return min(
            self.manhattan_distance(pos, obs)
            for obs in self.obstacles
        )


    def step(self, action):
        goal_weight = 0.05 # 목표에 다가갈 때 주어지는 페널티
        obstacle_weight = 0.005 # 장애물과의 거리 변화에 붙는 가중치
        step_penalty = 0.01 # Step 움직일 때마다 주는 작은 페널티
        invalid_move_penalty = 0.3 # 벽 안쪽에 있게 

        self.step_count += 1

        assert action in self.ACTIONS, f"invalid action: {action}"
        # assert는 이 조건이 반드시 참이여야 하는 검사
        # action이 self.ACTIONS 안에 있어야 한다 --> 없으면 에러 발생
        # 이동 전 목적지 거리
        old_goal_distance = self.manhattan_distance(
            self.robot_pos,
            self.goal_pos
        )

        # 이동 전 가장 가까운 장애물 거리
        old_obs_distance = self.nearest_obstacle_distance(
            self.robot_pos
        )

        dr, dc = self.ACTIONS[action]
        new_r = self.robot_pos[0] + dr # row(행) 방향으로 얼마나 움직일까
        new_c = self.robot_pos[1] + dc # col(열) 방향으로 얼마나 움직일까

        reward = -step_penalty  # 매 스텝 작은 시간 페널티
        done = False
        info = {}

        # 경계를 벗어나면 이동 취소 (제자리)
        if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
            self.robot_pos = (new_r, new_c)
        else:
            reward -= invalid_move_penalty
            self.invalid_count += 1

        new_goal_distance = self.manhattan_distance(
            self.robot_pos,
            self.goal_pos
        )

        new_obs_distance = self.nearest_obstacle_distance(
            self.robot_pos
        )
        # 목적지 방향 보상
        goal_change = old_goal_distance - new_goal_distance
        reward += goal_weight * goal_change

        # 장애물 거리 변화
        obstacle_change = new_obs_distance - old_obs_distance

        # 약한 장애물 근접 패널티 / 멀어지면 약한 보상
        reward += obstacle_weight * obstacle_change

        # 장애물 충돌
        if self.robot_pos in self.obstacles:
            reward = -4.0
            done = True
            info["result"] = "collision"

        # 목적지 도착
        elif self.robot_pos == self.goal_pos:
            reward = 7.0
            done = True
            info["result"] = "goal_reached"


        if self.step_count >= self.max_steps:
            done = True
            info.setdefault("result", "timeout")

        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        # Observation State 얻어오는 거, 여기가 상태 표현하는 코드
        grid = np.zeros(
            (3, self.grid_size, self.grid_size),
            dtype=np.float32
        )

        # 0번 채널: 장애물
        for r, c in self.obstacles:
            grid[0, r, c] = 1.0

        # 1번 채널: 로봇
        rr, rc = self.robot_pos
        grid[1, rr, rc] = 1.0

        # 2번 채널: 목적지
        gr, gc = self.goal_pos
        grid[2, gr, gc] = 1.0
        return grid
        # 하나의 grid에 1, 2, 3 이렇게 넣으면 신경망이 숫자 크기에 의미 부여할 수 있음
        # 채널을 분리해 장애물 채널, 로봇 채널, 목적지 채널 --> 이렇게 나눈다.
        # 이렇게 Python list로 나눈 것을 Numpy 배열로 바꾼다.

    def _random_empty_cell(self, occupied):
        while True:
            pos = (self.rng.randrange(self.grid_size), self.rng.randrange(self.grid_size))
            if pos not in occupied:
                return pos

    def get_shortest_distance(self):
        start = self.robot_pos
        goal = self.goal_pos
        # 시작과 보상의 좌표 가져오기

        queue = deque([(start, 0)])
        visited = {start}
        # start 좌표 넣고 시작

        while queue:
            (r, c), dist = queue.popleft() # Queue 맨 왼쪽에 있는 좌표 가져오기

            if (r, c) == goal: # 그게 목적지라면 끝
                return dist

            for dr, dc in self.ACTIONS.values():
                # Action에 있는 행동 4개 가져와서 더하기
                nr = r + dr
                nc = c + dc

                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size # 이동한게 지도 안쪽에 있고
                    and (nr, nc) not in self.obstacles
                    and (nr, nc) not in visited
                    # 방문하거나 장애물 위치가 아니여야 함  
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))

        return None
    
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, 4)
        # Actor는 상, 하, 좌, 우 이렇게 4번 움직인다.

        self.critic = nn.Linear(128, 1)
        # Critic은 상태 가치 1개를 출력한다.
    def forward(self, x):    
        x = self.shared(x)

        action_logits = self.actor(x)
        state_value = self.critic(x)

        return action_logits, state_value

class RouteRenderer:
    """간단한 pygame 렌더러. env.robot_pos / goal_pos / obstacles를 그려줌."""

    CELL_SIZE = 40

    def __init__(self, env: RouteFind):
        self.env = env
        pygame.init()
        size = env.grid_size * self.CELL_SIZE
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Grid Navigation (static obstacles)")
        self.clock = pygame.time.Clock()

    def draw(self):
        self.screen.fill((240, 240, 235))   
        cs = self.CELL_SIZE

        # 그리드 선
        for i in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * cs), (self.env.grid_size * cs, i * cs))
            pygame.draw.line(self.screen, (200, 200, 200), (i * cs, 0), (i * cs, self.env.grid_size * cs))

        # 장애물 (회색 사각형)
        for r, c in self.env.obstacles:
            rect = pygame.Rect(c * cs, r * cs, cs, cs)
            pygame.draw.rect(self.screen, (90, 90, 90), rect)

        # 목적지 (초록 원)
        gr, gc = self.env.goal_pos
        pygame.draw.circle(self.screen, (40, 180, 60), (gc * cs + cs // 2, gr * cs + cs // 2), cs // 2 - 4)

        # 로봇 (파란 원)
        rr, rc = self.env.robot_pos
        pygame.draw.circle(self.screen, (40, 90, 220), (rc * cs + cs // 2, rr * cs + cs // 2), cs // 2 - 4)

        pygame.display.flip()

    def tick(self, fps=10):
        self.clock.tick(fps)

def train_ppo(
    episodes=5000,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=2e-4,
    rollout_steps=2048,
    minibatch_size=256,
    update_epochs=4,
    entropy_coef=0.01,
    value_coef=0.5
):
    env = RouteFind()
    model = ActorCritic()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    # 성공 여부 기록
    success_history = []

    # 현재까지 종료된 episode 개수
    episode_count = 0

    # 첫 episode 시작
    obs = env.reset()

    # reset 직후의 BFS 최단거리
    shortest = env.get_shortest_distance()

    # 현재 episode 통계
    episode_reward = 0.0
    action_counts = [0, 0, 0, 0]

    # =========================================================
    # 전체 학습
    # =========================================================
    while episode_count < episodes:

        # -----------------------------------------------------
        # Rollout Buffer 초기화
        # -----------------------------------------------------
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        # =====================================================
        # rollout_steps 만큼 경험 수집
        # =====================================================
        for _ in range(rollout_steps):

            # NumPy observation → Tensor
            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32
            ).unsqueeze(0)

            # 현재 정책으로 행동 선택
            with torch.no_grad():

                logits, value = model(obs_tensor)

                dist = Categorical(
                    logits=logits
                )

                # 정책 확률분포에서 행동 sampling
                action = dist.sample()

                # 선택 행동의 log probability
                log_prob = dist.log_prob(action)

            action_item = action.item()

            # 행동 횟수 기록
            action_counts[action_item] += 1

            # 실제 환경에 행동 전달
            next_obs, reward, done, info = env.step(
                action_item
            )

            episode_reward += reward

            # -------------------------------------------------
            # Rollout 저장
            # -------------------------------------------------
            states.append(
                obs_tensor.squeeze(0)
            )

            actions.append(
                action.squeeze(0)
            )

            rewards.append(
                reward
            )

            dones.append(
                done
            )

            log_probs.append(
                log_prob.squeeze(0)
            )

            values.append(
                value.squeeze()
            )

            obs = next_obs

            # =================================================
            # Episode 종료
            # =================================================
            if done:

                episode_count += 1

                result = info.get("result")

                # goal_reached = 1
                # 나머지 = 0
                success = (
                    1
                    if result == "goal_reached"
                    else 0
                )

                success_history.append(success)

                # ------------------------------------------------
                # Episode별 로그
                # ------------------------------------------------
                print(
                    f"Episode {episode_count:4d} | "
                    f"Reward {episode_reward:7.3f} | "
                    f"Steps {env.step_count:3d} | "
                    f"Invalid {env.invalid_count:3d} | "
                    f"BFS {shortest} | "
                    f"Result {result} | "
                    f"UP {action_counts[0]} | "
                    f"DOWN {action_counts[1]} | "
                    f"LEFT {action_counts[2]} | "
                    f"RIGHT {action_counts[3]}"
                )
                # 목표 episode까지 다 학습했으면 종료
                if episode_count >= episodes:
                    break

                # ------------------------------------------------
                # 다음 episode 시작
                # ------------------------------------------------
                obs = env.reset()

                shortest = (
                    env.get_shortest_distance()
                )

                episode_reward = 0.0

                action_counts = [
                    0, 0, 0, 0
                ]

        # =====================================================
        # Rollout 수집 종료
        # =====================================================

        rollout_size = len(states)

        if rollout_size == 0:
            continue

        # -----------------------------------------------------
        # rollout 마지막 상태의 Value
        # -----------------------------------------------------
        if dones[-1]:

            last_value = 0.0

        else:

            with torch.no_grad():

                last_obs_tensor = torch.tensor(
                    obs,
                    dtype=torch.float32
                ).unsqueeze(0)

                _, last_value_tensor = model(
                    last_obs_tensor
                )

                last_value = (
                    last_value_tensor.item()
                )

        # =====================================================
        # GAE 계산
        # =====================================================

        advantages = [
            0.0
            for _ in range(rollout_size)
        ]

        gae = 0.0

        for t in reversed(
            range(rollout_size)
        ):

            if t == rollout_size - 1:
                next_value = last_value

            else:
                next_value = (
                    values[t + 1].item()
                )

            # done이면 미래가치 사용하지 않음
            not_done = (
                1.0
                - float(dones[t])
            )

            # TD Error
            delta = (
                rewards[t]
                + gamma
                * next_value
                * not_done
                - values[t].item()
            )

            # GAE
            gae = (
                delta
                + gamma
                * gae_lambda
                * not_done
                * gae
            )

            advantages[t] = gae

        # =====================================================
        # Tensor 변환
        # =====================================================

        states_tensor = torch.stack(
            states
        )

        actions_tensor = torch.stack(
            actions
        ).long()

        old_log_probs_tensor = torch.stack(
            log_probs
        )

        values_tensor = torch.stack(
            values
        )

        advantages_tensor = torch.tensor(
            advantages,
            dtype=torch.float32
        )

        # Return = Advantage + V(s)
        returns_tensor = (
            advantages_tensor
            + values_tensor
        )

        # =====================================================
        # Advantage 정규화
        # =====================================================

        if len(advantages_tensor) > 1:

            advantages_tensor = (
                advantages_tensor
                - advantages_tensor.mean()
            ) / (
                advantages_tensor.std()
                + 1e-8
            )

        # =====================================================
        # PPO Minibatch Update
        # =====================================================

        for _ in range(update_epochs):

            # rollout 데이터 섞기
            indices = torch.randperm(
                rollout_size
            )

            # -------------------------------------------------
            # minibatch 단위 학습
            # -------------------------------------------------
            for start in range(
                0,
                rollout_size,
                minibatch_size
            ):

                end = (
                    start
                    + minibatch_size
                )

                mb_idx = indices[
                    start:end
                ]

                # ---------------------------------------------
                # minibatch 데이터
                # ---------------------------------------------

                mb_states = (
                    states_tensor[mb_idx]
                )

                mb_actions = (
                    actions_tensor[mb_idx]
                )

                mb_old_log_probs = (
                    old_log_probs_tensor[
                        mb_idx
                    ]
                )

                mb_advantages = (
                    advantages_tensor[
                        mb_idx
                    ]
                )

                mb_returns = (
                    returns_tensor[
                        mb_idx
                    ]
                )

                # ---------------------------------------------
                # 현재 정책에서 다시 계산
                # ---------------------------------------------

                new_logits, new_values = model(
                    mb_states
                )

                new_dist = Categorical(
                    logits=new_logits
                )

                new_log_probs = (
                    new_dist.log_prob(
                        mb_actions
                    )
                )

                entropy = (
                    new_dist.entropy().mean()
                )

                # ---------------------------------------------
                # PPO Ratio
                # ---------------------------------------------

                ratio = torch.exp(
                    new_log_probs
                    - mb_old_log_probs
                )

                # ---------------------------------------------
                # PPO Clipped Objective
                # ---------------------------------------------

                surr1 = (
                    ratio
                    * mb_advantages
                )

                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - clip_eps,
                        1.0 + clip_eps
                    )
                    * mb_advantages
                )

                actor_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                # ---------------------------------------------
                # Critic Loss
                # ---------------------------------------------

                critic_loss = (
                    nn.functional.mse_loss(
                        new_values.squeeze(-1),
                        mb_returns
                    )
                )

                # ---------------------------------------------
                # 전체 PPO Loss
                # ---------------------------------------------

                loss = (
                    actor_loss
                    + value_coef
                    * critic_loss
                    - entropy_coef
                    * entropy
                )

                # ---------------------------------------------
                # Gradient Update
                # ---------------------------------------------

                optimizer.zero_grad()

                loss.backward()

                # Gradient 폭주 방지
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=0.5
                )

                optimizer.step()

    # =========================================================
    # 학습 종료 후 전체 통계 출력
    # =========================================================

    print("\n\n========== Training Summary ==========")

    total_episodes = len(success_history)

    # 1000 episode 단위 성공률
    for start in range(0, total_episodes, 1000):

        end = min(start + 1000, total_episodes)

        section = success_history[start:end]

        section_success_rate = (
            sum(section) / len(section)
        ) * 100

        print(
            f"Episode {start + 1:4d} ~ {end:4d} "
            f"| Success Rate : "
            f"{section_success_rate:.2f}%"
        )


    # 전체 누적 성공률
    total_success_rate = (
        sum(success_history)
        / len(success_history)
    ) * 100

    print("--------------------------------------")

    print(
        f"Total Episodes   : {total_episodes}"
    )

    print(
        f"Total Success    : {sum(success_history)}"
    )

    print(
        f"Overall Success Rate : "
        f"{total_success_rate:.2f}%"
    )

    print("======================================")
    # =========================================================
    # 모든 학습 종료 후 최종 모델만 저장
    # =========================================================

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\n모델 저장 완료")

    return model

def evaluate_ppo(model, test_episodes=5, render=True):
    env = RouteFind()

    renderer = RouteRenderer(env) if render else None

    success_count = 0
    total_success_steps = 0
    total_efficiency = 0.0

    model.eval()

    for episode in range(test_episodes):

        obs = env.reset()
        bfs_distance = env.get_shortest_distance()

        done = False

        while not done:

            # pygame 창 종료 처리
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_tensor)

                dist = Categorical(logits=logits)
                action = dist.sample().item()

            obs, _, done, info = env.step(action)

            # 화면에 현재 상태 그리기
            if render:
                renderer.draw()
                renderer.tick(fps=5)

        result = info.get("result")

        if result == "goal_reached":

            success_count += 1

            ppo_steps = env.step_count

            total_success_steps += ppo_steps

            efficiency = bfs_distance / ppo_steps

            total_efficiency += efficiency

            print(
                f"[TEST {episode}] "
                f"Success | "
                f"BFS={bfs_distance} | "
                f"PPO={ppo_steps} | "
                f"Efficiency={efficiency:.3f}"
            )

        else:

            print(
                f"[TEST {episode}] "
                f"Fail | "
                f"BFS={bfs_distance} | "
                f"Result={result}"
            )

    if render:
        pygame.quit()

    success_rate = success_count / test_episodes

    if success_count > 0:
        avg_steps = total_success_steps / success_count
        avg_efficiency = total_efficiency / success_count
    else:
        avg_steps = 0
        avg_efficiency = 0

    print("\n========== PPO Evaluation ==========")
    print(f"Test Episodes : {test_episodes}")
    print(f"Success       : {success_count}")
    print(f"Success Rate  : {success_rate * 100:.2f}%")
    print(f"Average Steps : {avg_steps:.2f}")
    print(f"Path Efficiency : {avg_efficiency * 100:.2f}%")
    print("====================================")

def load_and_test():
    model = ActorCritic()

    model.load_state_dict(
        torch.load(
            MODEL_PATH, 
            weights_only=True
        )
    )

    model.eval()

    evaluate_ppo(
        model, 
        test_episodes=100,
        render=False
    )

if __name__ == "__main__":
    trained_model = train_ppo(episodes=500000)

    load_and_test()