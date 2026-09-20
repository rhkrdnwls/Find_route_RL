import random
import pygame
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical


class RouteFind:
    ACTIONS = {
        0: (-1, 0),  # 위
        1: (1, 0),   # 아래
        2: (0, -1),  # 왼쪽
        3: (0, 1),   # 오른쪽
    }

    def __init__(self, grid_size=10, n_obstacles=15, max_steps=100, seed=None):
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.robot_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.step_count = 0

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
        goal_weight = 0.1
        obstacle_weight = 0.01
        step_penalty = 0.01

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

        reward = step_penalty  # 매 스텝 작은 시간 페널티
        done = False
        info = {}

        # 경계를 벗어나면 이동 취소 (제자리)
        if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
            self.robot_pos = (new_r, new_c)

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
            reward = -1.0
            done = True
            info["result"] = "collision"

        # 목적지 도착
        elif self.robot_pos == self.goal_pos:
            reward = 1.0
            done = True
            info["result"] = "goal_reached"

        self.step_count += 1

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
    episodes=500,
    gamma=0.99,
    clip_eps=0.2,
    lr=3e-4,
    update_epochs=4
):
    env = RouteFind()
    model = ActorCritic()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 신경망 가중치를 실제로 수정하는 도구

    for episode in range(episodes):

        obs = env.reset() # 맵 얻어옴
        shortest = env.get_shortest_distance()
        done = False

        # rollout 저장소
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        total_reward = 0.0

        # =========================
        # 1. Rollout 수집
        # =========================
        while not done:

            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32
            ).unsqueeze(0)
            # Actor-Critic은 파이토치 신경망이라 기본적으로 파이토치 텐서를 입력으로 받는다.
            # Numpy -> Tensor -> 신경망의 변환이 필요하다
            # Numpy: 일반적인 수치 계산용
            # Tensor: 신경망 학습용, 자동 미분 지원

            # Actor-Critic 실행
            logits, value = model(obs_tensor)

            # 행동 분포
            dist = Categorical(logits=logits)

            # 행동 샘플링
            action = dist.sample()

            # 현재 정책에서 선택 행동의 log probability
            log_prob = dist.log_prob(action)

            # 실제 환경에 행동 전달
            next_obs, reward, done, info = env.step(
                action.item()
            )

            # rollout 저장
            states.append(obs_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            rewards.append(reward)
            log_probs.append(log_prob.squeeze(0).detach())
            values.append(value.squeeze().detach())
            dones.append(done)

            total_reward += reward
            obs = next_obs

        # =========================
        # 2. Return 계산
        # =========================
        returns = []
        G = 0.0

        for reward, done_flag in zip(
            reversed(rewards),
            reversed(dones)
        ):
            if done_flag:
                G = 0.0

            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        values_tensor = torch.stack(values)

        # =========================
        # 3. Advantage 계산
        # =========================
        advantages = returns - values_tensor

        # 안정화를 위해 normalization
        if len(advantages) > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-8)

        # 기존 rollout을 Tensor로 합침
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)

        old_log_probs = torch.stack(log_probs)

        # =========================
        # 4. PPO Update
        # =========================
        for _ in range(update_epochs):

            new_logits, new_values = model(states_tensor)

            new_dist = Categorical(logits=new_logits)

            new_log_probs = new_dist.log_prob(
                actions_tensor
            )

            entropy = new_dist.entropy().mean()

            # PPO 확률비
            ratios = torch.exp(
                new_log_probs - old_log_probs
            )

            # unclipped objective
            surr1 = ratios * advantages

            # clipped objective
            surr2 = torch.clamp(
                ratios,
                1.0 - clip_eps,
                1.0 + clip_eps
            ) * advantages

            actor_loss = -torch.min(
                surr1,
                surr2
            ).mean()

            # Critic loss
            new_values = new_values.squeeze(-1)

            critic_loss = nn.functional.mse_loss(
                new_values,
                returns
            )

            # 전체 loss
            loss = (
                actor_loss
                + 0.5 * critic_loss
                - 0.01 * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # =========================
        # 5. 학습 로그
        # =========================

        print(
            f"Episode {episode:4d} | "
            f"Reward {total_reward:7.3f} | "
            f"Steps {env.step_count:3d} | "
            f"BFS {shortest} | "
            f"Result {info.get('result')}"
        )

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
                logits, value = model(obs_tensor)

                # 평가에서는 가장 확률 높은 행동 선택
                action = torch.argmax(
                    logits,
                    dim=1
                ).item()

            obs, reward, done, info = env.step(action)

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


if __name__ == "__main__":
    trained_model = train_ppo(episodes=200000)

    evaluate_ppo(trained_model,test_episodes=5, render=True)