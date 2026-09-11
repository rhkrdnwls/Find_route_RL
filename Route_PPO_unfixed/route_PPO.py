import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


# =====================================================================
# 1. 커스텀 그리드 환경 (Gymnasium 표준 인터페이스로 래핑)
# =====================================================================
class GridNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    ACTIONS = {
        0: (-1, 0),  # 위
        1: (1, 0),   # 아래
        2: (0, -1),  # 왼쪽
        3: (0, 1),   # 오른쪽
    }

    def __init__(self, grid_size=10, n_obstacles=8, max_steps=60, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 행동 공간: 4가지 (상, 하, 좌, 우)
        self.action_space = spaces.Discrete(4)

        # 관측 공간: 20차원 벡터 (로봇 2 + 목적지 2 + 장애물 8개*2 = 20)
        obs_dim = 2 + 2 + (n_obstacles * 2)
        self.observation_space = spaces.Box(
            low=0.0, high=float(grid_size), shape=(obs_dim,), dtype=np.float32
        )

        self.robot_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.step_count = 0

        # Pygame 렌더러 변수
        self.window = None
        self.clock = None
        self.cell_size = 40

    def _random_empty_cell(self, occupied):
        while True:
            pos = (self.np_random.integers(0, self.grid_size), self.np_random.integers(0, self.grid_size))
            if pos not in occupied:
                return pos

    def _get_obs(self):
        obs = list(self.robot_pos) + list(self.goal_pos)
        for obs_pos in self.obstacles:
            obs += list(obs_pos)
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        occupied = set()

        # 목적지 배치
        self.goal_pos = self._random_empty_cell(occupied)
        occupied.add(self.goal_pos)

        # 장애물 배치
        self.obstacles = []
        for _ in range(self.n_obstacles):
            pos = self._random_empty_cell(occupied)
            occupied.add(pos)
            self.obstacles.append(pos)

        # 로봇 배치
        self.robot_pos = self._random_empty_cell(occupied)
        occupied.add(self.robot_pos)

        self.step_count = 0
        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        prev_dist = abs(self.robot_pos[0] - self.goal_pos[0]) + abs(self.robot_pos[1] - self.goal_pos[1])

        dr, dc = self.ACTIONS[int(action)]
        new_r = self.robot_pos[0] + dr
        new_c = self.robot_pos[1] + dc

        reward = -0.08  # 스텝 페널티
        terminated = False
        truncated = False
        info = {}

        # 경계 체크 (벽 밖으로 나가면 제자리)
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            reward -= 0.2  # 벽 들이받기 페널티 (가장자리 비비기 방지)
            # 위치는 제자리 유지 (self.robot_pos 변경 없음)
        else:
            self.robot_pos = (new_r, new_c)
        
        # 2. 이동 후 목적지까지의 거리 계산 및 차분 보상 부여
        curr_dist = abs(self.robot_pos[0] - self.goal_pos[0]) + abs(self.robot_pos[1] - self.goal_pos[1])
        # 목표에 1칸 다가가면 +0.1, 멀어지면 -0.1
        reward += 0.1 * (prev_dist - curr_dist)

        # 장애물 충돌 판정
        if self.robot_pos in self.obstacles:
            reward = -4.0
            terminated = True
            info["result"] = "collision"
        # 목적지 도달 판정
        elif self.robot_pos == self.goal_pos:
            reward = 9.0  # 명확한 피드백을 위해 보상 상향
            terminated = True
            info["result"] = "goal_reached"

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
            info.setdefault("result", "timeout")

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            size = self.grid_size * self.cell_size
            self.window = pygame.display.set_mode((size, size))
            pygame.display.set_caption("PPO Grid Navigation")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        cs = self.cell_size
        self.window.fill((240, 240, 235))

        # 그리드 선
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.window, (200, 200, 200), (0, i * cs), (self.grid_size * cs, i * cs))
            pygame.draw.line(self.window, (200, 200, 200), (i * cs, 0), (i * cs, self.grid_size * cs))

        # 장애물 (회색 사각형)
        for r, c in self.obstacles:
            rect = pygame.Rect(c * cs, r * cs, cs, cs)
            pygame.draw.rect(self.window, (90, 90, 90), rect)

        # 목적지 (초록색 원)
        gr, gc = self.goal_pos
        pygame.draw.circle(self.window, (40, 180, 60), (gc * cs + cs // 2, gr * cs + cs // 2), cs // 2 - 4)

        # 로봇 (파란색 원)
        rr, rc = self.robot_pos
        pygame.draw.circle(self.window, (40, 90, 220), (rc * cs + cs // 2, rr * cs + cs // 2), cs // 2 - 4)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


# =====================================================================
# 2. 하이퍼파라미터 및 CleanRL PPO 설정
# =====================================================================
@dataclass
class Args:
    exp_name: str = "PPO"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # 그리드 맵 세팅 및 학습 스텝
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(idx):
    def thunk():
        env = GridNavEnv(grid_size=10, n_obstacles=8, max_steps=60)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# =====================================================================
# 3. Actor-Critic 에이전트 (20차원 입력 -> 4개 행동 출력)
# =====================================================================
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.prod(envs.single_observation_space.shape)
        action_dim = envs.single_action_space.n  # 4로 자동 연결됨

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        # Critic은 20차원 벡터를 입력받아 현재 판세가 얼마나 유리한지를 나타내는 스칼라 상태 가치 V(s) 1개를 출력한다. 
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),  # 출력 노드 4개
        )
        # Actor는 20차원 벡터를 입력받아 4개 행동(상/하/좌/우)에 대한 로짓 점수 4개를 출력한다.
        # Actor는 std=0.01로 설정해 초기 탐색 시 4개 방향을 고르게 시도하도록 한다.

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        # 출력 Logit에 소프트맥스를 적용한 카테고리컬 확률 분포를 형성
        # 해당 분포에서 행동을 하나 샘플링하고, 그 행동의 로그 확률(log_prob), 정책의 탐색 분산 수준(Entropy), Critic의 가치 추정치(critic) 한 번에 반환    


# =====================================================================
# 4. 메인 학습 및 평가 루프
# =====================================================================
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 병렬 환경 8개 동시 실행
    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 롤아웃 버퍼
    # 한 번의 롤아웃마다 128 스텝 * 8개 환경 = 1024개의 상태, 행동, 보상, 가치를 담아둘 텐서 저장소 호출
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print(">>> PPO 그리드 내비게이션 학습을 시작합니다... <<<")
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 롤아웃 데이터 수집
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            # 신경망울 거쳐 얻은 행동을 8개 환경에 넣고 한칸씩 전진

            # 에피소드 완료 로그 출력
            if "episode" in infos:
                for r, l in zip(infos["episode"]["r"], infos["episode"]["l"]):
                    if r is not None and not np.isnan(r):
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)
                        if r > 0:
                            print(f"[목적지 도달 성공!] Step: {global_step} | Return: {r:.2f} | Length: {l}")

        # GAE 계산
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            # 특정 행동이 평균 기대치보다 얼마나 더 좋았는지를 나타내는 Advantages와 Critic이 맞춰야 할 실제 타깃값인 returns를 반환

        # 데이터 평탄화
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 미니배치 최적화
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 정책 손실
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 가치 손실
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

    envs.close()
    writer.close()
    print(">>> 학습 완료! 시연 화면을 시작합니다. <<<")

    # =====================================================================
    # 5. 학습된 에이전트 실제 주행 시각화
    # =====================================================================
    import time

    test_env = GridNavEnv(grid_size=10, n_obstacles=8, max_steps=60, render_mode="human")
    print("\n>>> 시각화 창을 엽니다 (총 5회 테스트) <<<")

    for ep in range(19):
        obs, _ = test_env.reset()
        test_env.render()  # 초기 배치 즉시 렌더링
        done = False
        total_reward = 0.0
        step_num = 0

        while not done:
            time.sleep(0.15)  # 1스텝당 0.15초 유지

            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1)
                

            obs, reward, terminated, truncated, info = test_env.step(action.item())
            test_env.render()  # 이동 후 화면 갱신
            total_reward += reward
            step_num += 1
            done = terminated or truncated

        print(f"[시연 {ep + 1}] 걸음수: {step_num} | 누적 보상: {total_reward:.2f} | 결과: {info.get('result')}")
        time.sleep(1.2)  # 에피소드 종료 후 결과 화면 1.2초 정지

    test_env.close()