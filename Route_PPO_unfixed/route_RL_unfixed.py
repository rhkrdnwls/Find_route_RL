import random
import pygame


class routeRL:
    ACTIONS = {
        # 가능한 행동 공간
        0: (-1, 0),  # 위
        1: (1, 0),   # 아래
        2: (0, -1),  # 왼쪽
        3: (0, 1),   # 오른쪽
    }

    def __init__(self, grid_size=10, n_obstacles=8, max_steps=100, seed=None):
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.robot_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.step_count = 0

    def reset(self):
        # 이 reset을 통해 매 에피소드마다 목적지, 장애물, 로봇의 위치가 전부 달라지는 동적 맵이 생성된다.
        occupied = set()

        # 목적지 배치
        self.goal_pos = self._random_empty_cell(occupied)
        # 빈칸 중 임의의 좌표 하나를 뽑아 초록색 칸으로 지정한다.
        occupied.add(self.goal_pos)

        # 장애물 배치 (겹치지 않게)
        # 목적지를 제외한 빈칸 중 좌표를 뽑아 리스트에 등록한다.
        self.obstacles = []
        for _ in range(self.n_obstacles):
            pos = self._random_empty_cell(occupied)
            # 이미 누군가가 차지한 좌표가 아닐 때까지 난수로 계속 뽑는다.
            occupied.add(pos)
            self.obstacles.append(pos)

        # 로봇 배치
        self.robot_pos = self._random_empty_cell(occupied)
        occupied.add(self.robot_pos)

        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        assert action in self.ACTIONS, f"invalid action: {action}"
        dr, dc = self.ACTIONS[action]
        new_r = self.robot_pos[0] + dr
        new_c = self.robot_pos[1] + dc

        reward = -0.01  # 매 스텝 작은 시간 페널티
        # 로봇이 쓸데없이 맴돌지 않고 최단 경로로 가라고 압박하는 역할
        done = False
        info = {}

        # 경계를 벗어나면 이동 취소 (제자리)
        # 일명 벽 충돌 예외 처리
        if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
            self.robot_pos = (new_r, new_c)

        # 장애물과 충돌
        if self.robot_pos in self.obstacles:
            # 이동한 위치가 self.obstacles 리스트 안에 포함되어 있는 경우
            reward = -1.0
            done = True # 에피소드 즉시 실패 + 종료
            info["result"] = "collision" 

        # 목적지 도달
        elif self.robot_pos == self.goal_pos: 
            # 이동한 위치가 self.goal_pos와 일치 
            reward = 1.0
            done = True
            info["result"] = "goal_reached"

        self.step_count += 1
        if self.step_count >= self.max_steps: # 스텝 수가 max_steps에 도달하면 강제 종료
            done = True
            info.setdefault("result", "timeout")

        return self._get_obs(), reward, done, info

    def _get_obs(self):
    # 에이전트가 현재 상황을 파악할 수 있도록 1차원 리스트 형태로 정보를 묶어 반환한다.
    # [로봇이 위치한  row, column, 목적지 위치인 row, column, 장애물들 위치인 row, column]
    # 에이전트는 이 20개 숫자를 입력받아 로봇과 장애물, 목적지의 상대적 위치 관계를 파악하게 된다.
        # 가장 단순한 형태: 로봇/목적지 좌표 + 장애물 좌표를 하나의 벡터로 concat
        # (나중에 신경망 입력 설계할 때 이 부분을 8x8xN 텐서 등으로 바꿔도 됨)
        obs = list(self.robot_pos) + list(self.goal_pos)
        for obs_pos in self.obstacles:
            obs += list(obs_pos)
        return obs

    def _random_empty_cell(self, occupied):
        while True:
            pos = (self.rng.randrange(self.grid_size), self.rng.randrange(self.grid_size))
            if pos not in occupied:
                return pos


class GridNavRenderer:
    # 시각화 모듈
    """간단한 pygame 렌더러. env.robot_pos / goal_pos / obstacles를 그려줌."""

    CELL_SIZE = 40

    def __init__(self, env: routeRL):
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


def run_random_agent_demo(episodes=5, render=True):
    env = routeRL(grid_size=10, n_obstacles=8, max_steps=50)
    renderer = GridNavRenderer(env) if render else None

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            action = random.choice(list(routeRL.ACTIONS.keys()))
            # 위에서 봤던 4가지 행동 중 하나를 25%의 균등한 확률로 맹목적으로 뽑아낸다.
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if render:
                renderer.draw()
                renderer.tick(fps=10)

        print(f"[episode {ep}] steps={env.step_count} total_reward={total_reward:.2f} result={info.get('result')}")

    if render:
        pygame.quit()


if __name__ == "__main__":
    run_random_agent_demo(episodes=5, render=True)
    