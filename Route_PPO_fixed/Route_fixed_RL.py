import random
import pygame
from collections import deque
import numpy as np


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

    def step(self, action):
        assert action in self.ACTIONS, f"invalid action: {action}"
        # assert는 이 조건이 반드시 참이여야 하는 검사
        # action이 self.ACTIONS 안에 있어야 한다 --> 없으면 에러 발생

        old_distance = self.manhattan_distance(
            # 변화량 측정을 위해 이동 전 거리
            self.robot_pos,
            self.goal_pos
        )

        dr, dc = self.ACTIONS[action]
        new_r = self.robot_pos[0] + dr # row(행) 방향으로 얼마나 움직일까
        new_c = self.robot_pos[1] + dc # col(열) 방향으로 얼마나 움직일까

        reward = -0.01  # 매 스텝 작은 시간 페널티
        done = False
        info = {}

        # 경계를 벗어나면 이동 취소 (제자리)
        if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
            self.robot_pos = (new_r, new_c)

        new_distance = self.manhattan_distance(
            self.robot_pos,
            self.goal_pos
        )

        distance_change = old_distance - new_distance

        reward += 0.1 * distance_change

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


if __name__ == "__main__":
    env = RouteFind()
    obs = env.reset()
    print("Observation shape:", obs.shape)
    print("BFS 최단거리:", env.get_shortest_distance())

    renderer = RouteRenderer(env)
    renderer.draw()

    # 창이 바로 안 꺼지게 유지
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()