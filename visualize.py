"""
Pygame visualization for the trained rocket landing agent.

Usage:
    uv run python visualize.py

Controls:
    SPACE  — restart episode
    Q/ESC  — quit
"""

import math
import random
import sys

import pygame
from stable_baselines3 import PPO

from rocket_env import RocketLandingEnv

# ── colours ──────────────────────────────────────────────────────────
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (60, 60, 60)
DARK_GREY = (30, 30, 30)
SKY_TOP = (10, 10, 40)
SKY_BOT = (40, 40, 90)
GROUND = (50, 50, 50)
PAD_COLOR = (200, 200, 200)
ROCKET_BODY = (220, 220, 230)
ROCKET_NOSE = (200, 60, 60)
FLAME_CORE = (255, 200, 60)
FLAME_MID = (255, 120, 20)
FLAME_TIP = (255, 60, 10)
HUD_GREEN = (0, 255, 100)
HUD_RED = (255, 60, 60)
HUD_YELLOW = (255, 220, 60)
STAR_COLOR = (200, 200, 255)

# ── window / scale ──────────────────────────────────────────────────
WIDTH, HEIGHT = 600, 800
FPS = 30  # visual fps (env steps at its own dt)
GROUND_Y = HEIGHT - 80  # pixel y of the ground line
MAX_ALT_PX = GROUND_Y - 60  # pixels available for altitude range

# how many env altitude metres fit on screen
VIEW_ALT = 300.0


def alt_to_y(z: float) -> int:
    """Convert altitude (m) to screen y pixel."""
    frac = z / VIEW_ALT
    return int(GROUND_Y - frac * MAX_ALT_PX)


# ── particles ────────────────────────────────────────────────────────
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "radius", "color")

    def __init__(self, x, y):
        angle = math.pi / 2 + random.uniform(-0.3, 0.3)
        speed = random.uniform(2, 6)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed * random.choice([-1, 1])
        self.vy = speed  # downward on screen
        self.max_life = random.randint(10, 25)
        self.life = self.max_life
        self.radius = random.randint(2, 5)
        self.color = random.choice([FLAME_CORE, FLAME_MID, FLAME_TIP])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # slight gravity on particles
        self.life -= 1

    def draw(self, surface):
        alpha = max(self.life / self.max_life, 0)
        r = max(int(self.radius * alpha), 1)
        c = tuple(int(ch * alpha) for ch in self.color)
        pygame.draw.circle(surface, c, (int(self.x), int(self.y)), r)


# ── stars (background) ──────────────────────────────────────────────
def make_stars(n=80):
    return [(random.randint(0, WIDTH), random.randint(0, GROUND_Y),
             random.randint(1, 2), random.randint(100, 255)) for _ in range(n)]


# ── drawing helpers ──────────────────────────────────────────────────
def draw_gradient_sky(surface):
    for y in range(GROUND_Y):
        t = y / GROUND_Y
        r = int(SKY_TOP[0] * (1 - t) + SKY_BOT[0] * t)
        g = int(SKY_TOP[1] * (1 - t) + SKY_BOT[1] * t)
        b = int(SKY_TOP[2] * (1 - t) + SKY_BOT[2] * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (WIDTH, y))


def draw_stars(surface, stars):
    for sx, sy, sr, sb in stars:
        pygame.draw.circle(surface, (sb, sb, min(sb + 40, 255)), (sx, sy), sr)


def draw_ground(surface):
    pygame.draw.rect(surface, GROUND, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    # landing pad
    pad_w, pad_h = 80, 6
    pad_x = WIDTH // 2 - pad_w // 2
    pygame.draw.rect(surface, PAD_COLOR, (pad_x, GROUND_Y - pad_h // 2, pad_w, pad_h))
    # pad markings
    for i in range(3):
        mx = pad_x + 15 + i * 25
        pygame.draw.rect(surface, HUD_YELLOW, (mx, GROUND_Y - pad_h // 2 + 1, 10, pad_h - 2))


def draw_rocket(surface, rx, ry, thrusting):
    # body
    body_w, body_h = 20, 50
    body_rect = pygame.Rect(rx - body_w // 2, ry - body_h, body_w, body_h)
    pygame.draw.rect(surface, ROCKET_BODY, body_rect, border_radius=3)

    # nose cone (triangle)
    nose_pts = [
        (rx, ry - body_h - 18),
        (rx - body_w // 2, ry - body_h),
        (rx + body_w // 2, ry - body_h),
    ]
    pygame.draw.polygon(surface, ROCKET_NOSE, nose_pts)

    # fins
    fin_pts_l = [
        (rx - body_w // 2, ry),
        (rx - body_w // 2 - 8, ry + 10),
        (rx - body_w // 2, ry - 12),
    ]
    fin_pts_r = [
        (rx + body_w // 2, ry),
        (rx + body_w // 2 + 8, ry + 10),
        (rx + body_w // 2, ry - 12),
    ]
    pygame.draw.polygon(surface, ROCKET_NOSE, fin_pts_l)
    pygame.draw.polygon(surface, ROCKET_NOSE, fin_pts_r)

    # nozzle
    nozzle_w = 12
    pygame.draw.rect(surface, GREY, (rx - nozzle_w // 2, ry, nozzle_w, 5))

    # flame glow when thrusting
    if thrusting:
        flame_len = random.randint(18, 35)
        flame_pts = [
            (rx - 6, ry + 5),
            (rx + 6, ry + 5),
            (rx + random.randint(-3, 3), ry + 5 + flame_len),
        ]
        pygame.draw.polygon(surface, FLAME_CORE, flame_pts)
        # inner flame
        inner_len = flame_len * 0.6
        inner_pts = [
            (rx - 3, ry + 5),
            (rx + 3, ry + 5),
            (rx + random.randint(-2, 2), int(ry + 5 + inner_len)),
        ]
        pygame.draw.polygon(surface, (255, 255, 200), inner_pts)


def draw_hud(surface, font, z, v, fuel, mass, step, episode, total_reward, status):
    lines = [
        f"Episode: {episode}",
        f"Step:    {step}",
        f"Alt:     {z:.1f} m",
        f"Vel:     {v:+.1f} m/s",
        f"Fuel:    {fuel:.1f} kg",
        f"Mass:    {mass:.1f} kg",
        f"Reward:  {total_reward:.1f}",
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, HUD_GREEN)
        surface.blit(txt, (15, 15 + i * 22))

    # velocity indicator color
    vel_color = HUD_GREEN if abs(v) < 5.0 else HUD_YELLOW if abs(v) < 10.0 else HUD_RED
    vel_label = font.render(f"  {'SAFE' if abs(v) < 5 else 'FAST' if abs(v) < 10 else 'DANGER'}", True, vel_color)
    surface.blit(vel_label, (180, 15 + 3 * 22))

    # fuel bar
    bar_x, bar_y, bar_w, bar_h = WIDTH - 40, 60, 16, 200
    pygame.draw.rect(surface, GREY, (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2))
    fuel_frac = fuel / 20.0
    fuel_color = HUD_GREEN if fuel_frac > 0.3 else HUD_YELLOW if fuel_frac > 0.1 else HUD_RED
    fill_h = int(bar_h * fuel_frac)
    pygame.draw.rect(surface, fuel_color, (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))
    fuel_label = font.render("FUEL", True, WHITE)
    surface.blit(fuel_label, (bar_x - 5, bar_y - 20))

    # status message
    if status:
        color = HUD_GREEN if "LANDED" in status else HUD_RED
        status_font = pygame.font.SysFont("consolas", 32, bold=True)
        txt = status_font.render(status, True, color)
        rect = txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60))
        surface.blit(txt, rect)

        hint = font.render("SPACE to restart  |  Q to quit", True, WHITE)
        hint_rect = hint.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 25))
        surface.blit(hint, hint_rect)


# ── main ─────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Rocket Landing — PPO Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    # pre-render the sky gradient onto a surface (avoids redrawing every frame)
    sky_surface = pygame.Surface((WIDTH, HEIGHT))
    draw_gradient_sky(sky_surface)
    draw_ground(sky_surface)
    stars = make_stars()

    env = RocketLandingEnv()
    model = PPO.load("rocket_ppo_v2")

    episode = 0
    running = True

    while running:
        # ── new episode ──────────────────────────────────────────────
        episode += 1
        obs, info = env.reset()
        total_reward = 0.0
        boost_steps = 50  # forced thrust for this many steps before agent takes over
        done = False
        status = ""
        particles = []
        last_action = 0

        while not done and running:
            # ── events ───────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        done = True  # restart

            if not running:
                break

            # ── agent step ───────────────────────────────────────────
            if env.steps < boost_steps:
                action = 1  # forced thrust during boost phase
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            last_action = int(action)

            z, v, fuel, mass = obs

            # ── rocket screen position ───────────────────────────────
            rx = WIDTH // 2
            ry = alt_to_y(z)

            # ── particles ────────────────────────────────────────────
            if last_action == 1 and fuel > 0:
                for _ in range(3):
                    particles.append(Particle(rx + random.randint(-4, 4), ry + 5))

            particles = [p for p in particles if p.life > 0]
            for p in particles:
                p.update()

            # ── status on termination ────────────────────────────────
            if terminated:
                if z <= 0 and abs(v) < env.safe_velocity:
                    status = f"LANDED  v={v:+.1f} m/s"
                elif z <= 0:
                    status = f"CRASHED  v={v:+.1f} m/s"
                else:
                    status = "LOST IN SPACE"
            elif truncated:
                status = "TIMEOUT"

            # ── draw ─────────────────────────────────────────────────
            screen.blit(sky_surface, (0, 0))
            draw_stars(screen, stars)

            for p in particles:
                p.draw(screen)

            draw_rocket(screen, rx, ry, last_action == 1 and fuel > 0)
            draw_hud(screen, font, z, v, fuel, mass, env.steps, episode,
                     total_reward, status if done else "")

            pygame.display.flip()
            clock.tick(FPS)

        # ── pause on episode end so user can see result ──────────
        if running and not done:
            continue

        waiting = True
        while waiting and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        waiting = False

            # keep drawing the final frame
            screen.blit(sky_surface, (0, 0))
            draw_stars(screen, stars)
            for p in particles:
                p.draw(screen)
            draw_rocket(screen, rx, ry, False)
            draw_hud(screen, font, z, v, fuel, mass, env.steps, episode,
                     total_reward, status)
            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
