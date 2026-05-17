import os
import math
import pygame
import torch
import torch.nn as nn
import random
from collections import deque
import torch.optim as optim

os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.init()

# ----------------------------
# FENÊTRE
# ----------------------------
info = pygame.display.Info()
WIDTH = int(info.current_w * 0.9)
HEIGHT = int(info.current_h * 0.9)

TRACK_WIDTH = int(WIDTH * 0.70)
PANEL_WIDTH = WIDTH - TRACK_WIDTH

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Race IA")

clock = pygame.time.Clock()
FPS = 60

# ----------------------------
# COULEURS
# ----------------------------
GRASS = (200, 255, 200)
TRACK = (255, 255, 255)
BORDER = (0, 0, 0)
CAR_BLUE = (30, 100, 200)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 180, 0)
ORANGE = (255, 150, 0)
DARK = (40, 40, 40)
LIGHT_GRAY = (230, 230, 230)

# ----------------------------
# CIRCUIT
# ----------------------------
CENTER_Y = HEIGHT // 2
LEFT_X = int(TRACK_WIDTH * 0.25)
RIGHT_X = int(TRACK_WIDTH * 0.75)

OUTER_RADIUS = int(HEIGHT * 0.32)
INNER_RADIUS = int(HEIGHT * 0.18)
BORDER_WIDTH = 6

# ----------------------------
# VOITURE
# ----------------------------
START_X = TRACK_WIDTH // 2
START_Y = CENTER_Y - OUTER_RADIUS + 60
START_ANGLE = 0.0

car_x = TRACK_WIDTH // 2
car_y = CENTER_Y - OUTER_RADIUS + 60
car_angle = 0.0  # 0° = vers la droite
car_speed = 0.0

MAX_SPEED = 5
ACCELERATION = 0.15
FRICTION = 0.96
TURN_SPEED = 5.0

CAR_W = 60
CAR_H = 30

# ----------------------------
# CAPTEURS
# ----------------------------
SENSOR_LENGTH = 40
SENSOR_ANGLES = [-45, 0, 45]
LIDAR = (200, 200, 200)

# ----------------------------
# ACTIONS
# ----------------------------
ACTIONS = [
    "fort_gauche",
    "gauche",
    "tout_droit",
    "droite",
    "fort_droite",
]

pygame.font.init()
font = pygame.font.SysFont("Arial", 18)

# ----------------------------
# RÉSEAU DE NEURONES
# ----------------------------
class CarBrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.network(x)

MODEL_FILE = "car_brain.pth"
GAMMA = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 5000

epsilon = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

memory = deque(maxlen=MEMORY_SIZE)

def save_brain(brain):
    torch.save(brain.state_dict(), MODEL_FILE)
    print(f"Réseau sauvegardé dans {MODEL_FILE}")

def draw_text(surface, text, x, y, color=(0, 0, 0)):
    img = font.render(text, True, color)
    surface.blit(img, (x, y))

def load_brain(brain):
    if os.path.exists(MODEL_FILE):
        brain.load_state_dict(torch.load(MODEL_FILE))
        brain.eval()
        print(f"Réseau rechargé depuis {MODEL_FILE}")
    else:
        print("Aucun réseau sauvegardé trouvé. Nouveau réseau créé.")

def draw_track(surface):
    surface.fill(GRASS, (0, 0, TRACK_WIDTH, HEIGHT))

    # Bord extérieur
    pygame.draw.rect(
        surface,
        BORDER,
        (LEFT_X, CENTER_Y - OUTER_RADIUS, RIGHT_X - LEFT_X, 2 * OUTER_RADIUS),
    )
    pygame.draw.circle(surface, BORDER, (LEFT_X, CENTER_Y), OUTER_RADIUS)
    pygame.draw.circle(surface, BORDER, (RIGHT_X, CENTER_Y), OUTER_RADIUS)

    # Route
    pygame.draw.rect(
        surface,
        TRACK,
        (
            LEFT_X,
            CENTER_Y - OUTER_RADIUS + BORDER_WIDTH,
            RIGHT_X - LEFT_X,
            2 * (OUTER_RADIUS - BORDER_WIDTH),
        ),
    )
    pygame.draw.circle(surface, TRACK, (LEFT_X, CENTER_Y), OUTER_RADIUS - BORDER_WIDTH)
    pygame.draw.circle(surface, TRACK, (RIGHT_X, CENTER_Y), OUTER_RADIUS - BORDER_WIDTH)

    # Bord intérieur
    pygame.draw.rect(
        surface,
        BORDER,
        (LEFT_X, CENTER_Y - INNER_RADIUS, RIGHT_X - LEFT_X, 2 * INNER_RADIUS),
    )
    pygame.draw.circle(surface, BORDER, (LEFT_X, CENTER_Y), INNER_RADIUS)
    pygame.draw.circle(surface, BORDER, (RIGHT_X, CENTER_Y), INNER_RADIUS)

    # Herbe intérieure
    pygame.draw.rect(
        surface,
        GRASS,
        (
            LEFT_X,
            CENTER_Y - INNER_RADIUS + BORDER_WIDTH,
            RIGHT_X - LEFT_X,
            2 * (INNER_RADIUS - BORDER_WIDTH),
        ),
    )
    pygame.draw.circle(surface, GRASS, (LEFT_X, CENTER_Y), INNER_RADIUS - BORDER_WIDTH)
    pygame.draw.circle(surface, GRASS, (RIGHT_X, CENTER_Y), INNER_RADIUS - BORDER_WIDTH)

def draw_network_view(surface, brain, sensor_data, panel_x, y, w, h):
    pygame.draw.rect(surface, (245, 245, 245), (panel_x, y, w, h))
    pygame.draw.rect(surface, DARK, (panel_x, y, w, h), 2)

    layers_x = [
        panel_x + int(w * 0.15),
        panel_x + int(w * 0.40),
        panel_x + int(w * 0.65),
        panel_x + int(w * 0.88),
    ]

    neurons_per_layer = [6, 6, 6, 5]
    neuron_positions = []

    for layer_index, count in enumerate(neurons_per_layer):
        positions = []
        for i in range(count):
            ny = y + 20 + int((h * 0.45) * (i + 1) / (count + 1))
            positions.append((layers_x[layer_index], ny))
            pygame.draw.circle(surface, BORDER, (layers_x[layer_index], ny), 5)
        neuron_positions.append(positions)

    # Récupération des poids
    weights = []
    for module in brain.network:
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach())

    # Dessin simplifié des connexions
    for layer_idx in range(len(weights)):
        weight_matrix = weights[layer_idx]

        src_positions = neuron_positions[layer_idx]
        dst_positions = neuron_positions[layer_idx + 1]

        for i, src in enumerate(src_positions):
            for j, dst in enumerate(dst_positions):
                wi = min(i, weight_matrix.shape[1] - 1)
                wj = min(j, weight_matrix.shape[0] - 1)

                value = float(weight_matrix[wj, wi])
                intensity = int(max(0, min(255, 128 + value * 600)))

                color = (intensity, intensity, intensity)
                pygame.draw.line(surface, color, src, dst, 1)

    # Sorties live du réseau
    state = sensor_data_to_tensor(sensor_data)

    with torch.no_grad():
        q_values = brain(state)

    q_min = float(torch.min(q_values))
    q_max = float(torch.max(q_values))
    q_range = max(0.001, q_max - q_min)

    bar_y = y + int(h * 0.58)
    bar_h = 14
    max_bar_w = int(w * 0.55)

    for i, action_name in enumerate(ACTIONS):
        value = float(q_values[i])
        normalized = (value - q_min) / q_range
        bar_w = int(normalized * max_bar_w)

        text_y = bar_y + i * 24

        draw_text(surface, action_name, panel_x + 10, text_y - 2)
        pygame.draw.rect(
            surface,
            DARK,
            (panel_x + int(w * 0.42), text_y, max_bar_w, bar_h),
            1,
        )
        pygame.draw.rect(
            surface,
            (80, 80, 80),
            (panel_x + int(w * 0.42), text_y, bar_w, bar_h),
        )


def create_car_surface():
    car_surface = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)

    # Corps
    pygame.draw.rect(car_surface, CAR_BLUE, (10, 5, 40, 20))

    # Pneus
    pygame.draw.rect(car_surface, BORDER, (40, 2, 12, 6))
    pygame.draw.rect(car_surface, BORDER, (40, 22, 12, 6))
    pygame.draw.rect(car_surface, BORDER, (10, 2, 12, 6))
    pygame.draw.rect(car_surface, BORDER, (10, 22, 12, 6))

    # Phares avant
    pygame.draw.circle(car_surface, YELLOW, (55, 10), 5)
    pygame.draw.circle(car_surface, YELLOW, (55, 20), 5)

    # Feux arrière
    pygame.draw.rect(car_surface, RED, (2, 8, 6, 6))
    pygame.draw.rect(car_surface, RED, (2, 16, 6, 6))

    return car_surface


def draw_car(surface, x, y, angle):
    car_surface = create_car_surface()
    rotated = pygame.transform.rotate(car_surface, -angle)
    rect = rotated.get_rect(center=(x, y))
    surface.blit(rotated, rect)


def update_car(x, y, angle, speed):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        speed += ACCELERATION
    if keys[pygame.K_DOWN]:
        speed -= ACCELERATION

    speed = max(-MAX_SPEED, min(MAX_SPEED, speed))
    speed *= FRICTION

    if abs(speed) > 0.1:
        if keys[pygame.K_LEFT]:
            angle -= TURN_SPEED
        if keys[pygame.K_RIGHT]:
            angle += TURN_SPEED

    rad = math.radians(angle)
    x += math.cos(rad) * speed
    y += math.sin(rad) * speed

    return x, y, angle, speed


def get_sensor_points(x, y, angle):
    points_by_sensor = []

    # Le nez de la voiture est à environ CAR_W / 2 pixels du centre
    nose_distance = CAR_W // 2

    car_rad = math.radians(angle)
    nose_x = x + math.cos(car_rad) * nose_distance
    nose_y = y + math.sin(car_rad) * nose_distance

    for sensor_angle in SENSOR_ANGLES:
        absolute_angle = math.radians(angle + sensor_angle)
        points = []

        for distance in range(1, SENSOR_LENGTH + 1):
            px = int(nose_x + math.cos(absolute_angle) * distance)
            py = int(nose_y + math.sin(absolute_angle) * distance)
            points.append((px, py))

        points_by_sensor.append(points)

    return points_by_sensor

def pixel_is_dangerous(surface, x, y):
    if x < 0 or x >= TRACK_WIDTH or y < 0 or y >= HEIGHT:
        return True

    color = surface.get_at((int(x), int(y)))[:3]
    return color == BORDER or color == GRASS


def get_car_collision(surface, x, y):
    test_points = [
        (x, y),
        (x + 20, y),
        (x - 20, y),
        (x, y + 10),
        (x, y - 10),
    ]

    for px, py in test_points:
        if pixel_is_dangerous(surface, px, py):
            return True

    return False


def analyse_sensors(surface, sensor_points):
    sensor_data = []
    danger_ahead = False

    for points in sensor_points:
        line_data = []

        for px, py in points:
            if px < 0 or px >= TRACK_WIDTH or py < 0 or py >= HEIGHT:
                line_data.append(1)
                danger_ahead = True
                continue

            color = surface.get_at((px, py))[:3]

            if color == BORDER:
                line_data.append(1)
                danger_ahead = True
            elif color == TRACK:
                line_data.append(0)
            else:
                line_data.append(2)
                danger_ahead = True

        sensor_data.append(line_data)

    return sensor_data, danger_ahead


def draw_sensors(surface, sensor_points):
    for points in sensor_points:
        for px, py in points:
            if 0 <= px < TRACK_WIDTH and 0 <= py < HEIGHT:
                pygame.draw.circle(surface, LIGHT_GRAY, (px, py), 2)

def sensor_data_to_tensor(sensor_data):
    values = []

    for line in sensor_data:
        for pixel_value in line:
            if pixel_value == 0:      # route blanche
                values.append(0.0)
            elif pixel_value == 1:    # bord noir
                values.append(1.0)
            else:                     # herbe / hors piste
                values.append(0.5)

    return torch.tensor(values, dtype=torch.float32)

def choose_action(sensor_data):
    global epsilon

    state = sensor_data_to_tensor(sensor_data)

    if random.random() < epsilon:
        return random.randint(0, 4)

    with torch.no_grad():
        q_values = brain(state)

    return torch.argmax(q_values).item()

def train_brain():
    global epsilon

    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    states = torch.stack([item[0] for item in batch])
    actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
    rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    next_states = torch.stack([item[3] for item in batch])
    dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

    q_values = brain(states)
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_q_values = brain(next_states)
        max_next_q = torch.max(next_q_values, dim=1)[0]
        target_q = rewards + GAMMA * max_next_q * (1 - dones)

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

def apply_ai_action(angle, action_index):
    if action_index == 0:
        angle -= 9      # fortement à gauche
    elif action_index == 1:
        angle -= 4      # un peu à gauche
    elif action_index == 2:
        angle += 0      # tout droit
    elif action_index == 3:
        angle += 4      # un peu à droite
    elif action_index == 4:
        angle += 9      # fortement à droite

    return angle

def compute_reward(collision, sensor_data, speed, action):
    reward = 0.0

    # 1. Avancer est positif
    reward += speed * 0.2

    # 2. Plus la route est dégagée devant, mieux c'est
    center_sensor = sensor_data[1]

    center_distance = SENSOR_LENGTH
    for i, value in enumerate(center_sensor):
        if value != 0:
            center_distance = i
            break

    reward += center_distance * 0.03

    # 3. Rester au centre : gauche et droite doivent être à peu près équilibrés
    left_sensor = sensor_data[0]
    right_sensor = sensor_data[2]

    def first_obstacle_distance(sensor):
        for i, value in enumerate(sensor):
            if value != 0:
                return i
        return SENSOR_LENGTH

    left_distance = first_obstacle_distance(left_sensor)
    right_distance = first_obstacle_distance(right_sensor)

    balance = abs(left_distance - right_distance)
    reward -= balance * 0.02

    # 4. Préférer aller tout droit quand c'est possible
    if action == 2:
        reward += 0.15

    # 5. Pénaliser les grands coups de volant inutiles
    if action in [0, 4]:
        reward -= 0.15

    if action in [1, 3]:
        reward -= 0.05

    # 6. Collision = grosse punition
    if collision:
        reward -= 10.0

    return reward

def compute_reward_simple(collision, sensor_data, speed):
    reward = 0.0

    reward += speed * 0.1

    center_sensor = sensor_data[1]

    distance = SENSOR_LENGTH
    for i, value in enumerate(center_sensor):
        if value != 0:
            distance = i
            break

    reward += distance * 0.02

    if collision:
        reward -= 5.0

    return reward

def draw_panel(surface, angle, safety_state, sensor_data, reward, epsilon, memory_size):
    panel_x = TRACK_WIDTH
    pygame.draw.rect(surface, LIGHT_GRAY, (panel_x, 0, PANEL_WIDTH, HEIGHT))

    square_size = min(PANEL_WIDTH - 40, HEIGHT // 8)
    x = panel_x + (PANEL_WIDTH - square_size) // 2

    y1 = int(HEIGHT * 0.04)
    y2 = int(HEIGHT * 0.22)
    y3 = int(HEIGHT * 0.40)

    # Carré 1 : orientation
    pygame.draw.rect(surface, DARK, (x, y1, square_size, square_size), 3)
    cx = x + square_size // 2
    cy = y1 + square_size // 2
    length = square_size // 3
    rad = math.radians(angle)

    end_x = cx + math.cos(rad) * length
    end_y = cy + math.sin(rad) * length
    pygame.draw.line(surface, BORDER, (cx, cy), (end_x, end_y), 5)
    pygame.draw.circle(surface, BORDER, (int(end_x), int(end_y)), 6)

    # Carré 2 : sécurité
    if safety_state == "safe":
        color = GREEN
    elif safety_state == "warning":
        color = ORANGE
    else:
        color = RED

    pygame.draw.rect(surface, color, (x, y2, square_size, square_size))
    pygame.draw.rect(surface, DARK, (x, y2, square_size, square_size), 3)

    # Carré 3 : capteurs
    pygame.draw.rect(surface, DARK, (x, y3, square_size, square_size), 3)

    sensor_origin = (x + square_size // 2, y3 + int(square_size * 0.75))
    display_length = int(square_size * 0.45)

    for i, sensor_angle in enumerate(SENSOR_ANGLES):
        display_angle = math.radians(-90 + sensor_angle)
        ox, oy = sensor_origin

        for d, value in enumerate(sensor_data[i]):
            ratio = d / SENSOR_LENGTH
            px = int(ox + math.cos(display_angle) * display_length * ratio)
            py = int(oy + math.sin(display_angle) * display_length * ratio)

            if value == 0:
                c = GREEN
            elif value == 1:
                c = RED
            else:
                c = ORANGE

            pygame.draw.circle(surface, c, (px, py), 3)
    # --- Texte infos IA ---
    text_y = y3 + square_size + 20

    draw_text(surface, f"Reward: {reward:.2f}", x, text_y)
    draw_text(surface, f"Epsilon: {epsilon:.3f}", x, text_y + 25)
    draw_text(surface, f"Memory: {memory_size}", x, text_y + 50)

    network_y = int(HEIGHT * 0.62)
    network_h = HEIGHT - network_y - 20

    draw_network_view(
        surface,
        brain,
        sensor_data,
        panel_x + 10,
        network_y,
        PANEL_WIDTH - 20,
        network_h
    )

auto_mode = False
brain = CarBrain()
load_brain(brain)

optimizer = optim.Adam(brain.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# ----------------------------
# BOUCLE PRINCIPALE
# ----------------------------
running = True
auto_mode = False

while running:
    clock.tick(FPS)

    # ----------------------------
    # EVENTS
    # ----------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                auto_mode = not auto_mode
                print("Mode AUTO" if auto_mode else "Mode MANUEL")

            if event.key == pygame.K_s:
                save_brain(brain)

    # ----------------------------
    # DRAW TRACK
    # ----------------------------
    draw_track(screen)

    # ----------------------------
    # ETAT ACTUEL
    # ----------------------------
    sensor_points = get_sensor_points(car_x, car_y, car_angle)
    sensor_data, danger_ahead = analyse_sensors(screen, sensor_points)
    collision = get_car_collision(screen, car_x, car_y)

    state = sensor_data_to_tensor(sensor_data)

    # ----------------------------
    # ACTION
    # ----------------------------
    if auto_mode:
        action = choose_action(sensor_data)
        car_angle = apply_ai_action(car_angle, action)

        car_speed = 2
        rad = math.radians(car_angle)
        car_x += math.cos(rad) * car_speed
        car_y += math.sin(rad) * car_speed

    else:
        action = 2  # tout droit par défaut
        car_x, car_y, car_angle, car_speed = update_car(
            car_x, car_y, car_angle, car_speed
        )

    # ----------------------------
    # NOUVEL ETAT
    # ----------------------------
    next_sensor_points = get_sensor_points(car_x, car_y, car_angle)
    next_sensor_data, danger_ahead = analyse_sensors(screen, next_sensor_points)
    collision = get_car_collision(screen, car_x, car_y)

    reward = compute_reward(collision, next_sensor_data, car_speed, action)
    #reward = compute_reward(collision, next_sensor_data, car_speed)
    next_state = sensor_data_to_tensor(next_sensor_data)
    done = collision

    # ----------------------------
    # APPRENTISSAGE
    # ----------------------------
    if auto_mode:
        memory.append((state, action, reward, next_state, done))
        train_brain()

    # ----------------------------
    # RESET SI COLLISION
    # ----------------------------
    if collision:
        safety_state = "danger"

        pygame.display.flip()
        pygame.time.delay(200)

        car_x = START_X
        car_y = START_Y
        car_angle = START_ANGLE
        car_speed = 0.0

    elif danger_ahead:
        safety_state = "warning"
    else:
        safety_state = "safe"

    # ----------------------------
    # AFFICHAGE
    # ----------------------------
    draw_sensors(screen, sensor_points)
    draw_car(screen, car_x, car_y, car_angle)

    draw_panel(
        screen,
        car_angle,
        safety_state,
        sensor_data,
        reward,
        epsilon,
        len(memory)
    )

    pygame.display.flip()

# ----------------------------
# FIN
# ----------------------------
save_brain(brain)
pygame.quit()
