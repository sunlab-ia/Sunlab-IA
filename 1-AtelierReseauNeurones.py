import math
import random
import sys
from dataclasses import dataclass

import pygame

# ============================================================
# Mini réseau de neurones pédagogique : couleurs -> classes
# Version progressive avec :
#   1) apprentissage pas-à-pas
#   2) tests experts avec couleurs composées
#   3) bascule vers un réseau expert : 3 entrées -> 5 neurones cachés -> 5 sorties
# ============================================================
#
# Installation :
#     pip install pygame
#
# Lancement :
#     python neural_demo_pedagogique_expert.py
#

WIDTH, HEIGHT = 1500, 930
FPS = 60

# Couleurs interface
BG = (246, 248, 252)
PANEL = (255, 255, 255)
BORDER = (218, 226, 235)
TEXT = (25, 38, 59)
MUTED = (100, 116, 139)
GREEN = (34, 197, 94)
GREEN_LIGHT = (235, 252, 240)
BLUE = (37, 99, 235)
BLUE_LIGHT = (239, 246, 255)
PURPLE = (147, 92, 246)
PURPLE_LIGHT = (238, 228, 255)
ORANGE_UI = (249, 115, 22)
ORANGE_LINE = (255, 154, 67)
DARK = (15, 23, 42)
LIGHT_GRAY = (226, 232, 240)
PINK_LIGHT = (253, 242, 248)

YELLOW = (255, 212, 0)
RED = (255, 51, 51)
COLOR_BLUE = (37, 119, 229)
COLOR_ORANGE = (255, 140, 0)
COLOR_GREEN = (20, 170, 80)
COLOR_VIOLET = (150, 70, 220)

BASE_COLORS = [
    {
        "name": "JAUNE",
        "label": "Jaune",
        "code": [1, 0, 0],
        "color": YELLOW,
        "target": [1, 0, 0],
    },
    {
        "name": "ROUGE",
        "label": "Rouge",
        "code": [0, 1, 0],
        "color": RED,
        "target": [0, 1, 0],
    },
    {
        "name": "BLEU",
        "label": "Bleu",
        "code": [0, 0, 1],
        "color": COLOR_BLUE,
        "target": [0, 0, 1],
    },
]

EXPERT_TEST_COLORS = [
    {
        "name": "ORANGE",
        "label": "Orange",
        "code": [1, 1, 0],
        "color": COLOR_ORANGE,
        "comment": "mélange de JAUNE et ROUGE",
    },
    {
        "name": "VERT",
        "label": "Vert",
        "code": [1, 0, 1],
        "color": COLOR_GREEN,
        "comment": "mélange de JAUNE et BLEU",
    },
    {
        "name": "VIOLET",
        "label": "Violet",
        "code": [0, 1, 1],
        "color": COLOR_VIOLET,
        "comment": "mélange de ROUGE et BLEU",
    },
]

ADVANCED_COLORS = [
    {
        "name": "JAUNE",
        "label": "Jaune",
        "code": [1, 0, 0],
        "color": YELLOW,
        "target": [1, 0, 0, 0, 0],
    },
    {
        "name": "ORANGE",
        "label": "Orange",
        "code": [1, 1, 0],
        "color": COLOR_ORANGE,
        "target": [0, 1, 0, 0, 0],
    },
    {
        "name": "ROUGE",
        "label": "Rouge",
        "code": [0, 1, 0],
        "color": RED,
        "target": [0, 0, 1, 0, 0],
    },
    {
        "name": "VIOLET",
        "label": "Violet",
        "code": [0, 1, 1],
        "color": COLOR_VIOLET,
        "target": [0, 0, 0, 1, 0],
    },
    {
        "name": "BLEU",
        "label": "Bleu",
        "code": [0, 0, 1],
        "color": COLOR_BLUE,
        "target": [0, 0, 0, 0, 1],
    },
]


# ------------------------- Math réseau -------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [v / s for v in exps]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class SimpleNetwork:
    """
    Réseau pédagogique minimal :
        3 entrées -> 3 sorties

    C'est la version de base. Elle permet de voir les poids directement.
    """

    def __init__(self):
        self.input_size = 3
        self.output_size = 3
        self.w = [[0.50 for _ in range(3)] for _ in range(3)]
        self.b = [0.0 for _ in range(3)]

    def forward(self, x):
        raw = [dot(row, x) + self.b[o] for o, row in enumerate(self.w)]
        probs = softmax(raw)
        return raw, probs

    def train_one(self, x, target, lr=0.35):
        _, probs = self.forward(x)
        delta = [probs[i] - target[i] for i in range(3)]

        for o in range(3):
            for i in range(3):
                self.w[o][i] -= lr * delta[o] * x[i]
            self.b[o] -= lr * delta[o]

        return probs, delta


class ExpertNetwork:
    """
    Réseau expert stabilisé :
        3 entrées -> 5 neurones cachés -> 5 sorties

    Dans la version précédente, le réseau pouvait rester bloqué avec une sortie
    dominante, par exemple VIOLET, car 30 exemples et une seule passe étaient
    insuffisants pour stabiliser les poids.

    Ici, on conserve l'affichage pédagogique 3 -> 5 -> 5, mais on utilise :
    - une initialisation aléatoire centrée autour de 0 ;
    - une couche cachée suffisamment différenciée ;
    - quelques micro-itérations à chaque exemple pédagogique.
    """

    def __init__(self):
        self.input_size = 3
        self.hidden_size = 5
        self.output_size = 5

        random.seed(12)

        # Initialisation classique : petits poids autour de zéro.
        # Cela évite qu'une sortie prenne artificiellement l'avantage.
        self.w1 = [[random.uniform(-0.8, 0.8) for _ in range(3)] for _ in range(5)]
        self.b1 = [random.uniform(-0.1, 0.1) for _ in range(5)]

        self.w2 = [[random.uniform(-0.8, 0.8) for _ in range(5)] for _ in range(5)]
        self.b2 = [random.uniform(-0.1, 0.1) for _ in range(5)]

    def forward(self, x):
        hidden = [sigmoid(dot(row, x) + self.b1[h]) for h, row in enumerate(self.w1)]
        raw = [dot(row, hidden) + self.b2[o] for o, row in enumerate(self.w2)]
        probs = softmax(raw)
        return hidden, raw, probs

    def train_one(self, x, target, lr=0.18):
        hidden, raw, probs = self.forward(x)

        delta2 = [probs[i] - target[i] for i in range(5)]

        delta_hidden = []
        for h in range(5):
            signal = sum(delta2[o] * self.w2[o][h] for o in range(5))
            delta_hidden.append(signal * hidden[h] * (1 - hidden[h]))

        for o in range(5):
            for h in range(5):
                self.w2[o][h] -= lr * delta2[o] * hidden[h]
            self.b2[o] -= lr * delta2[o]

        for h in range(5):
            for i in range(3):
                self.w1[h][i] -= lr * delta_hidden[h] * x[i]
            self.b1[h] -= lr * delta_hidden[h]

        return probs, delta2

    def train_micro_batch(self, examples, repeats=8, lr=0.18):
        """
        Plusieurs micro-itérations internes.

        Pour l'utilisateur, un clic reste un exemple pédagogique.
        Mais mathématiquement, le réseau a besoin de plusieurs petites corrections
        pour bien séparer les cinq classes.
        """
        for _ in range(repeats):
            ex = random.choice(examples)
            self.train_one(ex["code"], ex["target"], lr=lr)


# ------------------------- UI helpers -------------------------

@dataclass
class Button:
    rect: pygame.Rect
    text: str
    kind: str = "light"

    def draw(self, screen, fonts):
        mouse = pygame.mouse.get_pos()
        hover = self.rect.collidepoint(mouse)

        if self.kind == "dark":
            bg = DARK
            fg = (255, 255, 255)
            border = DARK
        elif self.kind == "blue":
            bg = BLUE
            fg = (255, 255, 255)
            border = BLUE
        elif self.kind == "orange":
            bg = ORANGE_UI
            fg = (255, 255, 255)
            border = ORANGE_UI
        elif self.kind == "purple":
            bg = PURPLE
            fg = (255, 255, 255)
            border = PURPLE
        else:
            bg = (255, 255, 255)
            fg = TEXT
            border = BORDER

        if hover:
            bg = tuple(max(0, c - 8) for c in bg)

        round_rect(screen, self.rect, bg, 10, border)
        draw_centered_text(screen, self.text, fonts["button"], fg, self.rect.center)


def round_rect(screen, rect, color, radius=12, border=None, width=1):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(screen, border, rect, width, border_radius=radius)


def draw_text(screen, text, font, color, x, y):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))
    return surf.get_rect(topleft=(x, y))


def draw_centered_text(screen, text, font, color, center):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=center)
    screen.blit(surf, rect)
    return rect


def draw_right_text(screen, text, font, color, x, y):
    surf = font.render(text, True, color)
    rect = surf.get_rect(topright=(x, y))
    screen.blit(surf, rect)
    return rect


def draw_progress(screen, rect, value, color=GREEN):
    value = max(0, min(1, value))
    round_rect(screen, rect, LIGHT_GRAY, 999)
    inner = pygame.Rect(rect.x, rect.y, int(rect.w * value), rect.h)
    if inner.w > 0:
        round_rect(screen, inner, color, 999)


def draw_arrow_line(screen, color, start, end, width=1):
    width = int(round(width))
    width = max(1, width)

    start = (int(round(start[0])), int(round(start[1])))
    end = (int(round(end[0])), int(round(end[1])))

    pygame.draw.line(screen, color, start, end, width)

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 8
    spread = 0.45

    p1 = (
        end[0] - length * math.cos(angle - spread),
        end[1] - length * math.sin(angle - spread),
    )
    p2 = (
        end[0] - length * math.cos(angle + spread),
        end[1] - length * math.sin(angle + spread),
    )

    pygame.draw.polygon(screen, color, [end, p1, p2])


def draw_multiline(screen, text, font, color, x, y, max_width, line_gap=5):
    words = text.split()
    line = ""
    yy = y

    for word in words:
        test = line + word + " "
        if font.size(test)[0] > max_width and line:
            draw_text(screen, line, font, color, x, yy)
            yy += font.get_height() + line_gap
            line = word + " "
        else:
            line = test

    if line:
        draw_text(screen, line, font, color, x, yy)

    return yy


# ------------------------- Application -------------------------

class NeuralDemo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Réseau de neurones - démonstration pédagogique expert")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.fonts = {
            "title": pygame.font.SysFont("arial", 24, bold=True),
            "h2": pygame.font.SysFont("arial", 16, bold=True),
            "h3": pygame.font.SysFont("arial", 14, bold=True),
            "normal": pygame.font.SysFont("arial", 13),
            "small": pygame.font.SysFont("arial", 11),
            "button": pygame.font.SysFont("arial", 13, bold=True),
            "big": pygame.font.SysFont("arial", 22, bold=True),
        }

        self.mode = "BASIC"
        self.colors = BASE_COLORS
        self.outputs = ["JAUNE", "ROUGE", "BLEU"]
        self.training_set = [BASE_COLORS[i % 3] for i in range(21)]
        self.net = SimpleNetwork()

        self.phase = "INITIALISATION"
        self.step = 0
        self.current = self.colors[0]
        self.show_matrix = False
        self.buttons = {}
        self.expert_test_count = 0
        self.last_message = "Clique sur « Exemple suivant » pour observer les poids étape par étape."

    # ---------- États ----------

    def reset_basic(self):
        self.mode = "BASIC"
        self.colors = BASE_COLORS
        self.outputs = ["JAUNE", "ROUGE", "BLEU"]
        self.training_set = [BASE_COLORS[i % 3] for i in range(21)]
        self.net = SimpleNetwork()
        self.phase = "INITIALISATION"
        self.step = 0
        self.current = self.colors[0]
        self.show_matrix = False
        self.expert_test_count = 0
        self.last_message = "Réseau réinitialisé. Les poids repartent à 0.50."

    def switch_to_advanced(self):
        self.mode = "ADVANCED"
        self.colors = ADVANCED_COLORS
        self.outputs = ["JAUNE", "ORANGE", "ROUGE", "VIOLET", "BLEU"]
        self.training_set = [ADVANCED_COLORS[i % 5] for i in range(30)]
        self.net = ExpertNetwork()
        self.phase = "INITIALISATION"
        self.step = 0
        self.current = self.colors[0]
        self.show_matrix = False
        self.last_message = "Mode expert activé : le réseau va apprendre cinq mots de couleur, dont ORANGE et VIOLET."

    def learn_next(self):
        if self.phase == "TEST":
            self.last_message = "L'apprentissage est terminé. Réinitialise ou passe au mode expert."
            return

        if self.step >= len(self.training_set):
            self.phase = "TEST"
            self.current = random.choice(self.colors)
            self.last_message = "Apprentissage terminé. Tu peux lancer les tests."
            return

        self.phase = "APPRENTISSAGE"
        ex = self.training_set[self.step]
        self.current = ex

        if self.mode == "ADVANCED":
            # On apprend d'abord l'exemple visible.
            self.net.train_one(ex["code"], ex["target"], lr=0.18)

            # Puis on ajoute quelques corrections internes sur l'ensemble expert.
            # Cela évite que le réseau privilégie toujours la même sortie.
            self.net.train_micro_batch(self.training_set, repeats=12, lr=0.18)
        else:
            self.net.train_one(ex["code"], ex["target"])

        self.step += 1

        self.last_message = f"Exemple {self.step} appris : {ex['name']}. Observe les poids qui viennent de changer."

        if self.step >= len(self.training_set):
            self.phase = "TEST"
            self.current = random.choice(self.colors)
            self.last_message = "Apprentissage terminé. Le réseau peut maintenant être testé."

    def learn_all(self):
        while self.step < len(self.training_set):
            self.learn_next()

        if self.mode == "ADVANCED":
            # Consolidation finale : invisible pédagogiquement, mais utile pour
            # obtenir une démonstration fiable.
            for _ in range(250):
                ex = random.choice(self.training_set)
                self.net.train_one(ex["code"], ex["target"], lr=0.12)

        self.phase = "TEST"
        self.current = random.choice(self.colors)
        self.last_message = "Apprentissage complet effectué. Tu peux tester le réseau."

    def new_test_color(self):
        self.phase = "TEST"
        self.current = random.choice(self.colors)
        self.last_message = f"Test classique : {self.current['name']}."

    def expert_test(self):
        if self.mode == "BASIC":
            self.phase = "TEST_EXPERT"
            self.current = random.choice(EXPERT_TEST_COLORS)
            self.expert_test_count += 1
            self.last_message = (
                f"Test expert {self.expert_test_count}/10 : {self.current['name']} "
                f"({self.current['comment']}). Le réseau n'a pas appris cette classe."
            )
        else:
            self.phase = "TEST"
            self.current = random.choice(self.colors)
            self.last_message = f"En mode expert, {self.current['name']} est maintenant une vraie classe apprise."

    # ---------- Événements ----------

    def handle_click(self, pos):
        for key, button in list(self.buttons.items()):
            if button.rect.collidepoint(pos):
                if key == "reset":
                    self.reset_basic()
                elif key == "learn_next":
                    self.learn_next()
                elif key == "learn_all":
                    self.learn_all()
                elif key == "matrix":
                    self.show_matrix = not self.show_matrix
                elif key == "test":
                    self.new_test_color()
                elif key == "expert_test":
                    self.expert_test()
                elif key == "advanced":
                    self.switch_to_advanced()
                elif key == "new_color":
                    self.new_test_color()
                return

    # ---------- Calculs utiles ----------

    def current_probs(self):
        if self.mode == "BASIC":
            _, probs = self.net.forward(self.current["code"])
            return probs

        _, _, probs = self.net.forward(self.current["code"])
        return probs

    def current_gap(self):
        if "target" not in self.current:
            return None

        probs = self.current_probs()
        return sum(abs(self.current["target"][i] - probs[i]) for i in range(len(probs))) / len(probs)

    # ---------- Dessin ----------

    def draw_header(self):
        screen = self.screen
        f = self.fonts

        draw_text(screen, "RÉSEAU DE NEURONES – DÉMONSTRATION PÉDAGOGIQUE", f["title"], TEXT, 24, 20)

        if self.mode == "BASIC":
            subtitle = "Mode simple : 3 entrées codées → 3 sorties. Les couleurs composées ne sont pas encore des classes."
        else:
            subtitle = "Mode expert : 3 entrées codées → 5 neurones cachés → 5 sorties, avec couleurs composées apprises."

        draw_text(screen, subtitle, f["normal"], MUTED, 24, 52)

        x = 750
        gap = self.current_gap()
        gap_text = "—" if gap is None else f"{gap:.2f}"

        labels = [
            ("Mode", self.mode),
            ("Phase", self.phase),
            ("Étape", f"{self.step} / {len(self.training_set)}"),
            ("Écart", gap_text),
        ]

        for title, value in labels:
            draw_centered_text(screen, title, f["normal"], TEXT, (x + 55, 20))
            tag = pygame.Rect(x + 5, 42, 100, 28)
            round_rect(screen, tag, GREEN_LIGHT, 7, (188, 235, 205))
            draw_centered_text(screen, value, f["button"], (22, 163, 74), tag.center)
            x += 112

        self.buttons["reset"] = Button(pygame.Rect(1285, 24, 190, 40), "↻ Réinitialiser simple", "dark")
        self.buttons["reset"].draw(screen, f)

    def draw_stepper(self):
        screen = self.screen
        f = self.fonts

        rect = pygame.Rect(36, 88, 1428, 60)
        round_rect(screen, rect, PANEL, 10, BORDER)

        boxes = [
            ("1", "INITIALISATION", "Poids de départ", 55),
            ("2", "APPRENTISSAGE", "Un clic = un exemple", 390),
            ("3", "TEST", "Couleurs apprises", 735),
            ("4", "TEST_EXPERT", "Couleurs composées", 1045),
        ]

        for num, title, sub, x in boxes:
            if title == self.phase:
                round_rect(screen, pygame.Rect(x - 12, 89, 300, 58), GREEN_LIGHT, 10)

            pygame.draw.circle(screen, GREEN, (x + 35, 118), 16)
            draw_centered_text(screen, num, f["button"], (255, 255, 255), (x + 35, 118))
            draw_text(screen, title, f["h3"], TEXT, x + 60, 104)
            draw_text(screen, sub, f["small"], TEXT, x + 60, 126)

    def draw_training_panel(self):
        screen = self.screen
        f = self.fonts

        rect = pygame.Rect(36, 170, 300, 570)
        round_rect(screen, rect, PANEL, 12, BORDER)

        title = "JEU D'APPRENTISSAGE"
        subtitle = f"({len(self.training_set)} EXEMPLES)"

        draw_text(screen, title, f["h2"], TEXT, 50, 186)
        draw_text(screen, subtitle, f["normal"], TEXT, 210, 188)

        y = 218
        line_height = 18 if len(self.training_set) > 25 else 25

        for i, ex in enumerate(self.training_set):
            if y > 715:
                break

            row_rect = pygame.Rect(48, y - 2, 260, line_height)
            if self.phase == "APPRENTISSAGE" and i == self.step - 1:
                round_rect(screen, row_rect, BLUE_LIGHT, 6)

            draw_text(screen, str(i + 1), f["small"], MUTED, 56, y)
            pygame.draw.rect(screen, ex["color"], pygame.Rect(88, y - 1, 22, 15), border_radius=3)
            draw_text(screen, "→", f["small"], MUTED, 125, y)
            draw_text(screen, ex["name"], f["small"], TEXT, 150, y)

            y += line_height

        pbox = pygame.Rect(36, 755, 300, 105)
        round_rect(screen, pbox, PANEL, 12, BORDER)

        draw_text(screen, "Progression apprentissage", f["small"], MUTED, 54, 776)
        draw_right_text(screen, f"{self.step} / {len(self.training_set)}", f["button"], (22, 101, 52), 315, 775)
        draw_progress(screen, pygame.Rect(54, 810, 265, 13), self.step / len(self.training_set))

        if self.phase in ["TEST", "TEST_EXPERT"]:
            draw_centered_text(screen, "✓ Apprentissage terminé", f["normal"], (22, 163, 74), (185, 848))
        else:
            draw_centered_text(screen, "Clique pour apprendre pas à pas", f["small"], MUTED, (185, 848))

    def draw_basic_network(self):
        screen = self.screen
        f = self.fonts

        raw, probs = self.net.forward(self.current["code"])

        input_nodes = [(480, 340), (480, 500), (480, 660)]
        output_nodes = [(990, 340), (990, 500), (990, 660)]

        draw_centered_text(screen, "ENTRÉES", f["h2"], BLUE, (480, 260))
        draw_centered_text(screen, "SORTIES APPRISES", f["h2"], ORANGE_UI, (990, 260))

        for i, a in enumerate(input_nodes):
            for o, b in enumerate(output_nodes):
                w = self.net.w[o][i]
                width = max(1, min(8, abs(w) * 3))
                line_color = ORANGE_LINE if w >= 0.5 else (180, 190, 205)

                draw_arrow_line(screen, line_color, (a[0] + 30, a[1]), (b[0] - 38, b[1]), width)

                mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
                draw_centered_text(screen, f"{w:.2f}", f["small"], MUTED, (mx, my - 8))

        self.draw_input_nodes(input_nodes)
        self.draw_output_nodes(output_nodes, probs)

    def draw_advanced_network(self):
        screen = self.screen
        f = self.fonts

        hidden, raw, probs = self.net.forward(self.current["code"])

        input_nodes = [(455, 310), (455, 460), (455, 610)]
        hidden_nodes = [(735, 275), (735, 375), (735, 475), (735, 575), (735, 675)]
        output_nodes = [(1030, 275), (1030, 375), (1030, 475), (1030, 575), (1030, 675)]

        draw_centered_text(screen, "ENTRÉES", f["h2"], BLUE, (455, 240))
        draw_centered_text(screen, "5 NEURONES CACHÉS", f["h2"], PURPLE, (735, 240))
        draw_centered_text(screen, "5 SORTIES", f["h2"], ORANGE_UI, (1030, 240))

        for i, a in enumerate(input_nodes):
            for h, b in enumerate(hidden_nodes):
                w = self.net.w1[h][i]
                width = max(1, min(5, abs(w) * 2.5))
                draw_arrow_line(screen, (155, 170, 210), (a[0] + 25, a[1]), (b[0] - 35, b[1]), width)

        for h, a in enumerate(hidden_nodes):
            for o, b in enumerate(output_nodes):
                w = self.net.w2[o][h]
                width = max(1, min(5, abs(w) * 2.5))
                draw_arrow_line(screen, ORANGE_LINE, (a[0] + 35, a[1]), (b[0] - 35, b[1]), width)

        self.draw_input_nodes(input_nodes)

        for i, (x, y) in enumerate(hidden_nodes):
            pygame.draw.circle(screen, PURPLE_LIGHT, (x, y), 30)
            pygame.draw.circle(screen, PURPLE, (x, y), 30, 2)
            draw_centered_text(screen, f"{hidden[i]:.2f}", f["button"], TEXT, (x, y))
            draw_centered_text(screen, f"h{i + 1}", f["small"], TEXT, (x, y + 42))

        self.draw_output_nodes(output_nodes, probs)

    def draw_input_nodes(self, input_nodes):
        screen = self.screen
        f = self.fonts

        input_labels = [
            ("Jaune", [1, 0, 0], YELLOW),
            ("Rouge", [0, 1, 0], RED),
            ("Bleu", [0, 0, 1], COLOR_BLUE),
        ]

        for i, (label, code, color) in enumerate(input_labels):
            x, y = input_nodes[i]
            active = self.current["code"][i] == 1
            node_radius = 30 if active else 22
            border_width = 4 if active else 1

            draw_right_text(screen, label, f["normal"], TEXT, x - 45, y - 18)
            draw_right_text(screen, f"entrée {self.current['code'][i]}", f["small"], MUTED, x - 45, y + 5)

            pygame.draw.circle(screen, color, (x, y), node_radius)
            pygame.draw.circle(screen, DARK if active else (90, 100, 115), (x, y), node_radius, border_width)

            draw_centered_text(screen, str(self.current["code"][i]), f["button"], DARK if active else MUTED, (x, y + 44))

    def draw_output_nodes(self, output_nodes, probs):
        screen = self.screen
        f = self.fonts

        pred_i = probs.index(max(probs))

        for i, (x, y) in enumerate(output_nodes):
            color = self.colors[i]["color"] if i < len(self.colors) else LIGHT_GRAY
            is_pred = i == pred_i
            radius = 36 if is_pred else 30
            border_width = 4 if is_pred else 2

            pygame.draw.circle(screen, (255, 255, 255), (x, y), radius)
            pygame.draw.circle(screen, color, (x, y), radius, border_width)

            draw_centered_text(screen, f"{probs[i]:.2f}", f["button"], TEXT, (x, y))
            draw_text(screen, self.outputs[i], f["h2"], TEXT, x + 52, y - 8)

            if is_pred:
                draw_text(screen, "← plus forte", f["small"], (22, 163, 74), x + 52, y + 14)

    def draw_network_panel(self):
        screen = self.screen
        f = self.fonts

        rect = pygame.Rect(350, 170, 790, 705)
        round_rect(screen, rect, PANEL, 12, BORDER)

        if self.mode == "BASIC":
            draw_text(screen, "RÉSEAU SIMPLE : 3 ENTRÉES → 3 SORTIES", f["h2"], TEXT, 365, 186)
            draw_text(screen, "Un clic apprend un exemple : observe les poids juste après chaque correction.", f["normal"], MUTED, 365, 210)
            self.draw_basic_network()
        else:
            draw_text(screen, "RÉSEAU EXPERT : 3 ENTRÉES → 5 NEURONES CACHÉS → 5 SORTIES", f["h2"], TEXT, 365, 186)
            draw_text(screen, "Les couleurs composées deviennent des classes apprises. Le réseau a besoin de plus d'entraînement interne.", f["normal"], MUTED, 365, 210)
            self.draw_advanced_network()

        # Message pédagogique
        explain = pygame.Rect(455, 760, 580, 80)
        bg = PINK_LIGHT if self.phase == "TEST_EXPERT" else BLUE_LIGHT
        border = (245, 190, 220) if self.phase == "TEST_EXPERT" else (191, 219, 254)
        round_rect(screen, explain, bg, 10, border)

        draw_text(screen, "À observer", f["h3"], BLUE if self.phase != "TEST_EXPERT" else PURPLE, 475, 775)
        draw_multiline(screen, self.last_message, f["small"], TEXT, 475, 800, 535, 4)

        # Boutons bas
        self.buttons["learn_next"] = Button(pygame.Rect(350, 880, 230, 38), "Apprendre l'exemple suivant", "blue")
        self.buttons["learn_all"] = Button(pygame.Rect(590, 880, 150, 38), "Tout apprendre", "light")
        self.buttons["matrix"] = Button(pygame.Rect(750, 880, 160, 38), "Voir poids", "light")
        self.buttons["test"] = Button(pygame.Rect(920, 880, 105, 38), "Test", "light")

        self.buttons["learn_next"].draw(screen, f)
        self.buttons["learn_all"].draw(screen, f)
        self.buttons["matrix"].draw(screen, f)
        self.buttons["test"].draw(screen, f)

        if self.mode == "BASIC" and self.phase in ["TEST", "TEST_EXPERT"]:
            self.buttons["expert_test"] = Button(pygame.Rect(1035, 880, 105, 38), "Test Expert", "orange")
            self.buttons["expert_test"].draw(screen, f)

        if self.mode == "BASIC" and self.expert_test_count >= 10:
            self.buttons["advanced"] = Button(pygame.Rect(795, 832, 300, 34), "Basculer vers réseau expert 3→5→5", "purple")
            self.buttons["advanced"].draw(screen, f)

        if self.show_matrix:
            if self.mode == "BASIC":
                self.draw_basic_matrix()
            else:
                self.draw_advanced_matrix()

    def draw_basic_matrix(self):
        screen = self.screen
        f = self.fonts

        box = pygame.Rect(430, 555, 620, 270)
        round_rect(screen, box, DARK, 12)

        draw_text(screen, "Matrice des poids", f["h2"], (255, 255, 255), 455, 577)
        draw_text(screen, "Ligne = sortie ; colonne = entrée", f["small"], (219, 234, 254), 455, 603)

        headers = ["entrée JAUNE", "entrée ROUGE", "entrée BLEU"]
        x0 = 570

        for c, h in enumerate(headers):
            draw_centered_text(screen, h, f["small"], (219, 234, 254), (x0 + c * 125, 635))

        y = 675
        for o, out in enumerate(self.outputs):
            draw_text(screen, out, f["small"], (219, 234, 254), 455, y)
            for i in range(3):
                val = self.net.w[o][i]
                draw_centered_text(screen, f"{val:.2f}", f["button"], (255, 255, 255), (x0 + i * 125, y + 6))
            y += 38

        draw_text(screen, "Biais :", f["small"], (219, 234, 254), 455, y + 14)
        for o in range(3):
            draw_centered_text(screen, f"{self.net.b[o]:.2f}", f["small"], (255, 255, 255), (x0 + o * 125, y + 20))

    def draw_advanced_matrix(self):
        screen = self.screen
        f = self.fonts

        box = pygame.Rect(395, 540, 690, 300)
        round_rect(screen, box, DARK, 12)

        draw_text(screen, "Aperçu des matrices du réseau expert", f["h2"], (255, 255, 255), 420, 562)
        draw_text(screen, "w1 : entrées → cachée ; w2 : cachée → sorties", f["small"], (219, 234, 254), 420, 588)

        y = 620
        draw_text(screen, "w1", f["h3"], (255, 255, 255), 420, y)
        y += 25

        for h in range(5):
            vals = "  ".join(f"{v:.2f}" for v in self.net.w1[h])
            draw_text(screen, f"h{h + 1}: [{vals}]", f["small"], (219, 234, 254), 420, y)
            y += 18

        y += 8
        draw_text(screen, "w2", f["h3"], (255, 255, 255), 420, y)
        y += 25

        for o, out in enumerate(self.outputs):
            vals = "  ".join(f"{v:.2f}" for v in self.net.w2[o])
            draw_text(screen, f"{out:<7}: [{vals}]", f["small"], (219, 234, 254), 420, y)
            y += 18

    def draw_test_panel(self):
        screen = self.screen
        f = self.fonts

        # En mode expert, il y a 5 sorties au lieu de 3.
        # On agrandit donc le panneau et on sépare clairement :
        #   1) couleur test
        #   2) probabilités
        #   3) décision finale
        panel_h = 620 if len(self.outputs) <= 3 else 620
        rect = pygame.Rect(1155, 170, 310, panel_h)
        round_rect(screen, rect, PANEL, 12, BORDER)

        draw_text(screen, "TEST : PRÉDICTION", f["h2"], TEXT, 1168, 186)

        # ---------- Bloc couleur test ----------
        draw_centered_text(screen, "Entrée test", f["h3"], TEXT, (1310, 228))

        square = pygame.Rect(1252, 248, 116, 116)
        round_rect(screen, square, self.current["color"], 16)

        draw_centered_text(screen, self.current["name"], f["h3"], TEXT, (1310, 382))
        draw_centered_text(screen, f"Code : {self.current['code']}", f["normal"], TEXT, (1310, 405))

        self.buttons["new_color"] = Button(pygame.Rect(1195, 428, 240, 36), "↻ Nouvelle couleur", "light")
        self.buttons["new_color"].draw(screen, f)

        # ---------- Bloc probabilités ----------
        prob_box = pygame.Rect(1170, 485, 280, 190 if len(self.outputs) <= 3 else 205)
        round_rect(screen, prob_box, (255, 255, 255), 10, BORDER)

        draw_text(screen, "PROBABILITÉS DE SORTIE", f["h2"], TEXT, 1185, 502)

        probs = self.current_probs()
        pred_i = probs.index(max(probs))

        if len(self.outputs) <= 3:
            y = 545
            row_gap = 43
            bar_x = 1245
            bar_w = 130
            value_x = 1400
        else:
            # Version compacte pour 5 sorties : aucune superposition avec la prédiction.
            y = 540
            row_gap = 31
            bar_x = 1242
            bar_w = 118
            value_x = 1390

        for i, out in enumerate(self.outputs):
            color = self.colors[i]["color"] if i < len(self.colors) else LIGHT_GRAY
            draw_text(screen, out, f["h3"], color, 1185, y - 2)
            draw_progress(screen, pygame.Rect(bar_x, y, bar_w, 12), probs[i], color)
            draw_text(screen, f"{probs[i]:.2f}", f["normal"], TEXT, value_x, y - 3)
            y += row_gap

        # ---------- Bloc décision finale ----------
        if len(self.outputs) <= 3:
            pred = pygame.Rect(1175, 700, 270, 72)
            title_center = (1260, 728)
            value_pos = (1330, 716)
            subtitle_center = (1310, 756)
        else:
            # En mode 5 sorties, la prédiction descend sous la boîte des probabilités.
            pred = pygame.Rect(1175, 704, 270, 72)
            title_center = (1260, 732)
            value_pos = (1330, 720)
            subtitle_center = (1310, 760)

        round_rect(screen, pred, GREEN_LIGHT, 10, (188, 235, 205))
        draw_centered_text(screen, "Prédiction :", f["h2"], (22, 163, 74), title_center)
        draw_text(screen, self.outputs[pred_i], f["big"], self.colors[pred_i]["color"], value_pos[0], value_pos[1])
        draw_centered_text(screen, "classe avec la probabilité la plus élevée", f["small"], MUTED, subtitle_center)

        # ---------- Aide pédagogique, séparée du bloc probabilités ----------
        help_rect = pygame.Rect(1155, 805, 310, 105)
        round_rect(screen, help_rect, BLUE_LIGHT, 10, (191, 219, 254))

        draw_text(screen, "💡 Message pédagogique", f["h3"], BLUE, 1175, 820)

        if self.mode == "BASIC" and self.phase == "TEST_EXPERT":
            msg = (
                "Cette couleur n'a jamais été apprise comme un mot séparé. "
                "Le réseau répartit donc sa confiance entre les sorties connues."
            )
        elif self.mode == "ADVANCED":
            msg = (
                "En mode expert, les couleurs composées font partie du vocabulaire appris. "
                "Les 5 probabilités sont affichées séparément pour éviter toute ambiguïté."
            )
        else:
            msg = (
                "Le réseau choisit la sortie dont la probabilité est la plus élevée."
            )

        draw_multiline(screen, msg, f["small"], TEXT, 1175, 846, 270, 4)

    def draw(self):
        # Supprime les anciens boutons à chaque frame.
        self.buttons = {}

        self.screen.fill(BG)
        self.draw_header()
        self.draw_stepper()
        self.draw_training_panel()
        self.draw_network_panel()
        self.draw_test_panel()
        pygame.display.flip()

    def run(self):
        while True:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.learn_next()
                    elif event.key == pygame.K_a:
                        self.learn_all()
                    elif event.key == pygame.K_r:
                        self.reset_basic()
                    elif event.key == pygame.K_t:
                        self.new_test_color()
                    elif event.key == pygame.K_e:
                        self.expert_test()
                    elif event.key == pygame.K_m:
                        self.show_matrix = not self.show_matrix

            self.draw()


if __name__ == "__main__":
    NeuralDemo().run()
