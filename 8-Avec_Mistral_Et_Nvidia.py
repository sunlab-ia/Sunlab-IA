import math
import sys
from dataclasses import dataclass

import pygame

# ============================================================
# Atelier_Local_LLM.py
# Démonstrateur pédagogique : faire tourner un LLM localement
#
# Objectif :
#   Montrer comment Mistral 7B, une carte graphique Nvidia,
#   un moteur d'inférence, la quantification, la VRAM, une interface
#   de chat et éventuellement un RAG interviennent dans le workflow.
#
# Installation :
#   pip install pygame
#
# Lancement :
#   python Atelier_Local_LLM.py
# ============================================================

WIDTH, HEIGHT = 1500, 930
FPS = 60

BG = (255, 250, 240)
PANEL = (255, 255, 255)
LIGHT_PANEL = (255, 253, 247)
TEXT = (10, 10, 10)
MUTED = (82, 72, 58)
BORDER = (240, 217, 159)

YELLOW = (255, 191, 0)
YELLOW_LIGHT = (255, 246, 214)
ORANGE = (255, 136, 23)
ORANGE_LIGHT = (255, 229, 204)
BLUE = (30, 111, 159)
BLUE_LIGHT = (232, 245, 252)
GREEN = (90, 157, 47)
GREEN_LIGHT = (238, 247, 230)
PURPLE = (113, 69, 168)
PURPLE_LIGHT = (241, 232, 250)
RED = (216, 59, 45)
RED_LIGHT = (255, 235, 232)
DARK = (35, 35, 35)
GRAY = (228, 224, 215)

PROMPT = "Explique simplement ce qu'est un séquenceur ADN."
ANSWER = "Un séquenceur ADN lit l'ordre des lettres A, C, G et T dans une molécule d'ADN."

PIPELINE = [
    "Utilisateur",
    "Interface chat",
    "Moteur d'inférence",
    "Mistral 7B",
    "GPU Nvidia",
    "Réponse",
]

STAGES = [
    {
        "short": "Vue",
        "title": "1. Vue d'ensemble",
        "subtitle": "Une IA locale minimale est un assemblage de briques.",
        "concept": "L'utilisateur ne recrée pas un LLM. Il assemble un modèle déjà entraîné, un moteur d'inférence, une interface et du matériel capable d'effectuer les calculs.",
        "remember": "Mistral 7B est le cerveau ; le GPU Nvidia est l'accélérateur ; le moteur d'inférence les fait travailler ensemble.",
    },
    {
        "short": "Modèle",
        "title": "2. Mistral 7B",
        "subtitle": "Le modèle pré-entraîné évite de tout inventer.",
        "concept": "Mistral 7B contient déjà les milliards de poids appris pendant l'entraînement : vocabulaire, embeddings, attention, blocs Transformer et capacités linguistiques.",
        "remember": "On ne réentraîne pas un LLM à la maison : on télécharge un modèle déjà entraîné.",
    },
    {
        "short": "Quantif.",
        "title": "3. Quantification",
        "subtitle": "Compresser le modèle pour le faire tenir sur un PC.",
        "concept": "Un modèle en pleine précision peut être trop volumineux. La quantification, par exemple en 4 bits ou 5 bits, réduit la mémoire nécessaire au prix d'une légère perte de précision.",
        "remember": "La quantification rend un gros modèle utilisable sur une machine locale.",
    },
    {
        "short": "Moteur",
        "title": "4. Moteur d'inférence",
        "subtitle": "Le logiciel qui charge le modèle et génère les tokens.",
        "concept": "Ollama, LM Studio ou llama.cpp appliquent le chat template, tokenisent le prompt, chargent le modèle quantifié, appellent le CPU/GPU et décodent les tokens générés.",
        "remember": "Le moteur d'inférence est le chef d'orchestre.",
    },
    {
        "short": "GPU",
        "title": "5. GPU Nvidia",
        "subtitle": "La carte graphique accélère les calculs matriciels.",
        "concept": "Le LLM manipule d'énormes matrices. Le GPU Nvidia, via CUDA, accélère les multiplications de matrices, l'attention et les couches MLP.",
        "remember": "Le GPU ne comprend pas le texte : il accélère les calculs.",
    },
    {
        "short": "VRAM",
        "title": "6. RAM et VRAM",
        "subtitle": "Le modèle doit tenir quelque part.",
        "concept": "La VRAM de la carte graphique accueille tout ou partie du modèle. Si elle est insuffisante, une partie reste en RAM système, ce qui peut ralentir fortement l'inférence.",
        "remember": "Plus il y a de VRAM, plus le modèle tourne confortablement.",
    },
    {
        "short": "Template",
        "title": "7. Chat template",
        "subtitle": "Le bon format de conversation compte.",
        "concept": "Chaque famille de modèles attend souvent un format de prompt particulier : balises système, utilisateur, assistant, séparateurs. Le moteur doit appliquer le bon template.",
        "remember": "Un bon modèle mal formaté peut répondre moins bien.",
    },
    {
        "short": "Boucle",
        "title": "8. Génération locale",
        "subtitle": "Le modèle prédit les tokens sur ton PC.",
        "concept": "Une fois chargé localement, le LLM prédit un token, l'ajoute au contexte, puis recommence. Les calculs se font sur la machine locale, sans envoyer le prompt à un serveur distant.",
        "remember": "L'autonomie vient du fait que l'inférence se déroule localement.",
    },
    {
        "short": "Interface",
        "title": "9. Interface utilisateur",
        "subtitle": "Une interface évite d'utiliser seulement la ligne de commande.",
        "concept": "LM Studio, Ollama avec Open WebUI, ou d'autres interfaces permettent d'utiliser le modèle comme un ChatGPT local.",
        "remember": "L'interface rend le système utilisable par un non-informaticien.",
    },
    {
        "short": "RAG",
        "title": "10. Documents locaux / RAG",
        "subtitle": "Brancher le LLM sur ses propres documents.",
        "concept": "Le RAG découpe des documents, calcule des embeddings, retrouve les passages pertinents, puis les injecte dans le prompt pour guider la réponse.",
        "remember": "Le RAG ne réentraîne pas le modèle : il lui fournit le bon contexte au bon moment.",
    },
    {
        "short": "Bilan",
        "title": "11. IA locale minimale",
        "subtitle": "Les briques nécessaires pour une première autonomie.",
        "concept": "Une configuration minimale combine un PC correct, un moteur d'inférence, un modèle instruct quantifié, assez de RAM/VRAM, une interface et éventuellement un RAG.",
        "remember": "Le but est d'assembler intelligemment, pas de reconstruire toute l'IA.",
    },
]

COMPONENTS = [
    ("Interface chat", "LM Studio / Open WebUI", BLUE_LIGHT, BLUE),
    ("Moteur", "Ollama / llama.cpp", GREEN_LIGHT, GREEN),
    ("Modèle", "Mistral 7B Instruct", PURPLE_LIGHT, PURPLE),
    ("Format", "GGUF Q4 / Q5", YELLOW_LIGHT, YELLOW),
    ("GPU", "Nvidia CUDA", ORANGE_LIGHT, ORANGE),
    ("Mémoire", "RAM + VRAM", RED_LIGHT, RED),
]

VRAM_BARS = [
    ("Mistral 7B Q4", 0.48, "≈ 4–6 Go"),
    ("Mistral 7B Q5", 0.60, "≈ 5–7 Go"),
    ("Contexte long", 0.72, "plus de cache"),
    ("Marge confortable", 0.85, "8–12 Go+"),
]

@dataclass
class Button:
    rect: pygame.Rect
    text: str
    kind: str = "light"

    def draw(self, screen, fonts):
        mouse = pygame.mouse.get_pos()
        hover = self.rect.collidepoint(mouse)

        if self.kind == "dark":
            bg, fg, border = DARK, (255, 255, 255), DARK
        elif self.kind == "orange":
            bg, fg, border = ORANGE, TEXT, DARK
        elif self.kind == "blue":
            bg, fg, border = BLUE, (255, 255, 255), BLUE
        elif self.kind == "green":
            bg, fg, border = GREEN, (255, 255, 255), GREEN
        elif self.kind == "purple":
            bg, fg, border = PURPLE, (255, 255, 255), PURPLE
        else:
            bg, fg, border = PANEL, TEXT, BORDER

        if hover:
            bg = tuple(max(0, c - 10) for c in bg)

        pygame.draw.rect(screen, bg, self.rect, border_radius=8)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=8)
        draw_centered_text(screen, self.text, fonts["button"], fg, self.rect.center)


def draw_text(screen, text, font, color, x, y):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))
    return surf.get_rect(topleft=(x, y))


def draw_centered_text(screen, text, font, color, center):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=center)
    screen.blit(surf, rect)
    return rect


def draw_multiline(screen, text, font, color, x, y, max_width, line_gap=5, max_lines=None):
    words = text.split(" ")
    line = ""
    yy = y
    lines = 0

    for word in words:
        test = line + word + " "
        if font.size(test)[0] > max_width and line:
            draw_text(screen, line.rstrip(), font, color, x, yy)
            yy += font.get_height() + line_gap
            lines += 1
            if max_lines is not None and lines >= max_lines:
                draw_text(screen, "…", font, color, x, yy)
                return yy
            line = word + " "
        else:
            line = test

    if line:
        draw_text(screen, line.rstrip(), font, color, x, yy)
    return yy


def draw_panel(screen, rect, title, fonts):
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    draw_text(screen, title, fonts["h2"], TEXT, rect.x + 18, rect.y + 16)


def draw_arrow(screen, start, end, color, width=3):
    width = max(1, int(width))
    pygame.draw.line(screen, color, start, end, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 12
    spread = 0.45
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    pygame.draw.polygon(screen, color, [end, p1, p2])


def draw_progress(screen, rect, value, color):
    pygame.draw.rect(screen, (238, 235, 226), rect, border_radius=999)
    inner = pygame.Rect(rect.x, rect.y, max(2, int(rect.w * value)), rect.h)
    pygame.draw.rect(screen, color, inner, border_radius=999)
    pygame.draw.rect(screen, BORDER, rect, 1, border_radius=999)


class LocalLLMDemo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Atelier IA locale — Mistral 7B + Nvidia")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.fonts = {
            "title": pygame.font.SysFont("arial", 27, bold=True),
            "h2": pygame.font.SysFont("arial", 18, bold=True),
            "h3": pygame.font.SysFont("arial", 15, bold=True),
            "normal": pygame.font.SysFont("arial", 14),
            "small": pygame.font.SysFont("arial", 12),
            "tiny": pygame.font.SysFont("arial", 10),
            "button": pygame.font.SysFont("arial", 13, bold=True),
            "big": pygame.font.SysFont("arial", 24, bold=True),
            "mono": pygame.font.SysFont("consolas", 15),
            "mono_big": pygame.font.SysFont("consolas", 20, bold=True),
        }

        self.stage = 0
        self.reveal = 0
        self.generated = 0
        self.gpu_enabled = True
        self.rag_enabled = False
        self.buttons = {}

    @property
    def info(self):
        return STAGES[self.stage]

    def canvas_rect(self):
        return pygame.Rect(470, 170, 995, 695)

    def next_stage(self):
        self.stage = (self.stage + 1) % len(STAGES)
        self.reveal = 0
        if self.stage != 7:
            self.generated = 0

    def prev_stage(self):
        self.stage = (self.stage - 1) % len(STAGES)
        self.reveal = 0

    def next_reveal(self):
        self.reveal = (self.reveal + 1) % 7

    def generate_next(self):
        self.stage = 7
        self.generated = min(8, self.generated + 1)

    def reset(self):
        self.stage = 0
        self.reveal = 0
        self.generated = 0
        self.gpu_enabled = True
        self.rag_enabled = False

    def handle_click(self, pos):
        for key, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                if key == "prev": self.prev_stage()
                elif key == "next": self.next_stage()
                elif key == "suite": self.next_reveal()
                elif key == "generate": self.generate_next()
                elif key == "gpu": self.gpu_enabled = not self.gpu_enabled
                elif key == "rag": self.rag_enabled = not self.rag_enabled
                elif key == "reset": self.reset()
                return

    def handle_key(self, event):
        if event.key in [pygame.K_RIGHT, pygame.K_SPACE]:
            self.next_stage()
        elif event.key == pygame.K_LEFT:
            self.prev_stage()
        elif event.key == pygame.K_s:
            self.next_reveal()
        elif event.key == pygame.K_g:
            self.generate_next()
        elif event.key == pygame.K_n:
            self.gpu_enabled = not self.gpu_enabled
        elif event.key == pygame.K_d:
            self.rag_enabled = not self.rag_enabled
        elif event.key == pygame.K_r:
            self.reset()
        elif event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

    def draw_header(self):
        f = self.fonts
        draw_text(self.screen, "IA LOCALE — MISTRAL 7B + NVIDIA + OUTILS D'INFÉRENCE", f["title"], TEXT, 34, 22)
        draw_text(
            self.screen,
            "Comprendre les briques concrètes nécessaires pour faire tourner un premier LLM local minimal.",
            f["normal"], MUTED, 34, 56
        )

        stats = [
            ("Modèle", "Mistral 7B"),
            ("Accélération", "GPU Nvidia" if self.gpu_enabled else "CPU seul"),
            ("Documents", "RAG ON" if self.rag_enabled else "RAG OFF"),
            ("Étape", f"{self.stage + 1} / {len(STAGES)}"),
        ]
        x = 790
        widths = [130, 150, 130, 110]
        for (label, value), w in zip(stats, widths):
            draw_centered_text(self.screen, label, f["small"], MUTED, (x + w // 2, 25))
            rect = pygame.Rect(x, 42, w, 30)
            pygame.draw.rect(self.screen, (255, 241, 198), rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)
            draw_centered_text(self.screen, value, f["button"], TEXT, rect.center)
            x += w + 14

    def draw_buttons(self):
        f = self.fonts
        y = 92
        self.buttons["prev"] = Button(pygame.Rect(34, y, 120, 42), "← Étape", "light")
        self.buttons["next"] = Button(pygame.Rect(164, y, 210, 42), "▶ Étape suivante", "orange")
        self.buttons["suite"] = Button(pygame.Rect(470, y, 115, 42), "SUITE", "green")
        self.buttons["generate"] = Button(pygame.Rect(598, y, 165, 42), "Générer token", "purple")
        self.buttons["gpu"] = Button(
            pygame.Rect(778, y, 135, 42),
            "GPU ON" if self.gpu_enabled else "GPU OFF",
            "blue" if self.gpu_enabled else "light"
        )
        self.buttons["rag"] = Button(
            pygame.Rect(928, y, 135, 42),
            "RAG ON" if self.rag_enabled else "RAG OFF",
            "blue" if self.rag_enabled else "light"
        )
        self.buttons["reset"] = Button(pygame.Rect(1320, y, 145, 42), "Réinitialiser", "dark")

        for key in ["prev", "next", "suite", "generate", "gpu", "rag", "reset"]:
            self.buttons[key].draw(self.screen, f)

    def draw_left_panel(self):
        f = self.fonts
        rect = pygame.Rect(34, 170, 410, 695)
        draw_panel(self.screen, rect, "Concept couvert", f)

        draw_text(self.screen, self.info["title"], f["big"], TEXT, 54, 214)
        draw_text(self.screen, self.info["subtitle"], f["h3"], BLUE, 54, 252)
        draw_multiline(self.screen, self.info["concept"], f["normal"], MUTED, 54, 292, 350, 5, max_lines=11)

        pygame.draw.rect(self.screen, LIGHT_PANEL, pygame.Rect(54, 510, 350, 180), border_radius=10)
        pygame.draw.rect(self.screen, BORDER, pygame.Rect(54, 510, 350, 180), 1, border_radius=10)
        draw_text(self.screen, "À retenir", f["h3"], TEXT, 72, 530)
        draw_multiline(self.screen, self.info["remember"], f["normal"], MUTED, 72, 562, 310, 5, max_lines=6)

        draw_text(self.screen, "À ne pas confondre", f["h3"], ORANGE, 54, 725)
        warning = "Le modèle contient les connaissances apprises ; le GPU accélère les calculs ; l'interface rend l'ensemble utilisable."
        draw_multiline(self.screen, warning, f["small"], MUTED, 54, 752, 340, 4, max_lines=5)

    def draw_bottom_pipeline(self):
        f = self.fonts
        labels = [s["short"] for s in STAGES]
        x = 34
        y = 880
        total_gap = 8
        w = int((WIDTH - 68 - total_gap * (len(labels) - 1)) / len(labels))

        for i, label in enumerate(labels):
            rect = pygame.Rect(x, y, w, 32)
            bg = ORANGE_LIGHT if i == self.stage else PANEL
            border = ORANGE if i == self.stage else BORDER
            pygame.draw.rect(self.screen, bg, rect, border_radius=999)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=999)
            draw_centered_text(self.screen, label, f["small"], TEXT, rect.center)
            x += w + total_gap

    def draw_canvas(self):
        f = self.fonts
        rect = self.canvas_rect()
        draw_panel(self.screen, rect, "Vue principale", f)

        area = pygame.Rect(rect.x + 35, rect.y + 65, rect.w - 70, rect.h - 105)
        pygame.draw.rect(self.screen, LIGHT_PANEL, area, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, area, 1, border_radius=10)

        if self.stage == 0:
            self.draw_overview(area)
        elif self.stage == 1:
            self.draw_mistral(area)
        elif self.stage == 2:
            self.draw_quantization(area)
        elif self.stage == 3:
            self.draw_engine(area)
        elif self.stage == 4:
            self.draw_gpu(area)
        elif self.stage == 5:
            self.draw_memory(area)
        elif self.stage == 6:
            self.draw_template(area)
        elif self.stage == 7:
            self.draw_generation(area)
        elif self.stage == 8:
            self.draw_interface(area)
        elif self.stage == 9:
            self.draw_rag(area)
        else:
            self.draw_minimal_setup(area)

        draw_text(
            self.screen,
            "Touches : Espace/→ étape suivante · ← étape précédente · S suite · G générer · N GPU · D RAG · R réinitialiser",
            f["small"], MUTED, rect.x + 40, rect.bottom - 30
        )

    def draw_component_box(self, rect, title, subtitle, bg, border):
        f = self.fonts
        pygame.draw.rect(self.screen, bg, rect, border_radius=14)
        pygame.draw.rect(self.screen, border, rect, 2, border_radius=14)
        draw_centered_text(self.screen, title, f["h2"], TEXT, (rect.centerx, rect.y + 28))
        draw_centered_text(self.screen, subtitle, f["small"], MUTED, (rect.centerx, rect.y + 56))

    def draw_overview(self, area):
        f = self.fonts
        draw_text(self.screen, "Architecture minimale d'une IA locale", f["h3"], ORANGE, area.x + 35, area.y + 32)

        y = area.y + 130
        x = area.x + 55
        box_w, box_h, gap = 138, 92, 22

        boxes = []
        for i, name in enumerate(PIPELINE):
            rect = pygame.Rect(x + i * (box_w + gap), y, box_w, box_h)
            boxes.append(rect)
            if name == "Mistral 7B":
                bg, border = PURPLE_LIGHT, PURPLE
                sub = "cerveau"
            elif name == "GPU Nvidia":
                bg, border = ORANGE_LIGHT, ORANGE
                sub = "calcul"
            elif name == "Moteur d'inférence":
                bg, border = GREEN_LIGHT, GREEN
                sub = "orchestration"
            elif name == "Interface chat":
                bg, border = BLUE_LIGHT, BLUE
                sub = "usage"
            else:
                bg, border = PANEL, BORDER
                sub = ""
            self.draw_component_box(rect, name, sub, bg, border)

            if i > 0:
                draw_arrow(self.screen, (boxes[i-1].right + 4, boxes[i-1].centery), (rect.x - 4, rect.centery), DARK, 3)

        prompt = pygame.Rect(area.x + 170, area.y + 310, area.w - 340, 78)
        pygame.draw.rect(self.screen, BLUE_LIGHT, prompt, border_radius=12)
        pygame.draw.rect(self.screen, BLUE, prompt, 2, border_radius=12)
        draw_text(self.screen, "Prompt local", f["h3"], BLUE, prompt.x + 18, prompt.y + 12)
        draw_text(self.screen, PROMPT, f["mono_big"], TEXT, prompt.x + 18, prompt.y + 42)

        answer = pygame.Rect(area.x + 170, area.y + 430, area.w - 340, 90)
        pygame.draw.rect(self.screen, GREEN_LIGHT, answer, border_radius=12)
        pygame.draw.rect(self.screen, GREEN, answer, 2, border_radius=12)
        draw_text(self.screen, "Réponse locale", f["h3"], GREEN, answer.x + 18, answer.y + 12)
        draw_multiline(self.screen, ANSWER, f["mono_big"], TEXT, answer.x + 18, answer.y + 42, answer.w - 36)

    def draw_mistral(self, area):
        f = self.fonts
        draw_text(self.screen, "Mistral 7B = modèle pré-entraîné", f["h3"], ORANGE, area.x + 35, area.y + 32)

        brain = pygame.Rect(area.x + 330, area.y + 95, 300, 220)
        pygame.draw.ellipse(self.screen, PURPLE_LIGHT, brain)
        pygame.draw.ellipse(self.screen, PURPLE, brain, 4)
        draw_centered_text(self.screen, "Mistral 7B", f["big"], TEXT, (brain.centerx, brain.centery - 18))
        draw_centered_text(self.screen, "≈ 7 milliards de paramètres", f["h3"], MUTED, (brain.centerx, brain.centery + 20))

        items = [
            ("Tokenisation compatible", BLUE_LIGHT, BLUE),
            ("Embeddings appris", GREEN_LIGHT, GREEN),
            ("Attention entraînée", ORANGE_LIGHT, ORANGE),
            ("Blocs Transformer", YELLOW_LIGHT, YELLOW),
            ("Poids du réseau", PURPLE_LIGHT, PURPLE),
            ("Style instruct", RED_LIGHT, RED),
        ]

        positions = [
            (area.x + 80, area.y + 120),
            (area.x + 80, area.y + 230),
            (area.x + 80, area.y + 340),
            (area.x + 720, area.y + 120),
            (area.x + 720, area.y + 230),
            (area.x + 720, area.y + 340),
        ]

        for (label, bg, border), (x, y) in zip(items, positions):
            rect = pygame.Rect(x, y, 185, 70)
            self.draw_component_box(rect, label, "déjà fourni", bg, border)
            draw_arrow(self.screen, rect.center, brain.center, border, 2)

        note = pygame.Rect(area.x + 240, area.bottom - 115, area.w - 480, 60)
        pygame.draw.rect(self.screen, YELLOW_LIGHT, note, border_radius=12)
        pygame.draw.rect(self.screen, YELLOW, note, 2, border_radius=12)
        draw_centered_text(self.screen, "L'utilisateur exploite le modèle, il ne le réentraîne pas.", f["h2"], TEXT, note.center)

    def draw_quantization(self, area):
        f = self.fonts
        draw_text(self.screen, "Quantification : rendre le modèle plus léger", f["h3"], ORANGE, area.x + 35, area.y + 32)

        left = pygame.Rect(area.x + 80, area.y + 110, 300, 320)
        right = pygame.Rect(area.x + 620, area.y + 110, 300, 320)

        self.draw_component_box(left, "Modèle FP16", "plus précis mais lourd", RED_LIGHT, RED)
        self.draw_component_box(right, "Modèle GGUF Q4/Q5", "plus compact", GREEN_LIGHT, GREEN)

        for i in range(6):
            y = left.y + 90 + i * 32
            pygame.draw.rect(self.screen, RED, pygame.Rect(left.x + 60, y, 180, 18), border_radius=999)
        for i in range(6):
            y = right.y + 90 + i * 32
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(right.x + 95, y, 110, 18), border_radius=999)

        draw_arrow(self.screen, (left.right + 28, left.centery), (right.x - 28, right.centery), ORANGE, 5)
        draw_centered_text(self.screen, "compression", f["h2"], ORANGE, (area.centerx, left.centery - 28))

        txt = "Idée pédagogique : on accepte une petite approximation pour utiliser le modèle sur une machine beaucoup plus raisonnable."
        draw_multiline(self.screen, txt, f["normal"], MUTED, area.x + 185, area.bottom - 120, area.w - 370)

    def draw_engine(self, area):
        f = self.fonts
        draw_text(self.screen, "Moteur d'inférence : Ollama / LM Studio / llama.cpp", f["h3"], ORANGE, area.x + 35, area.y + 32)

        center = pygame.Rect(area.x + 350, area.y + 210, 260, 130)
        self.draw_component_box(center, "Moteur d'inférence", "chef d'orchestre", GREEN_LIGHT, GREEN)

        around = [
            ("Charge le modèle", area.x + 90, area.y + 100, BLUE_LIGHT, BLUE),
            ("Applique le template", area.x + 650, area.y + 100, PURPLE_LIGHT, PURPLE),
            ("Tokenise", area.x + 90, area.y + 380, YELLOW_LIGHT, YELLOW),
            ("Appelle CPU/GPU", area.x + 650, area.y + 380, ORANGE_LIGHT, ORANGE),
        ]

        for title, x, y, bg, border in around:
            r = pygame.Rect(x, y, 220, 88)
            self.draw_component_box(r, title, "", bg, border)
            draw_arrow(self.screen, r.center, center.center, border, 3)

        code_rect = pygame.Rect(area.x + 200, area.bottom - 130, area.w - 400, 70)
        pygame.draw.rect(self.screen, PANEL, code_rect, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, code_rect, 1, border_radius=12)
        draw_text(self.screen, "Exemple d'usage", f["h3"], TEXT, code_rect.x + 18, code_rect.y + 12)
        draw_text(self.screen, "ollama run mistral", f["mono_big"], BLUE, code_rect.x + 18, code_rect.y + 42)

    def draw_gpu(self, area):
        f = self.fonts
        draw_text(self.screen, "GPU Nvidia : accélérateur de calcul", f["h3"], ORANGE, area.x + 35, area.y + 32)

        cpu = pygame.Rect(area.x + 90, area.y + 125, 300, 210)
        gpu = pygame.Rect(area.x + 585, area.y + 125, 300, 210)

        self.draw_component_box(cpu, "CPU seul", "fonctionne mais lent", PANEL, BORDER)
        self.draw_component_box(gpu, "GPU Nvidia", "CUDA accélère", ORANGE_LIGHT if self.gpu_enabled else PANEL, ORANGE if self.gpu_enabled else BORDER)

        for i in range(5):
            draw_progress(self.screen, pygame.Rect(cpu.x + 55, cpu.y + 88 + i * 24, 190, 12), 0.32, RED)

        for i in range(5):
            draw_progress(self.screen, pygame.Rect(gpu.x + 55, gpu.y + 88 + i * 24, 190, 12), 0.86 if self.gpu_enabled else 0.32, GREEN if self.gpu_enabled else RED)

        draw_arrow(self.screen, (cpu.right + 35, cpu.centery), (gpu.x - 35, gpu.centery), ORANGE, 5)
        draw_centered_text(self.screen, "mêmes calculs, beaucoup plus rapides", f["h2"], ORANGE, (area.centerx, cpu.bottom + 42))

        matrix = pygame.Rect(area.x + 210, area.y + 440, area.w - 420, 100)
        pygame.draw.rect(self.screen, BLUE_LIGHT, matrix, border_radius=12)
        pygame.draw.rect(self.screen, BLUE, matrix, 2, border_radius=12)
        draw_centered_text(self.screen, "vecteurs × matrices de poids → nouveaux vecteurs", f["mono_big"], TEXT, matrix.center)

    def draw_memory(self, area):
        f = self.fonts
        draw_text(self.screen, "Mémoire : RAM système et VRAM de la carte graphique", f["h3"], ORANGE, area.x + 35, area.y + 32)

        vram = pygame.Rect(area.x + 110, area.y + 95, 330, 410)
        ram = pygame.Rect(area.x + 560, area.y + 95, 330, 410)

        self.draw_component_box(vram, "VRAM GPU", "rapide, mais limitée", ORANGE_LIGHT, ORANGE)
        self.draw_component_box(ram, "RAM système", "plus grande, plus lente", BLUE_LIGHT, BLUE)

        y = vram.y + 95
        for label, val, note in VRAM_BARS:
            draw_text(self.screen, label, f["h3"], TEXT, vram.x + 28, y - 4)
            draw_progress(self.screen, pygame.Rect(vram.x + 28, y + 24, 210, 16), val, ORANGE)
            draw_text(self.screen, note, f["small"], MUTED, vram.x + 248, y + 21)
            y += 72

        blocks = [
            ("Modèle", 0.70),
            ("KV cache", 0.45),
            ("Système", 0.25),
            ("Documents", 0.35),
        ]
        y = ram.y + 110
        for label, val in blocks:
            draw_text(self.screen, label, f["h3"], TEXT, ram.x + 40, y - 4)
            draw_progress(self.screen, pygame.Rect(ram.x + 150, y, 140, 16), val, BLUE)
            y += 60

        note = "Si le modèle ne tient pas assez en VRAM, il peut fonctionner, mais la vitesse chute souvent."
        draw_multiline(self.screen, note, f["normal"], MUTED, area.x + 180, area.bottom - 100, area.w - 360)

    def draw_template(self, area):
        f = self.fonts
        draw_text(self.screen, "Chat template : formater correctement la conversation", f["h3"], ORANGE, area.x + 35, area.y + 32)

        raw = pygame.Rect(area.x + 65, area.y + 95, area.w - 130, 90)
        pygame.draw.rect(self.screen, BLUE_LIGHT, raw, border_radius=12)
        pygame.draw.rect(self.screen, BLUE, raw, 2, border_radius=12)
        draw_text(self.screen, "Message utilisateur", f["h3"], BLUE, raw.x + 18, raw.y + 12)
        draw_text(self.screen, PROMPT, f["mono_big"], TEXT, raw.x + 18, raw.y + 45)

        draw_arrow(self.screen, (area.centerx, raw.bottom + 10), (area.centerx, raw.bottom + 65), ORANGE, 4)

        templ = pygame.Rect(area.x + 120, area.y + 260, area.w - 240, 210)
        pygame.draw.rect(self.screen, PANEL, templ, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, templ, 2, border_radius=12)
        lines = [
            "<s>[INST]",
            "<<SYS>> Tu es un assistant utile. <</SYS>>",
            "Explique simplement ce qu'est un séquenceur ADN.",
            "[/INST]",
        ]
        y = templ.y + 28
        for line in lines:
            draw_text(self.screen, line, f["mono_big"], PURPLE if line.startswith("<") or line.startswith("[") else TEXT, templ.x + 30, y)
            y += 42

        note = "Le moteur d'inférence applique normalement ce format automatiquement si le modèle est bien identifié."
        draw_multiline(self.screen, note, f["normal"], MUTED, area.x + 170, area.bottom - 100, area.w - 340)

    def draw_generation(self, area):
        f = self.fonts
        draw_text(self.screen, "Génération locale token par token", f["h3"], ORANGE, area.x + 35, area.y + 32)

        output = pygame.Rect(area.x + 55, area.y + 85, area.w - 110, 95)
        pygame.draw.rect(self.screen, GREEN_LIGHT, output, border_radius=12)
        pygame.draw.rect(self.screen, GREEN, output, 2, border_radius=12)
        draw_text(self.screen, "Réponse en cours", f["h3"], GREEN, output.x + 18, output.y + 14)

        tokens = ANSWER.split(" ")
        visible = tokens[:self.generated]
        text = " ".join(visible) if visible else "▌"
        if self.generated < len(tokens):
            text += " ▌"
        draw_multiline(self.screen, text, f["mono_big"], TEXT, output.x + 18, output.y + 50, output.w - 36)

        x_positions = [area.x + 100, area.x + 330, area.x + 560, area.x + 790]
        y = area.y + 245
        steps = [
            ("Prompt", BLUE_LIGHT, BLUE),
            ("Tokens", YELLOW_LIGHT, YELLOW),
            ("Mistral 7B", PURPLE_LIGHT, PURPLE),
            ("GPU", ORANGE_LIGHT if self.gpu_enabled else PANEL, ORANGE if self.gpu_enabled else BORDER),
            ("Logits", PANEL, BORDER),
            ("Sampling", GREEN_LIGHT, GREEN),
            ("Token choisi", ORANGE_LIGHT, ORANGE),
            ("Ajout contexte", BLUE_LIGHT, BLUE),
        ]

        for i, (label, bg, border) in enumerate(steps):
            x = x_positions[i % 4]
            yy = y + (i // 4) * 150
            rect = pygame.Rect(x, yy, 170, 80)
            self.draw_component_box(rect, label, "", bg, border)
            if i % 4 != 0:
                prev = pygame.Rect(x_positions[(i - 1) % 4], yy, 170, 80)
                draw_arrow(self.screen, (prev.right + 5, prev.centery), (rect.x - 5, rect.centery), DARK, 2)

        if self.generated == 0:
            hint = pygame.Rect(area.x + 250, area.bottom - 115, area.w - 500, 60)
            pygame.draw.rect(self.screen, YELLOW_LIGHT, hint, border_radius=12)
            pygame.draw.rect(self.screen, YELLOW, hint, 2, border_radius=12)
            draw_centered_text(self.screen, "Clique sur « Générer token » ou appuie sur G", f["h2"], TEXT, hint.center)

    def draw_interface(self, area):
        f = self.fonts
        draw_text(self.screen, "Interface utilisateur : rendre l'IA locale accessible", f["h3"], ORANGE, area.x + 35, area.y + 32)

        window = pygame.Rect(area.x + 120, area.y + 80, area.w - 240, 470)
        pygame.draw.rect(self.screen, PANEL, window, border_radius=14)
        pygame.draw.rect(self.screen, BORDER, window, 2, border_radius=14)

        titlebar = pygame.Rect(window.x, window.y, window.w, 36)
        pygame.draw.rect(self.screen, YELLOW_LIGHT, titlebar, border_top_left_radius=14, border_top_right_radius=14)
        pygame.draw.rect(self.screen, BORDER, titlebar, 1, border_top_left_radius=14, border_top_right_radius=14)
        draw_centered_text(self.screen, "Chat local — Mistral 7B", f["h3"], TEXT, titlebar.center)

        user = pygame.Rect(window.x + 35, window.y + 80, window.w - 70, 72)
        pygame.draw.rect(self.screen, BLUE_LIGHT, user, border_radius=12)
        pygame.draw.rect(self.screen, BLUE, user, 2, border_radius=12)
        draw_text(self.screen, "Utilisateur", f["h3"], BLUE, user.x + 18, user.y + 10)
        draw_text(self.screen, PROMPT, f["mono"], TEXT, user.x + 18, user.y + 40)

        bot = pygame.Rect(window.x + 35, window.y + 190, window.w - 70, 125)
        pygame.draw.rect(self.screen, GREEN_LIGHT, bot, border_radius=12)
        pygame.draw.rect(self.screen, GREEN, bot, 2, border_radius=12)
        draw_text(self.screen, "Assistant local", f["h3"], GREEN, bot.x + 18, bot.y + 10)
        draw_multiline(self.screen, ANSWER, f["mono"], TEXT, bot.x + 18, bot.y + 42, bot.w - 36)

        input_rect = pygame.Rect(window.x + 35, window.bottom - 75, window.w - 70, 44)
        pygame.draw.rect(self.screen, LIGHT_PANEL, input_rect, border_radius=999)
        pygame.draw.rect(self.screen, BORDER, input_rect, 2, border_radius=999)
        draw_text(self.screen, "Écris ton message ici…", f["normal"], MUTED, input_rect.x + 20, input_rect.y + 13)

        note = "L'interface ne remplace pas le modèle : elle rend simplement le système utilisable."
        draw_text(self.screen, note, f["normal"], MUTED, area.x + 190, area.bottom - 70)

    def draw_rag(self, area):
        f = self.fonts
        draw_text(self.screen, "RAG : connecter le modèle à ses documents locaux", f["h3"], ORANGE, area.x + 35, area.y + 32)

        stages = [
            ("Documents", "PDF, notes, fiches", BLUE_LIGHT, BLUE),
            ("Découpage", "chunks", YELLOW_LIGHT, YELLOW),
            ("Embeddings", "vecteurs", PURPLE_LIGHT, PURPLE),
            ("Base vectorielle", "recherche", GREEN_LIGHT, GREEN),
            ("Contexte injecté", "prompt enrichi", ORANGE_LIGHT, ORANGE),
            ("Mistral 7B", "réponse guidée", RED_LIGHT, RED),
        ]

        x = area.x + 65
        y = area.y + 170
        box_w = 135
        gap = 23
        prev = None

        for title, sub, bg, border in stages:
            rect = pygame.Rect(x, y, box_w, 95)
            self.draw_component_box(rect, title, sub, bg if self.rag_enabled else PANEL, border if self.rag_enabled else BORDER)
            if prev:
                draw_arrow(self.screen, (prev.right + 4, prev.centery), (rect.x - 4, rect.centery), border if self.rag_enabled else BORDER, 3)
            prev = rect
            x += box_w + gap

        doc = pygame.Rect(area.x + 180, area.y + 360, area.w - 360, 92)
        pygame.draw.rect(self.screen, GREEN_LIGHT if self.rag_enabled else PANEL, doc, border_radius=12)
        pygame.draw.rect(self.screen, GREEN if self.rag_enabled else BORDER, doc, 2, border_radius=12)
        draw_text(self.screen, "Passage retrouvé dans les documents", f["h3"], GREEN if self.rag_enabled else MUTED, doc.x + 18, doc.y + 14)
        draw_text(self.screen, "« Le séquenceur lit l'ordre des bases A, C, G et T... »", f["mono"], TEXT, doc.x + 18, doc.y + 48)

        note = "Le RAG est optionnel pour un premier test, mais essentiel pour interroger ses propres documents sans réentraîner le modèle."
        draw_multiline(self.screen, note, f["normal"], MUTED, area.x + 160, area.bottom - 100, area.w - 320)

    def draw_minimal_setup(self, area):
        f = self.fonts
        draw_text(self.screen, "Configuration minimale recommandée", f["h3"], ORANGE, area.x + 35, area.y + 32)

        x0 = area.x + 100
        y0 = area.y + 95
        cols = 2
        box_w = 360
        box_h = 88
        gap_x = 95
        gap_y = 38

        items = [
            ("PC", "16 Go RAM minimum, plus confortable avec 32 Go", BLUE_LIGHT, BLUE),
            ("GPU Nvidia", "6–8 Go VRAM pour démarrer, 12 Go+ confortable", ORANGE_LIGHT, ORANGE),
            ("Moteur", "Ollama, LM Studio ou llama.cpp", GREEN_LIGHT, GREEN),
            ("Modèle", "Mistral 7B Instruct en GGUF Q4/Q5", PURPLE_LIGHT, PURPLE),
            ("Interface", "LM Studio ou Open WebUI", YELLOW_LIGHT, YELLOW),
            ("Option RAG", "documents locaux + embeddings + recherche", RED_LIGHT, RED),
        ]

        for i, (title, sub, bg, border) in enumerate(items):
            x = x0 + (i % cols) * (box_w + gap_x)
            y = y0 + (i // cols) * (box_h + gap_y)
            rect = pygame.Rect(x, y, box_w, box_h)
            self.draw_component_box(rect, title, sub, bg, border)

        final = pygame.Rect(area.x + 215, area.bottom - 130, area.w - 430, 75)
        pygame.draw.rect(self.screen, GREEN_LIGHT, final, border_radius=14)
        pygame.draw.rect(self.screen, GREEN, final, 2, border_radius=14)
        draw_centered_text(self.screen, "Objectif : une IA locale minimale, autonome et réutilisable.", f["h2"], TEXT, final.center)

    def draw(self):
        self.buttons = {}
        self.screen.fill(BG)
        self.draw_header()
        self.draw_buttons()
        self.draw_left_panel()
        self.draw_canvas()
        self.draw_bottom_pipeline()
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
                    self.handle_key(event)
            self.draw()


if __name__ == "__main__":
    LocalLLMDemo().run()
