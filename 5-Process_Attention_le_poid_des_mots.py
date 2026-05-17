import math
import sys
from dataclasses import dataclass

import pygame

WIDTH, HEIGHT = 1500, 930
FPS = 60

BG = (255, 250, 240)
PANEL = (255, 255, 255)
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
RED = (216, 59, 45)
RED_LIGHT = (253, 232, 228)
PURPLE = (113, 69, 168)
DARK = (35, 35, 35)

LEVELS = [
    ("1. Lecture linéaire", "Un système simple lit les mots dans l'ordre.",
     "Avant l'attention, imaginons un programme qui lit les mots les uns après les autres. Il voit la phrase de gauche à droite, mais il ne sait pas encore quels mots sont importants pour comprendre un pronom comme il ou elle.",
     "Ajouter du contexte"),
    ("2. Chercher le mot important", "Tous les mots ne comptent pas pareil.",
     "L'idée d'attention consiste à demander : quand je lis ce mot, quels autres mots dois-je regarder ? Ici, le pronom doit être relié au bon mot précédent.",
     "Visualiser les poids"),
    ("3. Poids d'attention", "Chaque mot reçoit un score d'importance.",
     "Un Transformer attribue un poids à chaque mot du contexte. Les mots utiles reçoivent un poids fort, les mots peu utiles reçoivent un poids faible.",
     "Comprendre Query / Key / Value"),
    ("4. Query / Key / Value", "Ce que je cherche, ce que les mots proposent, ce que je récupère.",
     "Le mot courant produit une Query : ce qu'il cherche. Les autres mots possèdent des Keys : ce qu'ils proposent. Les Values contiennent l'information transmise si l'attention est forte.",
     "Calculer automatiquement les scores"),
    ("5. Softmax", "Transformer des scores en probabilités.",
     "Les scores bruts ne sont pas encore des probabilités. La fonction softmax les transforme en valeurs positives dont la somme vaut 1.",
     "Passer à l'attention multi-têtes"),
    ("6. Multi-head attention", "Plusieurs regards en parallèle.",
     "Un vrai Transformer n'a pas une seule attention. Il utilise plusieurs têtes d'attention qui peuvent regarder la grammaire, le sens ou la proximité.",
     "Relier à la génération de texte"),
    ("7. Génération de texte", "Utiliser le contexte pour proposer le mot suivant.",
     "Un Transformer utilise l'attention pour construire une représentation riche du contexte, puis produit une distribution de probabilités sur le prochain mot.",
     "Revenir au début"),
]

EXAMPLES = [
    {
        "name": "Qui aboie ?",
        "sentence": ["Le", "chat", "regarde", "le", "chien", "parce", "qu'il", "aboie"],
        "query_index": 6,
        "target_index": 4,
        "reason": "qu'il renvoie plutôt à chien, car c'est le chien qui aboie.",
        "raw": [0.2, 1.1, 0.1, 0.3, 3.2, 0.1, 0.0, 2.1],
        "next": [("fort", .38), ("près", .22), ("bruyamment", .18), ("encore", .12), ("vite", .10)],
    },
    {
        "name": "Qui miaule ?",
        "sentence": ["Le", "chat", "regarde", "le", "chien", "parce", "qu'il", "miaule"],
        "query_index": 6,
        "target_index": 1,
        "reason": "qu'il renvoie plutôt à chat, car c'est le chat qui miaule.",
        "raw": [0.2, 3.3, 0.1, 0.3, 1.2, 0.1, 0.0, 2.0],
        "next": [("doucement", .34), ("encore", .21), ("près", .17), ("fort", .16), ("souvent", .12)],
    },
    {
        "name": "Petite ?",
        "sentence": ["La", "clé", "est", "dans", "la", "boîte", "parce", "qu'elle", "est", "petite"],
        "query_index": 7,
        "target_index": 1,
        "reason": "qu'elle renvoie plutôt à clé, car une clé peut être petite.",
        "raw": [0.2, 3.4, 0.1, 0.1, 0.4, 1.1, 0.1, 0.0, 0.2, 2.1],
        "next": [("et", .35), ("légère", .28), ("ancienne", .16), ("utile", .12), ("fragile", .09)],
    },
    {
        "name": "Ouverte ?",
        "sentence": ["La", "clé", "est", "dans", "la", "boîte", "parce", "qu'elle", "est", "ouverte"],
        "query_index": 7,
        "target_index": 5,
        "reason": "qu'elle renvoie plutôt à boîte, car une boîte peut être ouverte.",
        "raw": [0.2, 1.2, 0.1, 0.1, 0.4, 3.5, 0.1, 0.0, 0.2, 2.0],
        "next": [("depuis", .31), ("sur", .24), ("et", .22), ("devant", .13), ("souvent", .10)],
    },
]

HEADS = [
    ("Tête 1 : grammaire", BLUE, "Regarde les noms compatibles avec le pronom."),
    ("Tête 2 : sens", GREEN, "Regarde le mot qui peut réaliser l'action."),
    ("Tête 3 : proximité", PURPLE, "Regarde les mots proches du pronom."),
]


def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [v / s for v in exps]


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


def draw_multiline(screen, text, font, color, x, y, max_width, line_gap=5, max_lines=None):
    words = text.split(" ")
    line, yy, lines = "", y, 0
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


def draw_progress(screen, rect, value, color):
    value = max(0, min(1, value))
    pygame.draw.rect(screen, (232, 232, 232), rect, border_radius=999)
    inner = pygame.Rect(rect.x, rect.y, int(rect.w * value), rect.h)
    if inner.w > 0:
        pygame.draw.rect(screen, color, inner, border_radius=999)


def draw_arrow(screen, start, end, color, width=3):
    width = max(1, int(width))
    pygame.draw.line(screen, color, start, end, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length, spread = 12, 0.45
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    pygame.draw.polygon(screen, color, [end, p1, p2])


@dataclass
class Button:
    rect: pygame.Rect
    text: str
    kind: str = "light"

    def draw(self, screen, fonts):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        if self.kind == "dark":
            bg, fg, border = DARK, (255, 255, 255), DARK
        elif self.kind == "orange":
            bg, fg, border = ORANGE, TEXT, DARK
        elif self.kind == "blue":
            bg, fg, border = BLUE, (255, 255, 255), BLUE
        else:
            bg, fg, border = PANEL, TEXT, BORDER
        if hover:
            bg = tuple(max(0, c - 8) for c in bg)
        pygame.draw.rect(screen, bg, self.rect, border_radius=8)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=8)
        draw_centered_text(screen, self.text, fonts["button"], fg, self.rect.center)


class AttentionDemo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Attention Visualizer — comprendre les Transformers")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": pygame.font.SysFont("arial", 27, bold=True),
            "h2": pygame.font.SysFont("arial", 18, bold=True),
            "h3": pygame.font.SysFont("arial", 15, bold=True),
            "normal": pygame.font.SysFont("arial", 14),
            "small": pygame.font.SysFont("arial", 12),
            "mono": pygame.font.SysFont("consolas", 14),
            "button": pygame.font.SysFont("arial", 13, bold=True),
            "big": pygame.font.SysFont("arial", 24, bold=True),
        }
        self.level = 0
        self.example_index = 0
        self.selected_token = None
        self.manual_target = None
        self.buttons = {}
        self.token_rects = []

    @property
    def ex(self):
        return EXAMPLES[self.example_index]

    @property
    def lvl(self):
        return LEVELS[self.level]

    def attention_weights(self):
        return softmax(self.ex["raw"])

    def next_level(self):
        self.level = (self.level + 1) % len(LEVELS)
        self.selected_token = None
        self.manual_target = None

    def prev_level(self):
        self.level = max(0, self.level - 1)
        self.selected_token = None
        self.manual_target = None

    def next_example(self):
        self.example_index = (self.example_index + 1) % len(EXAMPLES)
        self.selected_token = None
        self.manual_target = None

    def prev_example(self):
        self.example_index = (self.example_index - 1) % len(EXAMPLES)
        self.selected_token = None
        self.manual_target = None

    def handle_click(self, pos):
        for key, b in self.buttons.items():
            if b.rect.collidepoint(pos):
                if key == "next_level": self.next_level()
                elif key == "prev_level": self.prev_level()
                elif key == "next_example": self.next_example()
                elif key == "prev_example": self.prev_example()
                elif key == "reset":
                    self.level = 0; self.example_index = 0; self.selected_token = None; self.manual_target = None
                return
        if self.level >= 1:
            for i, r in enumerate(self.token_rects):
                if r.collidepoint(pos):
                    if self.selected_token is None:
                        self.selected_token = i
                    else:
                        self.manual_target = i
                    return

    def handle_key(self, event):
        if event.key in (pygame.K_RIGHT, pygame.K_SPACE): self.next_level()
        elif event.key == pygame.K_LEFT: self.prev_level()
        elif event.key == pygame.K_e: self.next_example()
        elif event.key == pygame.K_r: self.level = 0; self.selected_token = None; self.manual_target = None
        elif event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()

    def draw_header(self):
        f = self.fonts
        draw_text(self.screen, "ATTENTION VISUALIZER — COMPRENDRE LES TRANSFORMERS", f["title"], TEXT, 34, 22)
        draw_text(self.screen, "Un atelier progressif pour visualiser l’attention, les poids, le softmax et les têtes multiples.", f["normal"], MUTED, 34, 56)
        stats = [("Niveau", f"{self.level+1}/7"), ("Exemple", self.ex["name"]), ("Mot clé", self.ex["sentence"][self.ex["query_index"]]), ("Cible", self.ex["sentence"][self.ex["target_index"]])]
        x = 800
        for label, val in stats:
            draw_centered_text(self.screen, label, f["small"], MUTED, (x+65, 25))
            rect = pygame.Rect(x, 42, 130, 30)
            pygame.draw.rect(self.screen, (255,241,198), rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)
            draw_centered_text(self.screen, val[:18], f["button"], TEXT, rect.center)
            x += 142

    def draw_buttons(self):
        f = self.fonts; y = 92
        self.buttons["prev_level"] = Button(pygame.Rect(34, y, 140, 42), "← Niveau", "light")
        self.buttons["next_level"] = Button(pygame.Rect(184, y, 310, 42), "▶ " + self.lvl[3], "orange")
        self.buttons["prev_example"] = Button(pygame.Rect(516, y, 140, 42), "← Exemple", "light")
        self.buttons["next_example"] = Button(pygame.Rect(666, y, 170, 42), "Exemple suivant", "blue")
        self.buttons["reset"] = Button(pygame.Rect(1320, y, 145, 42), "Réinitialiser", "dark")
        for k in ["prev_level", "next_level", "prev_example", "next_example", "reset"]:
            self.buttons[k].draw(self.screen, f)

    def draw_left_panel(self):
        f = self.fonts
        rect = pygame.Rect(34, 158, 410, 710)
        draw_panel(self.screen, rect, "Concept couvert", f)
        draw_text(self.screen, self.lvl[0], f["big"], TEXT, 54, 200)
        draw_text(self.screen, self.lvl[1], f["h3"], BLUE, 54, 236)
        draw_multiline(self.screen, self.lvl[2], f["normal"], MUTED, 54, 274, 350, 5, max_lines=11)
        box = pygame.Rect(54, 470, 350, 180)
        pygame.draw.rect(self.screen, (255,253,247), box, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, box, 1, border_radius=10)
        draw_text(self.screen, "Phrase exemple", f["h3"], TEXT, 72, 490)
        draw_multiline(self.screen, " ".join(self.ex["sentence"]), f["normal"], TEXT, 72, 522, 315, 5, max_lines=4)
        draw_text(self.screen, "Interprétation attendue", f["h3"], BLUE, 72, 610)
        draw_multiline(self.screen, self.ex["reason"], f["small"], MUTED, 72, 636, 315, 4, max_lines=4)
        draw_text(self.screen, "Touches utiles", f["h3"], BLUE, 54, 704)
        tips = "Espace ou flèche droite : niveau suivant. Flèche gauche : niveau précédent. E : changer d'exemple. En niveaux 2 et +, clique sur les mots."
        draw_multiline(self.screen, tips, f["small"], MUTED, 54, 732, 340, 4, max_lines=5)

    def draw_tokens_panel(self):
        f = self.fonts
        rect = pygame.Rect(468, 158, 998, 380)
        draw_panel(self.screen, rect, "Phrase découpée en tokens", f)
        sent = self.ex["sentence"]
        self.token_rects = []
        token_widths = [max(58, f["h3"].size(t)[0] + 28) for t in sent]
        total = sum(token_widths) + 12 * (len(sent) - 1)
        x = rect.x + (rect.w - total) // 2
        y = 310
        weights = self.attention_weights()
        for i, t in enumerate(sent):
            r = pygame.Rect(x, y, token_widths[i], 54)
            self.token_rects.append(r)
            bg, border = PANEL, BORDER
            if i == self.ex["query_index"]: bg, border = ORANGE_LIGHT, ORANGE
            if i == self.ex["target_index"] and self.level >= 1: bg, border = GREEN_LIGHT, GREEN
            if self.selected_token == i: bg, border = YELLOW_LIGHT, YELLOW
            if self.manual_target == i: bg, border = BLUE_LIGHT, BLUE
            pygame.draw.rect(self.screen, bg, r, border_radius=10)
            pygame.draw.rect(self.screen, border, r, 2, border_radius=10)
            draw_centered_text(self.screen, t, f["h3"], TEXT, r.center)
            if self.level == 0 and i == (pygame.time.get_ticks() // 550) % len(sent):
                pygame.draw.rect(self.screen, ORANGE, r, 3, border_radius=10)
            if self.level >= 3:
                draw_centered_text(self.screen, f"K{i+1}", f["small"], MUTED, (r.centerx, r.y - 18))
            if self.level >= 4:
                draw_centered_text(self.screen, f"{weights[i]:.2f}", f["small"], BLUE, (r.centerx, r.bottom + 18))
            x += token_widths[i] + 12
        if self.level >= 1:
            q = self.token_rects[self.ex["query_index"]]
            target = self.token_rects[self.ex["target_index"]]
            draw_arrow(self.screen, (q.centerx, q.y - 22), (target.centerx, target.y - 22), GREEN, 4)
        if self.manual_target is not None and self.selected_token is not None:
            a = self.token_rects[self.selected_token]; b = self.token_rects[self.manual_target]
            draw_arrow(self.screen, (a.centerx, a.bottom + 24), (b.centerx, b.bottom + 24), BLUE, 3)
        msg = "Le curseur avance mot par mot : lecture linéaire." if self.level == 0 else "Orange : mot qui cherche le contexte. Vert : mot auquel il faut faire attention."
        draw_text(self.screen, msg, f["normal"], MUTED, rect.x + 28, rect.bottom - 42)

    def draw_bottom(self):
        f = self.fonts
        rect = pygame.Rect(468, 560, 998, 308)
        draw_panel(self.screen, rect, "Visualisation", f)
        if self.level == 0: self.draw_linear(rect)
        elif self.level == 1: self.draw_manual(rect)
        elif self.level == 2: self.draw_weights(rect)
        elif self.level == 3: self.draw_qkv(rect)
        elif self.level == 4: self.draw_softmax(rect)
        elif self.level == 5: self.draw_multihead(rect)
        else: self.draw_generation(rect)

    def draw_linear(self, rect):
        f = self.fonts
        draw_text(self.screen, "Lecture simple de gauche à droite", f["h3"], BLUE, rect.x+24, rect.y+58)
        draw_multiline(self.screen, "Un modèle simple pourrait seulement avancer mot par mot. Mais pour comprendre un pronom, il faut parfois regarder plus loin dans le contexte.", f["normal"], MUTED, rect.x+24, rect.y+92, 900, 5, 4)
        steps = ["Le", "chat", "regarde", "...", "qu'il", "?"]
        x, y = rect.x+60, rect.y+180
        for i, s in enumerate(steps):
            box = pygame.Rect(x, y, 95, 48)
            pygame.draw.rect(self.screen, YELLOW_LIGHT if i <= 4 else RED_LIGHT, box, border_radius=10)
            pygame.draw.rect(self.screen, BORDER if i <= 4 else RED, box, 2, border_radius=10)
            draw_centered_text(self.screen, s, f["h3"], TEXT, box.center)
            if i < len(steps)-1: draw_arrow(self.screen, (box.right+5, box.centery), (box.right+45, box.centery), MUTED, 2)
            x += 140

    def draw_manual(self, rect):
        f = self.fonts
        draw_text(self.screen, "À toi de jouer : clique sur le pronom, puis sur le mot important.", f["h3"], BLUE, rect.x+24, rect.y+58)
        draw_multiline(self.screen, "Cette étape simule l'intuition humaine : quels mots sont importants pour comprendre le mot courant ?", f["normal"], MUTED, rect.x+24, rect.y+92, 900, 5, 3)
        if self.selected_token is None: status = "Étape 1 : clique sur le mot qui cherche du contexte, par exemple qu'il."
        elif self.manual_target is None: status = "Étape 2 : clique sur le mot auquel il faut faire attention."
        else: status = f"Tu as relié {self.ex['sentence'][self.selected_token]} à {self.ex['sentence'][self.manual_target]}. Compare avec la cible verte."
        box = pygame.Rect(rect.x+24, rect.y+190, 660, 60)
        pygame.draw.rect(self.screen, YELLOW_LIGHT, box, border_radius=10); pygame.draw.rect(self.screen, YELLOW, box, 2, border_radius=10)
        draw_multiline(self.screen, status, f["normal"], TEXT, box.x+18, box.y+18, 620, 4, 2)

    def draw_weights(self, rect):
        f = self.fonts
        draw_text(self.screen, "Scores d'attention : quels mots comptent le plus ?", f["h3"], BLUE, rect.x+24, rect.y+56)
        vals = self.ex["raw"]; max_v = max(vals); y = rect.y+96
        for i, (tok, val) in enumerate(zip(self.ex["sentence"], vals)):
            col = GREEN if i == self.ex["target_index"] else BLUE
            draw_text(self.screen, f"{tok:>10}", f["mono"], TEXT, rect.x+36, y)
            draw_progress(self.screen, pygame.Rect(rect.x+145, y+2, 260, 14), val/max_v, col)
            draw_text(self.screen, f"{val:.2f}", f["mono"], MUTED, rect.x+420, y)
            y += 24
        draw_multiline(self.screen, "Le mot vert reçoit un score fort : c'est celui auquel le modèle devrait prêter le plus d'attention.", f["normal"], MUTED, rect.x+560, rect.y+105, 360, 5, 6)

    def draw_qkv(self, rect):
        f = self.fonts
        draw_text(self.screen, "Query / Key / Value", f["h3"], BLUE, rect.x+24, rect.y+56)
        cards = [("Query", "Ce que le mot courant cherche.", ORANGE, "qu'il cherche un antécédent compatible."), ("Key", "Ce que chaque mot propose.", BLUE, "chien propose : nom masculin, peut aboyer."), ("Value", "L'information récupérée.", GREEN, "Si l'attention est forte, l'information est transmise.")]
        x = rect.x+24
        for title, sub, color, ex in cards:
            card = pygame.Rect(x, rect.y+98, 285, 145)
            pygame.draw.rect(self.screen, (255,253,247), card, border_radius=12); pygame.draw.rect(self.screen, color, card, 2, border_radius=12)
            draw_text(self.screen, title, f["big"], color, card.x+18, card.y+16)
            draw_multiline(self.screen, sub, f["normal"], TEXT, card.x+18, card.y+54, 240, 4, 2)
            draw_multiline(self.screen, ex, f["small"], MUTED, card.x+18, card.y+98, 240, 4, 2)
            x += 310

    def draw_softmax(self, rect):
        f = self.fonts
        draw_text(self.screen, "Softmax : des scores aux probabilités", f["h3"], BLUE, rect.x+24, rect.y+56)
        formula_rect = pygame.Rect(rect.x+24, rect.y+88, 390, 48)
        pygame.draw.rect(self.screen, (255,241,198), formula_rect, border_radius=10); pygame.draw.rect(self.screen, BORDER, formula_rect, 1, border_radius=10)
        draw_centered_text(self.screen, "softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)", f["mono"], TEXT, formula_rect.center)
        raw = self.ex["raw"]; probs = self.attention_weights(); y = rect.y+160
        for i, tok in enumerate(self.ex["sentence"][:8]):
            draw_text(self.screen, tok, f["mono"], TEXT, rect.x+36, y)
            draw_text(self.screen, f"score {raw[i]:.1f}", f["mono"], MUTED, rect.x+145, y)
            draw_progress(self.screen, pygame.Rect(rect.x+260, y+2, 190, 14), probs[i]/max(probs), GREEN if i == self.ex["target_index"] else BLUE)
            draw_text(self.screen, f"p={probs[i]:.2f}", f["mono"], TEXT, rect.x+470, y)
            y += 26
        draw_multiline(self.screen, "Après softmax, les valeurs peuvent être lues comme une répartition de l'attention : leur somme vaut 1.", f["normal"], MUTED, rect.x+650, rect.y+165, 300, 5, 5)

    def draw_multihead(self, rect):
        f = self.fonts
        draw_text(self.screen, "Multi-head attention : plusieurs regards simultanés", f["h3"], BLUE, rect.x+24, rect.y+56)
        for hi, (name, color, desc) in enumerate(HEADS):
            card = pygame.Rect(rect.x+24+hi*315, rect.y+92, 285, 170)
            pygame.draw.rect(self.screen, (255,253,247), card, border_radius=12); pygame.draw.rect(self.screen, color, card, 2, border_radius=12)
            draw_text(self.screen, name, f["h3"], color, card.x+14, card.y+14)
            draw_multiline(self.screen, desc, f["small"], MUTED, card.x+14, card.y+42, 245, 4, 2)
            choices = [self.ex["sentence"][self.ex["target_index"]], self.ex["sentence"][-1], self.ex["sentence"][max(0, self.ex["query_index"]-1)]]
            draw_text(self.screen, "Regarde surtout :", f["small"], TEXT, card.x+14, card.y+92)
            draw_text(self.screen, choices[hi], f["big"], color, card.x+14, card.y+112)
            draw_progress(self.screen, pygame.Rect(card.x+120, card.y+122, 130, 14), [.85, .65, .55][hi], color)

    def draw_generation(self, rect):
        f = self.fonts
        draw_text(self.screen, "Génération : prédire le prochain mot", f["h3"], BLUE, rect.x+24, rect.y+56)
        draw_text(self.screen, "Contexte :", f["h3"], TEXT, rect.x+24, rect.y+96)
        draw_multiline(self.screen, " ".join(self.ex["sentence"]), f["normal"], MUTED, rect.x+120, rect.y+96, 720, 4, 2)
        draw_text(self.screen, "Mots suivants possibles :", f["h3"], TEXT, rect.x+24, rect.y+150)
        y = rect.y+188
        maxp = self.ex["next"][0][1]
        for word, p in self.ex["next"]:
            draw_text(self.screen, word, f["mono"], TEXT, rect.x+44, y)
            draw_progress(self.screen, pygame.Rect(rect.x+160, y+2, 240, 16), p/maxp, ORANGE)
            draw_text(self.screen, f"{p*100:.0f}%", f["mono"], MUTED, rect.x+420, y)
            y += 32
        draw_multiline(self.screen, "C'est le lien avec les LLM : à chaque étape, le modèle produit une distribution de probabilités sur le prochain token.", f["normal"], MUTED, rect.x+560, rect.y+180, 360, 5, 5)

    def draw_footer(self):
        draw_text(self.screen, "Touches : Espace / → niveau suivant · ← niveau précédent · E changer d'exemple · R réinitialiser · Échap quitter", self.fonts["small"], MUTED, 34, 902)

    def draw(self):
        self.buttons = {}
        self.screen.fill(BG)
        self.draw_header(); self.draw_buttons(); self.draw_left_panel(); self.draw_tokens_panel(); self.draw_bottom(); self.draw_footer()
        pygame.display.flip()

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: self.handle_click(event.pos)
                if event.type == pygame.KEYDOWN: self.handle_key(event)
            self.draw()


if __name__ == "__main__":
    AttentionDemo().run()
