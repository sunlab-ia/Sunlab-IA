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
PURPLE = (113, 69, 168)
PURPLE_LIGHT = (241, 232, 250)
DARK = (35, 35, 35)

LEVELS = [
    {
        "title": "1. Axe masculin / féminin",
        "subtitle": "Les mots se placent selon une dimension de sens.",
        "concept": "On représente chaque mot comme un point dans un espace. À gauche, les mots plutôt masculins ; à droite, les mots plutôt féminins ; au centre, les mots dont le genre n'est pas déterminé par le mot seul.",
        "button": "Passer aux synonymes",
    },
    {
        "title": "2. Même relation, même vecteur",
        "subtitle": "Roi → Reine peut être vu comme un déplacement de genre.",
        "concept": "On isole Roi et Reine, puis on ajoute des mots proches : Monarque, Suzerain, Empereur. Quand la relation masculin → féminin est la même, le vecteur a la même direction.",
        "button": "Ajouter la parenté",
    },
    {
        "title": "3. Deux axes de sens",
        "subtitle": "Genre en X, ascendance / descendance en Y.",
        "concept": "On garde l'axe masculin → féminin, puis on ajoute un deuxième axe : parent → enfant. On commence à voir un espace vectoriel où plusieurs dimensions de sens se combinent.",
        "button": "Revenir au début",
    },
]

LEVEL1_WORDS = [
    ("Homme", -0.88, -0.62, "male"),
    ("Mari", -0.72, -0.42, "male"),
    ("Époux", -0.55, -0.22, "male"),
    ("Père", -0.82, 0.02, "male"),
    ("Roi", -0.62, 0.28, "male"),
    ("Paysan", -0.80, 0.50, "male"),
    ("Laboureur", -0.48, 0.62, "male"),
    ("Acteur", -0.34, 0.40, "male"),
    ("Il", -0.34, -0.62, "male"),

    # Femme comme opposé de Homme
    ("Femme", 0.88, -0.62, "female"),

    # Femme comme conjointe de Mari : même vecteur que Homme -> Femme
    ("Femme (épouse)", 1.04, -0.42, "female"),

    ("Épouse", 0.55, -0.22, "female"),
    ("Mère", 0.82, 0.02, "female"),
    ("Reine", 0.62, 0.28, "female"),
    ("Paysanne", 0.80, 0.50, "female"),
    ("Lavandière", 0.48, 0.62, "female"),
    ("Actrice", 0.34, 0.40, "female"),
    ("Elle", 0.34, -0.62, "female"),

    ("Médecin", 0.00, -0.36, "neutral"),
    ("Personne", 0.00, -0.12, "neutral"),
    ("Ils", -0.05, 0.12, "neutral"),
    ("Les villageois", 0.02, 0.36, "neutral"),
    ("Enfant", 0.00, 0.62, "neutral"),
]

LEVEL1_PAIRS = [
    ("Homme", "Femme"),
    ("Mari", "Femme (épouse)"),
    ("Époux", "Épouse"),
    ("Père", "Mère"),
    ("Roi", "Reine"),
    ("Paysan", "Paysanne"),
    ("Acteur", "Actrice"),
    ("Il", "Elle"),
]

LEVEL2_PAIRS = [
    ("Roi", "Reine", -0.55, -0.15, 0.55, -0.15),
    ("Suzerain", "Suzeraine", -0.55, -0.45, 0.55, -0.45),
    ("Monarque", "Son Altesse", -0.55, 0.16, 0.55, 0.16),
    ("Empereur", "Impératrice", -0.55, 0.46, 0.55, 0.46),
]
LEVEL2_SQUARE = [("Roi", -0.45, -0.36), ("Reine", 0.45, -0.36), ("Époux de la Reine", -0.45, 0.38), ("Épouse du Roi", 0.45, 0.38)]

LEVEL3_WORDS = [
    ("Grand-père paternel", -0.55, -0.72, "male"),
    ("Grand-mère paternelle", 0.15, -0.72, "female"),
    ("Roi", -0.55, -0.22, "male"), ("Reine", 0.55, -0.22, "female"),
    ("Grand-père maternel", -0.15, -0.72, "male"),
    ("Grand-mère maternelle", 0.55, -0.72, "female"),
    ("Enfant", 0.00, 0.55, "neutral"),
    ("Fils", -0.42, 0.72, "male"), ("Fille", 0.42, 0.72, "female"),
    ("Prince", -0.42, 0.88, "male"), ("Princesse", 0.42, 0.88, "female"),
]
GENRE_PAIRS_L3 = [("Grand-père paternel", "Grand-mère paternelle"), ("Roi", "Reine"), ("Grand-père maternel", "Grand-mère maternelle"), ("Fils", "Fille"), ("Prince", "Princesse")]
PARENT_ARROWS_L3 = [("Grand-père paternel", "Roi"), ("Grand-mère paternelle", "Roi"), ("Grand-père maternel", "Reine"), ("Grand-mère maternelle", "Reine"), ("Roi", "Enfant"), ("Reine", "Enfant"), ("Enfant", "Fils"), ("Enfant", "Fille"), ("Fils", "Prince"), ("Fille", "Princesse")]


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
        else:
            bg, fg, border = PANEL, TEXT, BORDER
        if hover:
            bg = tuple(max(0, c - 8) for c in bg)
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


class WordVectorDemo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Vecteurs de mots — genre, tokens et sens")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": pygame.font.SysFont("arial", 27, bold=True),
            "h2": pygame.font.SysFont("arial", 18, bold=True),
            "h3": pygame.font.SysFont("arial", 15, bold=True),
            "normal": pygame.font.SysFont("arial", 14),
            "small": pygame.font.SysFont("arial", 12),
            "button": pygame.font.SysFont("arial", 13, bold=True),
            "big": pygame.font.SysFont("arial", 24, bold=True),
        }
        self.level = 0
        self.reveal_step = 0
        self.show_vectors = True
        self.show_axes = True
        self.show_square = False
        self.buttons = {}

    @property
    def level_info(self):
        return LEVELS[self.level]

    def canvas_rect(self):
        return pygame.Rect(470, 170, 995, 695)

    def to_screen(self, x, y):
        rect = self.canvas_rect()
        sx = rect.x + rect.w / 2 + x * rect.w * 0.42
        sy = rect.y + rect.h / 2 + y * rect.h * 0.42
        return int(sx), int(sy)

    def next_level(self):
        self.level = (self.level + 1) % len(LEVELS)
        self.show_square = False
        self.reveal_step = 0

    def prev_level(self):
        self.level = (self.level - 1) % len(LEVELS)
        self.show_square = False
        self.reveal_step = 0

    def next_reveal(self):
        max_steps = {0: 6, 1: 5, 2: 7}
        self.reveal_step += 1
        if self.reveal_step > max_steps.get(self.level, 5):
            self.reveal_step = 0

    def handle_click(self, pos):
        for key, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                if key == "next_level": self.next_level()
                elif key == "prev_level": self.prev_level()
                elif key == "toggle_vectors": self.show_vectors = not self.show_vectors
                elif key == "toggle_axes": self.show_axes = not self.show_axes
                elif key == "toggle_square": self.show_square = not self.show_square
                elif key == "suite": self.next_reveal()
                elif key == "reset":
                    self.level = 0; self.reveal_step = 0; self.show_vectors = True; self.show_axes = True; self.show_square = False
                return

    def handle_key(self, event):
        if event.key in [pygame.K_RIGHT, pygame.K_SPACE]: self.next_level()
        elif event.key == pygame.K_LEFT: self.prev_level()
        elif event.key == pygame.K_s: self.next_reveal()
        elif event.key == pygame.K_v: self.show_vectors = not self.show_vectors
        elif event.key == pygame.K_a: self.show_axes = not self.show_axes
        elif event.key == pygame.K_c: self.show_square = not self.show_square
        elif event.key == pygame.K_r:
            self.level = 0; self.reveal_step = 0; self.show_vectors = True; self.show_axes = True; self.show_square = False
        elif event.key == pygame.K_ESCAPE:
            pygame.quit(); sys.exit()

    def draw_header(self):
        f = self.fonts
        draw_text(self.screen, "VECTEURS DE MOTS — GENRE, TOKENS ET SENS", f["title"], TEXT, 34, 22)
        draw_text(self.screen, "Un atelier pour comprendre comment un mot peut devenir un point, puis un vecteur dans un espace de sens.", f["normal"], MUTED, 34, 56)
        stats = [("Niveau", f"{self.level + 1} / {len(LEVELS)}"), ("Axe X", "genre"), ("Axe Y", "parenté" if self.level == 2 else "non utilisé"), ("Type", "pédagogique")]
        x = 790
        widths = [115, 115, 140, 170]
        for (label, value), w in zip(stats, widths):
            draw_centered_text(self.screen, label, f["small"], MUTED, (x + w // 2, 25))
            rect = pygame.Rect(x, 42, w, 30)
            pygame.draw.rect(self.screen, (255, 241, 198), rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)
            draw_centered_text(self.screen, value, f["button"], TEXT, rect.center)
            x += w + 14

    def draw_buttons(self):
        f = self.fonts; y = 92
        self.buttons["prev_level"] = Button(pygame.Rect(34, y, 130, 42), "← Niveau", "light")
        self.buttons["next_level"] = Button(pygame.Rect(174, y, 270, 42), "▶ " + self.level_info["button"], "orange")
        self.buttons["toggle_vectors"] = Button(pygame.Rect(470, y, 155, 42), "Vecteurs ON" if self.show_vectors else "Vecteurs OFF", "blue" if self.show_vectors else "light")
        self.buttons["toggle_axes"] = Button(pygame.Rect(640, y, 130, 42), "Axes ON" if self.show_axes else "Axes OFF", "blue" if self.show_axes else "light")
        self.buttons["suite"] = Button(pygame.Rect(785, y, 120, 42), "SUITE", "green")
        keys = ["prev_level", "next_level", "toggle_vectors", "toggle_axes", "suite"]
        if self.level == 1:
            self.buttons["toggle_square"] = Button(pygame.Rect(920, y, 200, 42), "Carré relationnel" if not self.show_square else "Synonymes", "green" if self.show_square else "light")
            keys.append("toggle_square")
        self.buttons["reset"] = Button(pygame.Rect(1320, y, 145, 42), "Réinitialiser", "dark")
        keys.append("reset")
        for key in keys: self.buttons[key].draw(self.screen, f)

    def draw_left_panel(self):
        f = self.fonts
        rect = pygame.Rect(34, 170, 410, 695)
        draw_panel(self.screen, rect, "Concept couvert", f)
        draw_text(self.screen, self.level_info["title"], f["big"], TEXT, 54, 214)
        draw_text(self.screen, self.level_info["subtitle"], f["h3"], BLUE, 54, 252)
        draw_multiline(self.screen, self.level_info["concept"], f["normal"], MUTED, 54, 292, 350, 5, max_lines=10)
        pygame.draw.rect(self.screen, (255, 253, 247), pygame.Rect(54, 500, 350, 190), border_radius=10)
        pygame.draw.rect(self.screen, BORDER, pygame.Rect(54, 500, 350, 190), 1, border_radius=10)
        draw_text(self.screen, "À retenir", f["h3"], TEXT, 72, 520)
        if self.level == 0:
            note = "Un token peut être vu comme un point. Ici, on force une seule dimension : le genre. Les mots neutres restent au centre."
        elif self.level == 1:
            note = "Une relation de sens peut être vue comme un déplacement. Roi → Reine ressemble à Empereur → Impératrice."
        else:
            note = "Avec deux axes, on peut composer des idées : genre horizontalement, parenté verticalement."
        draw_multiline(self.screen, note, f["normal"], MUTED, 72, 552, 310, 5, max_lines=6)
        draw_text(self.screen, "Important", f["h3"], ORANGE, 54, 724)
        warning = "Les positions sont choisies à la main pour la pédagogie. Dans un vrai modèle, elles seraient apprises automatiquement à partir d'un corpus."
        draw_multiline(self.screen, warning, f["small"], MUTED, 54, 752, 340, 4, max_lines=5)

    def draw_canvas(self):
        f = self.fonts
        rect = self.canvas_rect()
        draw_panel(self.screen, rect, "Espace vectoriel pédagogique", f)
        plot = pygame.Rect(rect.x + 35, rect.y + 70, rect.w - 70, rect.h - 105)
        pygame.draw.rect(self.screen, (255, 253, 247), plot, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, plot, 1, border_radius=10)
        if self.show_axes: self.draw_axes(plot)
        if self.level == 0: self.draw_level1(plot)
        elif self.level == 1:
            if self.show_square: self.draw_level2_square(plot)
            else: self.draw_level2_synonyms(plot)
        else: self.draw_level3_family(plot)
        draw_text(self.screen, "Touches : Espace/→ niveau suivant · ← niveau précédent · S suite · V vecteurs · A axes · C carré · R réinitialiser", f["small"], MUTED, rect.x + 40, rect.bottom - 30)

    def draw_axes(self, plot):
        f = self.fonts; cx = plot.centerx; cy = plot.centery
        pygame.draw.line(self.screen, (220, 210, 190), (plot.x + 25, cy), (plot.right - 25, cy), 2)
        pygame.draw.line(self.screen, (220, 210, 190), (cx, plot.y + 25), (cx, plot.bottom - 25), 2)
        draw_arrow(self.screen, (plot.x + 30, cy), (plot.right - 30, cy), (220, 210, 190), 2)
        draw_text(self.screen, "masculin", f["small"], MUTED, plot.x + 35, cy + 10)
        draw_text(self.screen, "féminin", f["small"], MUTED, plot.right - 90, cy + 10)
        if self.level == 2:
            draw_arrow(self.screen, (cx, plot.y + 30), (cx, plot.bottom - 30), (220, 210, 190), 2)
            draw_text(self.screen, "ascendance", f["small"], MUTED, cx + 8, plot.y + 34)
            draw_text(self.screen, "descendance", f["small"], MUTED, cx + 8, plot.bottom - 50)
        else:
            draw_text(self.screen, "mots neutres", f["small"], MUTED, cx + 8, cy + 10)

    def word_style(self, kind):
        if kind == "male": return BLUE_LIGHT, BLUE
        if kind == "female": return ORANGE_LIGHT, ORANGE
        return YELLOW_LIGHT, YELLOW

    def draw_word(self, label, x, y, kind="neutral", big=False):
        f = self.fonts
        sx, sy = self.to_screen(x, y)
        font = f["h3"] if not big else f["big"]
        w = max(72, font.size(label)[0] + 26)
        h = 38 if not big else 46
        rect = pygame.Rect(sx - w // 2, sy - h // 2, w, h)
        bg, border = self.word_style(kind)
        pygame.draw.rect(self.screen, bg, rect, border_radius=10)
        pygame.draw.rect(self.screen, border, rect, 2, border_radius=10)
        draw_centered_text(self.screen, label, font, TEXT, rect.center)
        return rect

    def draw_vector_between(self, a, b, color=ORANGE, width=3, offset_y=0):
        ax, ay = a.center; bx, by = b.center
        start = (ax + (10 if bx > ax else -10), ay + offset_y)
        end = (bx - (10 if bx > ax else -10), by + offset_y)
        draw_arrow(self.screen, start, end, color, width)

    def draw_level1(self, plot):
        groups = [
            ["Homme", "Femme"],
            ["Mari", "Femme (épouse)", "Époux", "Épouse"],
            ["Père", "Mère"],
            ["Roi", "Reine"],
            ["Paysan", "Paysanne", "Laboureur", "Lavandière"],
            ["Acteur", "Actrice", "Il", "Elle"],
            ["Médecin", "Personne", "Ils", "Les villageois", "Enfant"],
        ]

        visible_labels = set()
        for group in groups[: self.reveal_step + 1]:
            visible_labels.update(group)

        rects = {}
        for label, x, y, kind in LEVEL1_WORDS:
            if label in visible_labels:
                rects[label] = self.draw_word(label, x, y, kind)

        if self.show_vectors:
            for a, b in LEVEL1_PAIRS:
                if a in rects and b in rects:
                    self.draw_vector_between(rects[a], rects[b], ORANGE, 2)

        f = self.fonts
        draw_text(self.screen, "Même direction : masculin → féminin", f["h3"], ORANGE, plot.x + 28, plot.y + 22)
        draw_text(self.screen, f"Étape d'apparition : {self.reveal_step + 1} / {len(groups)}", f["small"], MUTED, plot.x + 28, plot.y + 50)

    def draw_level2_synonyms(self, plot):
        visible_pairs = LEVEL2_PAIRS[: self.reveal_step + 1]

        rects = {}
        for male, female, x1, y1, x2, y2 in visible_pairs:
            rects[male] = self.draw_word(male, x1, y1, "male", big=(male == "Roi"))
            rects[female] = self.draw_word(female, x2, y2, "female", big=(female == "Reine"))

        if self.show_vectors:
            for male, female, *_ in visible_pairs:
                self.draw_vector_between(rects[male], rects[female], ORANGE, 3)

        f = self.fonts
        draw_text(self.screen, "Roi → Reine : même déplacement que Empereur → Impératrice", f["h3"], ORANGE, plot.x + 28, plot.y + 22)
        draw_text(self.screen, f"Familles de synonymes affichées : {len(visible_pairs)} / {len(LEVEL2_PAIRS)}", f["small"], MUTED, plot.x + 28, plot.y + 50)

    def draw_level2_square(self, plot):
        order = [
            "Roi",
            "Reine",
            "Époux de la Reine",
            "Épouse du Roi",
        ]

        visible = set(order[: self.reveal_step + 1])
        rects = {}
        for label, x, y in LEVEL2_SQUARE:
            if label not in visible:
                continue
            kind = "male" if x < 0 else "female"
            rects[label] = self.draw_word(label, x, y, kind, big=(label in ["Roi", "Reine"]))

        if self.show_vectors:
            if "Roi" in rects and "Reine" in rects:
                self.draw_vector_between(rects["Roi"], rects["Reine"], ORANGE, 4)
            if "Époux de la Reine" in rects and "Épouse du Roi" in rects:
                self.draw_vector_between(rects["Époux de la Reine"], rects["Épouse du Roi"], ORANGE, 4)
            if "Roi" in rects and "Époux de la Reine" in rects:
                draw_arrow(self.screen, (rects["Roi"].centerx, rects["Roi"].bottom + 8), (rects["Époux de la Reine"].centerx, rects["Époux de la Reine"].top - 8), GREEN, 3)
            if "Reine" in rects and "Épouse du Roi" in rects:
                draw_arrow(self.screen, (rects["Reine"].centerx, rects["Reine"].bottom + 8), (rects["Épouse du Roi"].centerx, rects["Épouse du Roi"].top - 8), GREEN, 3)

        f = self.fonts
        draw_text(self.screen, "Carré relationnel : le vecteur de genre reste parallèle", f["h3"], GREEN, plot.x + 28, plot.y + 22)
        draw_text(self.screen, f"Coins affichés : {min(self.reveal_step + 1, len(order))} / {len(order)}", f["small"], MUTED, plot.x + 28, plot.y + 50)

    def draw_level3_family(self, plot):
        groups = [
            ["Roi", "Reine"],
            ["Grand-père paternel", "Grand-mère paternelle"],
            ["Grand-père maternel", "Grand-mère maternelle"],
            ["Enfant"],
            ["Fils", "Fille"],
            ["Prince", "Princesse"],
        ]

        visible_labels = set()
        for group in groups[: self.reveal_step + 1]:
            visible_labels.update(group)

        rects = {}
        for label, x, y, kind in LEVEL3_WORDS:
            if label not in visible_labels:
                continue
            big = label in ["Roi", "Reine", "Prince", "Princesse"]
            rects[label] = self.draw_word(label, x, y, kind, big=big)

        if self.show_vectors:
            for a, b in GENRE_PAIRS_L3:
                if a in rects and b in rects:
                    self.draw_vector_between(rects[a], rects[b], ORANGE, 2)

            for a, b in PARENT_ARROWS_L3:
                if a in rects and b in rects:
                    start = (rects[a].centerx, rects[a].bottom + 6)
                    end = (rects[b].centerx, rects[b].top - 6)
                    draw_arrow(self.screen, start, end, GREEN, 2)

            # Dernier carré pédagogique : Princesse = Fille de la Reine.
            if "Reine" in rects and "Princesse" in rects:
                draw_arrow(self.screen, (rects["Reine"].centerx + 18, rects["Reine"].bottom + 10), (rects["Princesse"].centerx + 18, rects["Princesse"].top - 10), PURPLE, 4)
            if "Fille" in rects and "Princesse" in rects:
                draw_arrow(self.screen, (rects["Fille"].centerx, rects["Fille"].bottom + 6), (rects["Princesse"].centerx, rects["Princesse"].top - 6), PURPLE, 3)

        f = self.fonts
        draw_text(self.screen, "Deux dimensions : genre en orange, parenté en vert", f["h3"], GREEN, plot.x + 28, plot.y + 22)
        draw_text(self.screen, f"Groupe affiché : {min(self.reveal_step + 1, len(groups))} / {len(groups)}", f["small"], MUTED, plot.x + 28, plot.y + 50)

        if self.reveal_step >= 5:
            draw_text(self.screen, "Princesse ≈ fille + relation avec la reine", f["h3"], PURPLE, plot.x + 28, plot.y + 76)

    def draw(self):
        self.buttons = {}
        self.screen.fill(BG)
        self.draw_header(); self.draw_buttons(); self.draw_left_panel(); self.draw_canvas()
        pygame.display.flip()

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                if event.type == pygame.KEYDOWN:
                    self.handle_key(event)
            self.draw()


if __name__ == "__main__":
    WordVectorDemo().run()
