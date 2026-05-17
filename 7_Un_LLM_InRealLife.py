import math
import random
import sys
from dataclasses import dataclass

import pygame

# ============================================================
# Atelier_Post_Final.py
# Décomposition pédagogique réaliste d'un LLM moderne
#
# Objectif :
#   Illustrer le workflow actuel d'un LLM de type ChatGPT / Mistral :
#   chat template -> tokenisation -> IDs -> embeddings -> position/RoPE
#   -> blocs Transformer -> logits -> probabilités -> sampling
#   -> génération auto-régressive -> décodage texte.
#
# Installation :
#   pip install pygame
#
# Lancement :
#   python Atelier_Post_Final.py
# ============================================================

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
RED = (216, 59, 45)
RED_LIGHT = (255, 235, 232)
DARK = (35, 35, 35)
GRAY = (228, 224, 215)
LIGHT_PANEL = (255, 253, 247)

# -----------------------------------------------------------------
# Exemple pédagogique fixe
# -----------------------------------------------------------------

USER_PROMPT = "Explique pourquoi le ciel est bleu."
SYSTEM_MESSAGE = "Tu es un assistant utile. Réponds en français, simplement."
DEVELOPER_MESSAGE = "Sois pédagogique et concis."
HISTORY_MESSAGE = "L'utilisateur prépare un atelier sur l'IA."

CHAT_TEMPLATE = [
    ("system", SYSTEM_MESSAGE),
    ("developer", DEVELOPER_MESSAGE),
    ("history", HISTORY_MESSAGE),
    ("user", USER_PROMPT),
]

TOKENS = [
    {"txt": "<|system|>", "id": 100001, "role": "system", "vec": (-0.85, 0.55)},
    {"txt": "Tu", "id": 3541, "role": "system", "vec": (-0.74, 0.45)},
    {"txt": "assistant", "id": 9210, "role": "system", "vec": (-0.60, 0.42)},
    {"txt": "français", "id": 18420, "role": "system", "vec": (-0.46, 0.34)},

    {"txt": "<|developer|>", "id": 100002, "role": "developer", "vec": (-0.32, 0.24)},
    {"txt": "pédagogique", "id": 42120, "role": "developer", "vec": (-0.20, 0.18)},
    {"txt": "concis", "id": 12560, "role": "developer", "vec": (-0.10, 0.12)},

    {"txt": "<|history|>", "id": 100003, "role": "history", "vec": (0.02, 0.02)},
    {"txt": "atelier", "id": 14880, "role": "history", "vec": (0.12, 0.04)},
    {"txt": "IA", "id": 8321, "role": "history", "vec": (0.22, 0.02)},

    {"txt": "<|user|>", "id": 100004, "role": "user", "vec": (0.32, -0.08)},
    {"txt": "Explique", "id": 27130, "role": "user", "vec": (0.42, -0.14)},
    {"txt": "pourquoi", "id": 1841, "role": "user", "vec": (0.50, -0.22)},
    {"txt": "le", "id": 312, "role": "user", "vec": (0.58, -0.28)},
    {"txt": "ciel", "id": 6042, "role": "user", "vec": (0.66, -0.20)},
    {"txt": "est", "id": 768, "role": "user", "vec": (0.76, -0.30)},
    {"txt": "bleu", "id": 9365, "role": "user", "vec": (0.86, -0.18)},
    {"txt": ".", "id": 13, "role": "user", "vec": (0.94, -0.28)},
]

ATTENTION_FOCUS = "bleu"
ATTENTION_WEIGHTS = {
    "ciel": 0.34,
    "bleu": 0.24,
    "pourquoi": 0.14,
    "Explique": 0.10,
    "français": 0.08,
    "pédagogique": 0.06,
    "atelier": 0.04,
}

LOGITS = [
    ("Le", 6.4),
    ("Parce", 5.9),
    ("La", 4.1),
    ("Un", 2.7),
    ("Bleu", 2.1),
    ("<fin>", 0.8),
]

NEXT_TOKEN_PROBS = [
    ("Le", 0.39),
    ("Parce", 0.31),
    ("La", 0.14),
    ("Un", 0.08),
    ("Bleu", 0.05),
    ("<fin>", 0.03),
]

GENERATION = [
    ("Le", "Le premier token choisi lance la réponse."),
    ("ciel", "Le token généré devient une partie du nouveau contexte."),
    ("est", "Le modèle réévalue les probabilités à chaque étape."),
    ("bleu", "La phrase se construit progressivement."),
    ("car", "Le modèle continue tant qu'il n'a pas produit de fin."),
    ("la", "Chaque nouveau token dépend de tous les précédents."),
    ("lumière", "Le sens apparent émerge de la prédiction séquentielle."),
    ("du", "Le modèle ne sort pas toute la phrase d'un coup."),
    ("Soleil", "Le vocabulaire possible contient des milliers de tokens."),
    ("est", "Les logits sont recalculés à chaque tour."),
    ("diffusée", "La réponse reste guidée par le prompt et le contexte."),
    (".", "Un token de ponctuation peut terminer la phrase."),
]

STAGES = [
    {
        "short": "Template",
        "title": "1. Chat template",
        "subtitle": "Le prompt utilisateur est intégré dans une entrée structurée.",
        "concept": "Un LLM de chat ne reçoit pas seulement le texte écrit par l'utilisateur. Il reçoit une séquence structurée avec message système, règles développeur, historique et message utilisateur.",
        "remember": "Le vrai prompt du modèle est plus large que ce que voit l'utilisateur.",
    },
    {
        "short": "Tokens",
        "title": "2. Tokenisation",
        "subtitle": "Le texte est découpé en tokens.",
        "concept": "Le modèle ne lit pas directement des mots. Il manipule des tokens : morceaux de mots, mots, ponctuation, ou balises spéciales comme <|user|>.",
        "remember": "Tokeniser, c'est découper le texte en unités calculables.",
    },
    {
        "short": "IDs",
        "title": "3. Identifiants numériques",
        "subtitle": "Chaque token devient un numéro.",
        "concept": "Chaque token est remplacé par un identifiant dans le vocabulaire du modèle. Ces IDs sont ensuite utilisés pour récupérer des vecteurs dans une table d'embeddings.",
        "remember": "Le modèle ne voit pas des lettres : il voit d'abord des nombres.",
    },
    {
        "short": "Embeddings",
        "title": "4. Embeddings",
        "subtitle": "Chaque token devient un vecteur dense.",
        "concept": "La table d'embeddings transforme les IDs de tokens en vecteurs numériques. Ces vecteurs portent une information statistique apprise pendant l'entraînement.",
        "remember": "Un embedding est une position mathématique dans un espace de sens.",
    },
    {
        "short": "Position",
        "title": "5. Position / RoPE",
        "subtitle": "Le modèle ajoute l'ordre des tokens.",
        "concept": "Les modèles récents utilisent souvent des encodages positionnels comme RoPE. L'idée : modifier les vecteurs pour que l'ordre des tokens soit pris en compte.",
        "remember": "Même avec les mêmes tokens, l'ordre change le sens.",
    },
    {
        "short": "Attention",
        "title": "6. Attention causale",
        "subtitle": "Chaque token regarde les tokens précédents utiles.",
        "concept": "Dans un modèle auto-régressif, un token ne peut regarder que le passé. L'attention calcule dynamiquement quels tokens précédents sont importants.",
        "remember": "Les poids d'attention ne sont pas écrits à l'avance : ils dépendent du contexte.",
    },
    {
        "short": "Blocs",
        "title": "7. Blocs Transformer",
        "subtitle": "Attention + MLP + résidus + normalisation.",
        "concept": "Un LLM empile de nombreux blocs Transformer. Chaque bloc mélange self-attention, réseau feed-forward, connexions résiduelles et normalisation.",
        "remember": "Le cœur du LLM est une pile de transformations de vecteurs.",
    },
    {
        "short": "Logits",
        "title": "8. Logits",
        "subtitle": "Le modèle calcule un score pour chaque token possible.",
        "concept": "À la fin du réseau, le modèle produit des scores appelés logits. Il y en a un pour chaque token du vocabulaire.",
        "remember": "Avant les probabilités, le modèle produit des scores bruts.",
    },
    {
        "short": "Sampling",
        "title": "9. Probabilités et sampling",
        "subtitle": "Les logits deviennent des probabilités, puis un token est choisi.",
        "concept": "Softmax transforme les logits en probabilités. Ensuite, le modèle peut choisir le plus probable ou tirer au sort parmi les bons candidats selon température, top-k ou top-p.",
        "remember": "Deux réponses différentes peuvent venir du même prompt à cause du sampling.",
    },
    {
        "short": "Boucle",
        "title": "10. Génération auto-régressive",
        "subtitle": "Le token choisi est ajouté, puis le modèle recommence.",
        "concept": "Le LLM prédit un token, l'ajoute au contexte, puis prédit le suivant. La réponse se construit progressivement jusqu'à un token de fin ou une limite.",
        "remember": "Un LLM écrit token par token, pas phrase par phrase.",
    },
    {
        "short": "Texte",
        "title": "11. Décodage final",
        "subtitle": "Les tokens générés sont reconvertis en texte lisible.",
        "concept": "La dernière étape convertit les IDs/tokens générés en texte. Le résultat semble naturel, mais il est issu d'une succession de prédictions.",
        "remember": "Le texte final est le décodage d'une suite de tokens choisis.",
    },
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


def draw_right_text(screen, text, font, color, x, y):
    surf = font.render(text, True, color)
    rect = surf.get_rect(topright=(x, y))
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


def softmax(logits):
    m = max(v for _, v in logits)
    exps = [(tok, math.exp(v - m)) for tok, v in logits]
    total = sum(v for _, v in exps)
    return [(tok, v / total) for tok, v in exps]


class AtelierPostFinal:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Atelier Post-Final — Architecture réelle d'un LLM")
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
            "mono_small": pygame.font.SysFont("consolas", 12),
            "mono_big": pygame.font.SysFont("consolas", 20, bold=True),
        }

        self.stage = 0
        self.reveal = 0
        self.generated_count = 0
        self.show_attention = True
        self.show_kv_cache = True
        self.temperature = 0.8
        self.buttons = {}

    @property
    def info(self):
        return STAGES[self.stage]

    def canvas_rect(self):
        return pygame.Rect(470, 170, 995, 695)

    def next_stage(self):
        self.stage = (self.stage + 1) % len(STAGES)
        self.reveal = 0
        if self.stage != 9:
            self.generated_count = 0

    def prev_stage(self):
        self.stage = (self.stage - 1) % len(STAGES)
        self.reveal = 0

    def next_reveal(self):
        self.reveal = (self.reveal + 1) % 7

    def generate_next(self):
        self.stage = 9
        self.generated_count = min(len(GENERATION), self.generated_count + 1)

    def reset(self):
        self.stage = 0
        self.reveal = 0
        self.generated_count = 0
        self.show_attention = True
        self.show_kv_cache = True

    def handle_click(self, pos):
        for key, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                if key == "prev": self.prev_stage()
                elif key == "next": self.next_stage()
                elif key == "suite": self.next_reveal()
                elif key == "generate": self.generate_next()
                elif key == "attention": self.show_attention = not self.show_attention
                elif key == "cache": self.show_kv_cache = not self.show_kv_cache
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
        elif event.key == pygame.K_a:
            self.show_attention = not self.show_attention
        elif event.key == pygame.K_k:
            self.show_kv_cache = not self.show_kv_cache
        elif event.key == pygame.K_r:
            self.reset()
        elif event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

    # ---------------------------------------------------------
    # Layout général
    # ---------------------------------------------------------

    def draw_header(self):
        f = self.fonts
        draw_text(self.screen, "ATELIER POST-FINAL — DÉCOMPOSITION RÉALISTE D'UN LLM", f["title"], TEXT, 34, 22)
        draw_text(
            self.screen,
            "Chat template, tokens, embeddings, RoPE, attention causale, blocs Transformer, logits, sampling et génération.",
            f["normal"], MUTED, 34, 56
        )

        stats = [
            ("Étape", f"{self.stage + 1} / {len(STAGES)}"),
            ("Famille", "GPT / Mistral"),
            ("Mécanisme", "next token"),
            ("Mode", "pédagogique"),
        ]
        x = 790
        widths = [110, 150, 150, 135]
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
        self.buttons["attention"] = Button(
            pygame.Rect(778, y, 145, 42),
            "Attention ON" if self.show_attention else "Attention OFF",
            "blue" if self.show_attention else "light"
        )
        self.buttons["cache"] = Button(
            pygame.Rect(938, y, 130, 42),
            "KV cache ON" if self.show_kv_cache else "KV cache OFF",
            "blue" if self.show_kv_cache else "light"
        )
        self.buttons["reset"] = Button(pygame.Rect(1320, y, 145, 42), "Réinitialiser", "dark")

        for key in ["prev", "next", "suite", "generate", "attention", "cache", "reset"]:
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

        draw_text(self.screen, "Limite de la démo", f["h3"], ORANGE, 54, 725)
        warning = "Les nombres, positions et probabilités sont fabriqués pour être lisibles. Le pipeline, lui, suit la logique réelle d'un LLM moderne."
        draw_multiline(self.screen, warning, f["small"], MUTED, 54, 752, 340, 4, max_lines=5)

    def draw_canvas(self):
        f = self.fonts
        rect = self.canvas_rect()
        draw_panel(self.screen, rect, "Vue principale", f)

        area = pygame.Rect(rect.x + 35, rect.y + 65, rect.w - 70, rect.h - 105)
        pygame.draw.rect(self.screen, LIGHT_PANEL, area, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, area, 1, border_radius=10)

        if self.stage == 0:
            self.draw_chat_template(area)
        elif self.stage == 1:
            self.draw_tokenization(area)
        elif self.stage == 2:
            self.draw_token_ids(area)
        elif self.stage == 3:
            self.draw_embeddings(area, rope=False)
        elif self.stage == 4:
            self.draw_embeddings(area, rope=True)
        elif self.stage == 5:
            self.draw_attention(area)
        elif self.stage == 6:
            self.draw_transformer_blocks(area)
        elif self.stage == 7:
            self.draw_logits(area)
        elif self.stage == 8:
            self.draw_sampling(area)
        elif self.stage == 9:
            self.draw_generation_loop(area)
        else:
            self.draw_decoding(area)

        draw_text(
            self.screen,
            "Touches : Espace/→ étape suivante · ← étape précédente · S suite · G générer · A attention · K cache · R réinitialiser",
            f["small"], MUTED, rect.x + 40, rect.bottom - 30
        )

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

    # ---------------------------------------------------------
    # Outils graphiques internes
    # ---------------------------------------------------------

    def role_style(self, role):
        if role == "system":
            return PURPLE_LIGHT, PURPLE
        if role == "developer":
            return YELLOW_LIGHT, YELLOW
        if role == "history":
            return GREEN_LIGHT, GREEN
        if role == "user":
            return BLUE_LIGHT, BLUE
        if role == "generated":
            return ORANGE_LIGHT, ORANGE
        return PANEL, BORDER

    def draw_token_box(self, token, rect, big=False):
        f = self.fonts
        bg, border = self.role_style(token["role"])
        pygame.draw.rect(self.screen, bg, rect, border_radius=9)
        pygame.draw.rect(self.screen, border, rect, 2, border_radius=9)
        draw_centered_text(self.screen, token["txt"], f["mono_big"] if big else f["mono"], TEXT, rect.center)
        return rect

    def token_rects(self, area):
        rects = {}
        x = area.x + 25
        y = area.y + 82
        for i, token in enumerate(TOKENS):
            w = max(58, self.fonts["mono"].size(token["txt"])[0] + 22)
            if x + w > area.right - 25:
                x = area.x + 25
                y += 54
            r = pygame.Rect(x, y, w, 38)
            rects[token["txt"]] = r
            x += w + 8
        return rects

    def vector_plot(self, area):
        return pygame.Rect(area.x + 85, area.y + 115, area.w - 170, area.h - 210)

    def to_vec_screen(self, area, x, y):
        plot = self.vector_plot(area)
        sx = plot.x + plot.w / 2 + x * plot.w * 0.44
        sy = plot.y + plot.h / 2 - y * plot.h * 0.44
        return int(sx), int(sy)

    # ---------------------------------------------------------
    # Scènes
    # ---------------------------------------------------------

    def draw_chat_template(self, area):
        f = self.fonts
        draw_text(self.screen, "Entrée réelle d'un modèle de chat", f["h3"], ORANGE, area.x + 35, area.y + 32)

        y = area.y + 78
        for role, message in CHAT_TEMPLATE:
            bg, border = self.role_style(role)
            rect = pygame.Rect(area.x + 55, y, area.w - 110, 88)
            pygame.draw.rect(self.screen, bg, rect, border_radius=12)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=12)
            draw_text(self.screen, f"[{role}]", f["h3"], border, rect.x + 18, rect.y + 12)
            draw_text(self.screen, message, f["mono"], TEXT, rect.x + 18, rect.y + 45)
            y += 108

        draw_arrow(self.screen, (area.centerx, y - 10), (area.centerx, y + 35), ORANGE, 4)
        merged = pygame.Rect(area.x + 230, y + 45, area.w - 460, 70)
        pygame.draw.rect(self.screen, YELLOW_LIGHT, merged, border_radius=12)
        pygame.draw.rect(self.screen, YELLOW, merged, 2, border_radius=12)
        draw_centered_text(self.screen, "Séquence unique envoyée au modèle", f["h2"], TEXT, merged.center)

    def draw_tokenization(self, area):
        f = self.fonts
        draw_text(self.screen, "Découpage en tokens, incluant les balises de rôle", f["h3"], ORANGE, area.x + 35, area.y + 32)

        rects = self.token_rects(area)
        max_visible = min(len(TOKENS), 5 + self.reveal * 3)
        for token in TOKENS[:max_visible]:
            self.draw_token_box(token, rects[token["txt"]])

        draw_text(self.screen, f"{max_visible} / {len(TOKENS)} tokens visibles", f["small"], MUTED, area.x + 35, area.bottom - 55)

        legend = [
            ("system", "consigne système"),
            ("developer", "règle développeur"),
            ("history", "historique"),
            ("user", "message utilisateur"),
        ]
        x = area.x + 35
        y = area.bottom - 95
        for role, label in legend:
            bg, border = self.role_style(role)
            pygame.draw.rect(self.screen, bg, pygame.Rect(x, y, 22, 14), border_radius=4)
            pygame.draw.rect(self.screen, border, pygame.Rect(x, y, 22, 14), 1, border_radius=4)
            draw_text(self.screen, label, f["small"], MUTED, x + 28, y - 2)
            x += 190

    def draw_token_ids(self, area):
        f = self.fonts
        draw_text(self.screen, "Chaque token correspond à un ID du vocabulaire", f["h3"], ORANGE, area.x + 35, area.y + 32)

        cols = [area.x + 80, area.x + 340, area.x + 620]
        y0 = area.y + 80
        row_h = 42

        draw_text(self.screen, "TOKEN", f["h3"], BLUE, cols[0], y0)
        draw_text(self.screen, "ID", f["h3"], BLUE, cols[1], y0)
        draw_text(self.screen, "RÔLE", f["h3"], BLUE, cols[2], y0)

        max_visible = min(len(TOKENS), 7 + self.reveal * 3)
        for i, token in enumerate(TOKENS[:max_visible]):
            y = y0 + 36 + i * row_h
            if y > area.bottom - 80:
                break
            bg, border = self.role_style(token["role"])
            pygame.draw.rect(self.screen, bg, pygame.Rect(cols[0] - 12, y - 8, 190, 30), border_radius=8)
            pygame.draw.rect(self.screen, border, pygame.Rect(cols[0] - 12, y - 8, 190, 30), 1, border_radius=8)
            draw_text(self.screen, token["txt"], f["mono"], TEXT, cols[0], y)
            draw_text(self.screen, str(token["id"]), f["mono"], TEXT, cols[1], y)
            draw_text(self.screen, token["role"], f["mono"], TEXT, cols[2], y)

        note = "L'ID sert d'adresse dans une table d'embeddings : token_id → vecteur."
        draw_text(self.screen, note, f["normal"], MUTED, area.x + 80, area.bottom - 50)

    def draw_embeddings(self, area, rope=False):
        f = self.fonts
        plot = self.vector_plot(area)

        pygame.draw.rect(self.screen, PANEL, plot, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, plot, 1, border_radius=10)
        pygame.draw.line(self.screen, GRAY, (plot.x + 24, plot.centery), (plot.right - 24, plot.centery), 2)
        pygame.draw.line(self.screen, GRAY, (plot.centerx, plot.y + 24), (plot.centerx, plot.bottom - 24), 2)

        title = "Embeddings : les IDs deviennent des vecteurs"
        if rope:
            title = "Position / RoPE simplifiée : chaque vecteur est légèrement tourné selon sa position"
        draw_text(self.screen, title, f["h3"], ORANGE, area.x + 35, area.y + 32)

        visible = TOKENS if self.reveal >= 2 else TOKENS[-8:]
        previous = None

        for idx, token in enumerate(visible):
            x, y = token["vec"]
            if rope:
                # Représentation pédagogique de RoPE :
                # on applique une petite rotation dépendante de la position.
                angle = 0.09 * idx
                xr = x * math.cos(angle) - y * math.sin(angle)
                yr = x * math.sin(angle) + y * math.cos(angle)
                x, y = xr, yr

            sx, sy = self.to_vec_screen(area, x, y)
            radius = 14 if token["txt"] in ["ciel", "bleu", "Explique"] else 10
            bg, border = self.role_style(token["role"])
            pygame.draw.circle(self.screen, bg, (sx, sy), radius)
            pygame.draw.circle(self.screen, border, (sx, sy), radius, 2)
            draw_centered_text(self.screen, token["txt"], f["tiny"], TEXT, (sx, sy - 22))

            if rope and previous:
                draw_arrow(self.screen, previous, (sx, sy), (200, 190, 170), 1)
            previous = (sx, sy)

        if rope:
            msg = "RoPE réel agit dans l'espace vectoriel de grande dimension ; ici, on le montre comme une rotation visible."
        else:
            msg = "Dans un vrai modèle, les vecteurs ont souvent des milliers de dimensions, pas seulement 2."
        draw_text(self.screen, msg, f["small"], MUTED, area.x + 35, area.bottom - 50)

    def draw_attention(self, area):
        f = self.fonts
        draw_text(self.screen, "Self-attention causale : le token courant regarde le passé", f["h3"], ORANGE, area.x + 35, area.y + 32)

        rects = self.token_rects(area)
        for token in TOKENS:
            self.draw_token_box(token, rects[token["txt"]], big=(token["txt"] == ATTENTION_FOCUS))

        if self.show_attention and ATTENTION_FOCUS in rects:
            source = rects[ATTENTION_FOCUS].center
            for label, weight in ATTENTION_WEIGHTS.items():
                if label not in rects:
                    continue
                target = rects[label].center
                color = PURPLE if label == "ciel" else ORANGE
                width = 1 + int(weight * 10)
                draw_arrow(self.screen, source, target, color, width)
                mx = int((source[0] + target[0]) / 2)
                my = int((source[1] + target[1]) / 2) - 14
                draw_text(self.screen, f"{weight:.2f}", f["small"], color, mx, my)

        # Causal mask
        mask_rect = pygame.Rect(area.x + 70, area.bottom - 170, area.w - 140, 95)
        pygame.draw.rect(self.screen, PANEL, mask_rect, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, mask_rect, 1, border_radius=12)
        draw_text(self.screen, "Masque causal", f["h3"], TEXT, mask_rect.x + 18, mask_rect.y + 14)
        draw_multiline(
            self.screen,
            "Pour générer le prochain token, le modèle peut utiliser les tokens précédents, mais pas les tokens futurs. C'est ce qui permet l'écriture auto-régressive.",
            f["normal"], MUTED, mask_rect.x + 18, mask_rect.y + 42, mask_rect.w - 36
        )

    def draw_transformer_blocks(self, area):
        f = self.fonts
        draw_text(self.screen, "Pile de blocs Transformer", f["h3"], ORANGE, area.x + 35, area.y + 32)

        x = area.x + 80
        y = area.y + 95
        w = 175
        h = 390
        gap = 35

        block_names = ["Embedding", "Bloc 1", "Bloc 2", "Bloc 3", "...", "Bloc N"]
        colors = [
            (YELLOW_LIGHT, YELLOW),
            (BLUE_LIGHT, BLUE),
            (BLUE_LIGHT, BLUE),
            (BLUE_LIGHT, BLUE),
            (PANEL, BORDER),
            (ORANGE_LIGHT, ORANGE),
        ]

        previous_center = None
        for i, name in enumerate(block_names):
            rect = pygame.Rect(x + i * (w + gap), y, w, h)
            bg, border = colors[i]
            pygame.draw.rect(self.screen, bg, rect, border_radius=16)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=16)
            draw_centered_text(self.screen, name, f["h2"], TEXT, (rect.centerx, rect.y + 28))

            if name.startswith("Bloc"):
                parts = [
                    ("Norm", PURPLE),
                    ("Attention", ORANGE),
                    ("Résidu +", GREEN),
                    ("Norm", PURPLE),
                    ("MLP", BLUE),
                    ("Résidu +", GREEN),
                ]
                yy = rect.y + 70
                for label, color in parts:
                    part = pygame.Rect(rect.x + 22, yy, rect.w - 44, 34)
                    pygame.draw.rect(self.screen, PANEL, part, border_radius=8)
                    pygame.draw.rect(self.screen, color, part, 2, border_radius=8)
                    draw_centered_text(self.screen, label, f["small"], TEXT, part.center)
                    yy += 48
            elif name == "...":
                draw_centered_text(self.screen, "× dizaines", f["big"], MUTED, rect.center)
            else:
                for n in range(6):
                    cy = rect.y + 88 + n * 42
                    pygame.draw.circle(self.screen, border, (rect.centerx, cy), 11)
                    pygame.draw.circle(self.screen, PANEL, (rect.centerx, cy), 8)

            if previous_center:
                draw_arrow(self.screen, (previous_center[0] + 86, previous_center[1]), (rect.x - 8, rect.centery), DARK, 3)
            previous_center = rect.center

        if self.show_kv_cache:
            cache = pygame.Rect(area.x + 230, area.bottom - 130, area.w - 460, 70)
            pygame.draw.rect(self.screen, GREEN_LIGHT, cache, border_radius=12)
            pygame.draw.rect(self.screen, GREEN, cache, 2, border_radius=12)
            draw_text(self.screen, "KV cache", f["h3"], GREEN, cache.x + 18, cache.y + 13)
            draw_text(self.screen, "Pendant la génération, on réutilise les clés/valeurs déjà calculées pour aller plus vite.", f["normal"], MUTED, cache.x + 18, cache.y + 40)

    def draw_logits(self, area):
        f = self.fonts
        draw_text(self.screen, "Sortie brute du réseau : les logits", f["h3"], ORANGE, area.x + 35, area.y + 32)

        prompt_rect = pygame.Rect(area.x + 55, area.y + 78, area.w - 110, 72)
        pygame.draw.rect(self.screen, BLUE_LIGHT, prompt_rect, border_radius=12)
        pygame.draw.rect(self.screen, BLUE, prompt_rect, 2, border_radius=12)
        draw_text(self.screen, "Contexte courant", f["h3"], BLUE, prompt_rect.x + 18, prompt_rect.y + 12)
        draw_text(self.screen, USER_PROMPT, f["mono_big"], TEXT, prompt_rect.x + 18, prompt_rect.y + 40)

        y = area.y + 200
        max_logit = max(v for _, v in LOGITS)
        for token, logit in LOGITS:
            draw_text(self.screen, f"« {token} »", f["mono_big"], TEXT, area.x + 135, y - 4)
            draw_progress(self.screen, pygame.Rect(area.x + 260, y, 480, 18), logit / max_logit, BLUE)
            draw_text(self.screen, f"logit = {logit:.1f}", f["h3"], TEXT, area.x + 770, y - 4)
            y += 58

        note = "Un logit est un score brut. Il n'est pas encore une probabilité."
        draw_text(self.screen, note, f["normal"], MUTED, area.x + 135, area.bottom - 70)

    def draw_sampling(self, area):
        f = self.fonts
        draw_text(self.screen, "Softmax + sampling : choisir le prochain token", f["h3"], ORANGE, area.x + 35, area.y + 32)

        left = pygame.Rect(area.x + 70, area.y + 82, 380, 430)
        right = pygame.Rect(area.x + 520, area.y + 82, 380, 430)

        pygame.draw.rect(self.screen, PANEL, left, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, left, 1, border_radius=12)
        draw_text(self.screen, "Probabilités après softmax", f["h3"], BLUE, left.x + 20, left.y + 18)

        y = left.y + 65
        for token, prob in NEXT_TOKEN_PROBS:
            draw_text(self.screen, token, f["mono"], TEXT, left.x + 25, y - 3)
            draw_progress(self.screen, pygame.Rect(left.x + 115, y, 180, 16), prob, ORANGE if token == "Le" else BLUE)
            draw_text(self.screen, f"{prob:.0%}", f["small"], TEXT, left.x + 310, y - 1)
            y += 48

        pygame.draw.rect(self.screen, PANEL, right, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, right, 1, border_radius=12)
        draw_text(self.screen, "Stratégies de choix", f["h3"], PURPLE, right.x + 20, right.y + 18)

        strategies = [
            ("Greedy", "prendre le token le plus probable"),
            ("Température", "plus bas = plus déterministe"),
            ("Top-k", "tirer parmi les k meilleurs"),
            ("Top-p", "tirer dans la masse probable"),
        ]

        yy = right.y + 72
        for title, desc in strategies:
            box = pygame.Rect(right.x + 25, yy, right.w - 50, 58)
            pygame.draw.rect(self.screen, PURPLE_LIGHT, box, border_radius=9)
            pygame.draw.rect(self.screen, PURPLE, box, 1, border_radius=9)
            draw_text(self.screen, title, f["h3"], TEXT, box.x + 14, box.y + 10)
            draw_text(self.screen, desc, f["small"], MUTED, box.x + 14, box.y + 34)
            yy += 78

        result = pygame.Rect(area.x + 250, area.bottom - 115, area.w - 500, 62)
        pygame.draw.rect(self.screen, GREEN_LIGHT, result, border_radius=12)
        pygame.draw.rect(self.screen, GREEN, result, 2, border_radius=12)
        draw_centered_text(self.screen, "Token choisi dans cette démo : « Le »", f["big"], TEXT, result.center)

    def draw_generation_loop(self, area):
        f = self.fonts
        draw_text(self.screen, "Boucle auto-régressive", f["h3"], ORANGE, area.x + 35, area.y + 32)

        generated_tokens = GENERATION[:self.generated_count]
        sentence = " ".join(tok for tok, _ in generated_tokens)
        if sentence:
            text = sentence
        else:
            text = "▌"

        output = pygame.Rect(area.x + 55, area.y + 85, area.w - 110, 95)
        pygame.draw.rect(self.screen, BLUE_LIGHT, output, border_radius=12)
        pygame.draw.rect(self.screen, BLUE, output, 2, border_radius=12)
        draw_text(self.screen, "Réponse en cours", f["h3"], BLUE, output.x + 18, output.y + 14)
        draw_text(self.screen, text + (" ▌" if self.generated_count < len(GENERATION) else ""), f["mono_big"], TEXT, output.x + 18, output.y + 50)

        y = area.y + 215
        max_rows = 6
        start = max(0, self.generated_count - max_rows)
        visible = GENERATION[start:self.generated_count]

        for idx, (token, msg) in enumerate(visible):
            global_idx = start + idx + 1
            row = pygame.Rect(area.x + 75, y, area.w - 150, 62)
            pygame.draw.rect(self.screen, GREEN_LIGHT, row, border_radius=10)
            pygame.draw.rect(self.screen, GREEN, row, 2, border_radius=10)
            draw_text(self.screen, f"Tour {global_idx}", f["h3"], GREEN, row.x + 16, row.y + 10)
            draw_text(self.screen, f"token = « {token} »", f["mono"], TEXT, row.x + 115, row.y + 11)
            draw_text(self.screen, msg, f["small"], MUTED, row.x + 16, row.y + 36)
            y += 76

        if self.generated_count == 0:
            hint = pygame.Rect(area.x + 190, area.y + 260, area.w - 380, 90)
            pygame.draw.rect(self.screen, YELLOW_LIGHT, hint, border_radius=12)
            pygame.draw.rect(self.screen, YELLOW, hint, 2, border_radius=12)
            draw_centered_text(self.screen, "Clique sur « Générer token » ou appuie sur G", f["big"], TEXT, hint.center)

        if self.show_kv_cache:
            cache = pygame.Rect(area.x + 250, area.bottom - 95, area.w - 500, 45)
            pygame.draw.rect(self.screen, GREEN_LIGHT, cache, border_radius=10)
            pygame.draw.rect(self.screen, GREEN, cache, 1, border_radius=10)
            draw_centered_text(self.screen, "KV cache : les anciens calculs sont conservés pour accélérer la suite", f["small"], TEXT, cache.center)

    def draw_decoding(self, area):
        f = self.fonts
        draw_text(self.screen, "Décodage final : tokens → texte lisible", f["h3"], ORANGE, area.x + 35, area.y + 32)

        tokens = [tok for tok, _ in GENERATION]
        ids = [2401, 6042, 768, 9365, 842, 312, 12056, 417, 9150, 768, 33082, 13]

        y = area.y + 92
        for i, (token, token_id) in enumerate(zip(tokens, ids)):
            x = area.x + 70 + (i % 4) * 215
            yy = y + (i // 4) * 85
            box = pygame.Rect(x, yy, 180, 55)
            pygame.draw.rect(self.screen, ORANGE_LIGHT, box, border_radius=10)
            pygame.draw.rect(self.screen, ORANGE, box, 2, border_radius=10)
            draw_centered_text(self.screen, f"{token_id} → {token}", f["mono"], TEXT, box.center)

        draw_arrow(self.screen, (area.centerx, area.y + 370), (area.centerx, area.y + 430), ORANGE, 4)

        final = pygame.Rect(area.x + 130, area.y + 450, area.w - 260, 90)
        pygame.draw.rect(self.screen, BLUE_LIGHT, final, border_radius=14)
        pygame.draw.rect(self.screen, BLUE, final, 2, border_radius=14)
        draw_centered_text(self.screen, "Le ciel est bleu car la lumière du Soleil est diffusée.", f["mono_big"], TEXT, final.center)

        note = "Ce texte final cache toute la mécanique précédente : embeddings, attention, logits, sampling et boucle de génération."
        draw_multiline(self.screen, note, f["normal"], MUTED, area.x + 150, area.bottom - 75, area.w - 300)

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
    AtelierPostFinal().run()
