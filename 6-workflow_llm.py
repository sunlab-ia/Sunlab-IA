import math
import sys
from dataclasses import dataclass

import pygame

# ============================================================
# Atelier final IA — Workflow pédagogique d'un LLM
# Prompt -> contexte -> tokens -> embeddings -> positions
# -> attention -> transformer simplifié -> prédiction -> texte
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
DARK = (35, 35, 35)
LIGHT_GRAY = (232, 228, 218)

USER_PROMPT = "Le roi parle à la reine car elle"
SYSTEM_CONTEXT = "Réponds en français, avec une phrase courte."
HISTORY_CONTEXT = "Contexte : on parle d'un royaume."

TOKENS = [
    {"txt": "<système>", "kind": "system", "vec": (-0.85, -0.55)},
    {"txt": "Réponds", "kind": "system", "vec": (-0.70, -0.42)},
    {"txt": "français", "kind": "system", "vec": (-0.50, -0.40)},
    {"txt": "<contexte>", "kind": "context", "vec": (-0.34, -0.18)},
    {"txt": "royaume", "kind": "context", "vec": (-0.18, -0.05)},
    {"txt": "<user>", "kind": "user", "vec": (0.00, 0.05)},
    {"txt": "Le", "kind": "user", "vec": (0.12, 0.12)},
    {"txt": "roi", "kind": "user", "vec": (-0.38, 0.22)},
    {"txt": "parle", "kind": "user", "vec": (0.02, 0.45)},
    {"txt": "à", "kind": "user", "vec": (0.16, 0.38)},
    {"txt": "la", "kind": "user", "vec": (0.34, 0.13)},
    {"txt": "reine", "kind": "user", "vec": (0.48, 0.24)},
    {"txt": "car", "kind": "user", "vec": (0.60, 0.04)},
    {"txt": "elle", "kind": "target", "vec": (0.70, 0.30)},
]

ATTENTION_FROM = "elle"
ATTENTION_WEIGHTS = {"reine": 0.50, "elle": 0.18, "roi": 0.13, "parle": 0.09, "royaume": 0.06, "français": 0.04}
NEXT_TOKEN_PROBS = [("est", 0.44), ("semble", 0.20), ("répond", 0.13), ("sourit", 0.10), ("parle", 0.07), ("?", 0.06)]
GENERATION_STEPS = [("est", "Le modèle prédit le prochain token."), ("inquiète", "Le texte généré est réinjecté comme nouveau contexte."), (".", "La génération s'arrête quand la phrase est complète.")]

LEVELS = [
    ("1. Prompt + contexte", "Le message utilisateur n'arrive pas seul.", "Un LLM reçoit le prompt, mais aussi des informations invisibles : consignes système, historique de conversation, langue attendue, filtres, outils disponibles, etc."),
    ("2. Tokenisation", "Le texte est découpé en petits morceaux.", "Le modèle ne lit pas directement des mots comme nous. Il reçoit une suite de tokens : mots, morceaux de mots, signes, balises de rôle."),
    ("3. Embeddings", "Chaque token devient un vecteur.", "Chaque token est converti en un vecteur numérique. Dans ce démonstrateur, on place les vecteurs dans un espace 2D simplifié."),
    ("4. Position", "L'ordre des tokens est ajouté.", "Sans information de position, le modèle verrait les mêmes tokens dans le désordre. On ajoute donc une information de position à chaque vecteur."),
    ("5. Attention", "Les tokens importants s'allument.", "Le modèle calcule dynamiquement quels tokens doivent être regardés par les autres. Ici, « elle » regarde surtout « reine »."),
    ("6. Transformer simplifié", "Les vecteurs sont transformés couche après couche.", "Attention + petits réseaux neuronaux transforment progressivement les vecteurs. Le contexte, le sens et les relations sont mélangés."),
    ("7. Probabilités", "Le modèle propose le prochain token.", "Le LLM ne sort pas directement une phrase entière. Il calcule une distribution de probabilités pour le prochain token."),
    ("8. Génération", "La réponse apparaît token par token.", "Le token choisi est ajouté au texte, puis le modèle recommence. La réponse finale est reconstruite en langage lisible."),
]

NOTES = [
    "Le prompt visible n'est qu'une partie de l'entrée réelle du modèle.",
    "Tokeniser, ce n'est pas comprendre : c'est découper le texte en unités manipulables.",
    "Un embedding transforme un identifiant de token en point dans un espace de sens.",
    "La position permet de distinguer « le chien mord » de « mord le chien ».",
    "L'attention calcule des poids d'importance dépendants du contexte.",
    "Un Transformer mélange attention et réseaux neuronaux pour enrichir les vecteurs.",
    "La sortie est une probabilité pour chaque token possible du vocabulaire.",
    "La réponse est fabriquée progressivement, token après token.",
]

@dataclass
class Button:
    rect: pygame.Rect
    text: str
    kind: str = "light"
    def draw(self, screen, fonts):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        if self.kind == "dark": bg, fg, border = DARK, (255, 255, 255), DARK
        elif self.kind == "orange": bg, fg, border = ORANGE, TEXT, DARK
        elif self.kind == "blue": bg, fg, border = BLUE, (255, 255, 255), BLUE
        elif self.kind == "green": bg, fg, border = GREEN, (255, 255, 255), GREEN
        elif self.kind == "purple": bg, fg, border = PURPLE, (255, 255, 255), PURPLE
        else: bg, fg, border = PANEL, TEXT, BORDER
        if hover: bg = tuple(max(0, c - 10) for c in bg)
        pygame.draw.rect(screen, bg, self.rect, border_radius=8)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=8)
        draw_centered_text(screen, self.text, fonts["button"], fg, self.rect.center)

def draw_text(screen, text, font, color, x, y):
    surf = font.render(text, True, color); screen.blit(surf, (x, y)); return surf.get_rect(topleft=(x, y))

def draw_centered_text(screen, text, font, color, center):
    surf = font.render(text, True, color); rect = surf.get_rect(center=center); screen.blit(surf, rect); return rect

def draw_multiline(screen, text, font, color, x, y, max_width, line_gap=5, max_lines=None):
    words, line, yy, lines = text.split(" "), "", y, 0
    for word in words:
        test = line + word + " "
        if font.size(test)[0] > max_width and line:
            draw_text(screen, line.rstrip(), font, color, x, yy)
            yy += font.get_height() + line_gap; lines += 1
            if max_lines is not None and lines >= max_lines:
                draw_text(screen, "…", font, color, x, yy); return yy
            line = word + " "
        else: line = test
    if line: draw_text(screen, line.rstrip(), font, color, x, yy)
    return yy

def draw_panel(screen, rect, title, fonts):
    pygame.draw.rect(screen, PANEL, rect, border_radius=14); pygame.draw.rect(screen, BORDER, rect, 2, border_radius=14)
    draw_text(screen, title, fonts["h2"], TEXT, rect.x + 18, rect.y + 16)

def draw_arrow(screen, start, end, color, width=3):
    width = max(1, int(width)); pygame.draw.line(screen, color, start, end, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0]); length, spread = 12, 0.45
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    pygame.draw.polygon(screen, color, [end, p1, p2])

def draw_progress(screen, rect, value, color):
    pygame.draw.rect(screen, (238, 235, 226), rect, border_radius=999)
    inner = pygame.Rect(rect.x, rect.y, max(2, int(rect.w * value)), rect.h)
    pygame.draw.rect(screen, color, inner, border_radius=999); pygame.draw.rect(screen, BORDER, rect, 1, border_radius=999)

class LLMSynthesisDemo:
    def __init__(self):
        pygame.init(); pygame.display.set_caption("Atelier final IA — Workflow d'un LLM")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE); self.clock = pygame.time.Clock()
        self.fonts = {"title": pygame.font.SysFont("arial", 27, bold=True), "h2": pygame.font.SysFont("arial", 18, bold=True), "h3": pygame.font.SysFont("arial", 15, bold=True), "normal": pygame.font.SysFont("arial", 14), "small": pygame.font.SysFont("arial", 12), "button": pygame.font.SysFont("arial", 13, bold=True), "big": pygame.font.SysFont("arial", 24, bold=True), "mono": pygame.font.SysFont("consolas", 15), "mono_big": pygame.font.SysFont("consolas", 20, bold=True)}
        self.level = 0; self.reveal_step = 0; self.generated_count = 0; self.show_attention = True; self.buttons = {}
    def info(self): return LEVELS[self.level]
    def canvas_rect(self): return pygame.Rect(470, 170, 995, 695)
    def next_level(self):
        self.level = (self.level + 1) % len(LEVELS); self.reveal_step = 0
        if self.level != 7: self.generated_count = 0
    def prev_level(self): self.level = (self.level - 1) % len(LEVELS); self.reveal_step = 0
    def next_reveal(self): self.reveal_step = (self.reveal_step + 1) % 7
    def generate_next(self):
        if self.level != 7: self.level = 7
        else: self.generated_count = min(len(GENERATION_STEPS), self.generated_count + 1)
    def reset(self): self.level = 0; self.reveal_step = 0; self.generated_count = 0; self.show_attention = True
    def handle_click(self, pos):
        for key, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                {"prev": self.prev_level, "next": self.next_level, "suite": self.next_reveal, "generate": self.generate_next, "reset": self.reset}.get(key, lambda: None)()
                if key == "attention": self.show_attention = not self.show_attention
                return
    def handle_key(self, event):
        if event.key in [pygame.K_RIGHT, pygame.K_SPACE]: self.next_level()
        elif event.key == pygame.K_LEFT: self.prev_level()
        elif event.key == pygame.K_s: self.next_reveal()
        elif event.key == pygame.K_g: self.generate_next()
        elif event.key == pygame.K_a: self.show_attention = not self.show_attention
        elif event.key == pygame.K_r: self.reset()
        elif event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
    def draw_header(self):
        f = self.fonts
        draw_text(self.screen, "ATELIER FINAL — COMMENT UN LLM TRAITE UN PROMPT", f["title"], TEXT, 34, 22)
        draw_text(self.screen, "Un démonstrateur de synthèse : tokens, vecteurs, attention, réseau neuronal, probabilités et génération.", f["normal"], MUTED, 34, 56)
        stats = [("Étape", f"{self.level + 1} / {len(LEVELS)}"), ("Architecture", "Transformer"), ("Sortie", "token suivant"), ("But", "pédagogique")]
        x = 790
        for (label, value), w in zip(stats, [110, 150, 150, 135]):
            draw_centered_text(self.screen, label, f["small"], MUTED, (x + w // 2, 25))
            rect = pygame.Rect(x, 42, w, 30); pygame.draw.rect(self.screen, (255, 241, 198), rect, border_radius=8); pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)
            draw_centered_text(self.screen, value, f["button"], TEXT, rect.center); x += w + 14
    def draw_buttons(self):
        f, y = self.fonts, 92
        self.buttons = {"prev": Button(pygame.Rect(34, y, 130, 42), "← Étape"), "next": Button(pygame.Rect(174, y, 230, 42), "▶ Étape suivante", "orange"), "suite": Button(pygame.Rect(470, y, 120, 42), "SUITE", "green"), "generate": Button(pygame.Rect(605, y, 165, 42), "Générer token", "purple"), "attention": Button(pygame.Rect(785, y, 155, 42), "Attention ON" if self.show_attention else "Attention OFF", "blue" if self.show_attention else "light"), "reset": Button(pygame.Rect(1320, y, 145, 42), "Réinitialiser", "dark")}
        for k in ["prev", "next", "suite", "generate", "attention", "reset"]: self.buttons[k].draw(self.screen, f)
    def draw_left_panel(self):
        f = self.fonts; rect = pygame.Rect(34, 170, 410, 695); title, subtitle, concept = self.info()
        draw_panel(self.screen, rect, "Concept couvert", f); draw_text(self.screen, title, f["big"], TEXT, 54, 214); draw_text(self.screen, subtitle, f["h3"], BLUE, 54, 252)
        draw_multiline(self.screen, concept, f["normal"], MUTED, 54, 292, 350, 5, 10)
        box = pygame.Rect(54, 495, 350, 205); pygame.draw.rect(self.screen, (255,253,247), box, border_radius=10); pygame.draw.rect(self.screen, BORDER, box, 1, border_radius=10)
        draw_text(self.screen, "À retenir", f["h3"], TEXT, 72, 515); draw_multiline(self.screen, NOTES[self.level], f["normal"], MUTED, 72, 548, 310, 5, 7)
        draw_text(self.screen, "Important", f["h3"], ORANGE, 54, 734); draw_multiline(self.screen, "Ce programme simplifie volontairement un vrai LLM. Il montre les idées, pas les milliards de paramètres réels.", f["small"], MUTED, 54, 762, 340, 4, 5)
    def token_color(self, kind):
        return {"system": (PURPLE_LIGHT, PURPLE), "context": (GREEN_LIGHT, GREEN), "target": (ORANGE_LIGHT, ORANGE)}.get(kind, (BLUE_LIGHT, BLUE))
    def draw_token(self, token, rect, big=False):
        bg, border = self.token_color(token["kind"]); pygame.draw.rect(self.screen, bg, rect, border_radius=9); pygame.draw.rect(self.screen, border, rect, 2, border_radius=9)
        draw_centered_text(self.screen, token["txt"], self.fonts["mono_big"] if big else self.fonts["mono"], TEXT, rect.center); return rect
    def token_positions(self, area):
        pos, x, y = {}, area.x + 25, area.y + 88
        for token in TOKENS:
            w = max(75, self.fonts["mono"].size(token["txt"])[0] + 24)
            if x + w > area.right - 25: x = area.x + 25; y += 56
            rect = pygame.Rect(x, y, w, 38); pos[token["txt"]] = rect; x += w + 10
        return pos
    def draw_prompt_context(self, area):
        f = self.fonts; boxes = [("Consigne système", SYSTEM_CONTEXT, PURPLE_LIGHT, PURPLE), ("Historique / contexte", HISTORY_CONTEXT, GREEN_LIGHT, GREEN), ("Prompt utilisateur", USER_PROMPT, BLUE_LIGHT, BLUE)]
        y = area.y + 85
        for title, content, bg, border in boxes:
            rect = pygame.Rect(area.x + 45, y, area.w - 90, 92); pygame.draw.rect(self.screen, bg, rect, border_radius=12); pygame.draw.rect(self.screen, border, rect, 2, border_radius=12)
            draw_text(self.screen, title, f["h3"], border, rect.x + 18, rect.y + 14); draw_text(self.screen, content, f["mono_big"], TEXT, rect.x + 18, rect.y + 46); y += 120
        draw_arrow(self.screen, (area.centerx, y - 15), (area.centerx, y + 45), ORANGE, 4)
        merged = pygame.Rect(area.x + 210, y + 55, area.w - 420, 72); pygame.draw.rect(self.screen, YELLOW_LIGHT, merged, border_radius=12); pygame.draw.rect(self.screen, YELLOW, merged, 2, border_radius=12); draw_centered_text(self.screen, "Entrée complète du modèle", f["h2"], TEXT, merged.center)
    def draw_tokenization(self, area):
        f = self.fonts; positions = self.token_positions(area); draw_text(self.screen, "Suite de tokens transmise au modèle", f["h3"], ORANGE, area.x + 35, area.y + 35)
        max_visible = min(len(TOKENS), 3 + self.reveal_step * 3)
        for token in TOKENS[:max_visible]: self.draw_token(token, positions[token["txt"]])
        draw_text(self.screen, f"{max_visible} / {len(TOKENS)} tokens affichés", f["small"], MUTED, area.x + 35, area.bottom - 45)
    def to_vector_screen(self, area, x, y):
        plot = pygame.Rect(area.x + 90, area.y + 120, area.w - 180, area.h - 210)
        return int(plot.x + plot.w / 2 + x * plot.w * 0.44), int(plot.y + plot.h / 2 - y * plot.h * 0.44), plot
    def draw_embeddings(self, area, with_position=False):
        f = self.fonts; _, _, plot = self.to_vector_screen(area, 0, 0)
        pygame.draw.rect(self.screen, (255,253,247), plot, border_radius=10); pygame.draw.rect(self.screen, BORDER, plot, 1, border_radius=10)
        pygame.draw.line(self.screen, LIGHT_GRAY, (plot.x+25, plot.centery), (plot.right-25, plot.centery), 2); pygame.draw.line(self.screen, LIGHT_GRAY, (plot.centerx, plot.y+25), (plot.centerx, plot.bottom-25), 2)
        draw_text(self.screen, "Embedding + position : le rang du token modifie légèrement son vecteur" if with_position else "Embedding : chaque token devient un vecteur 2D simplifié", f["h3"], ORANGE, area.x+35, area.y+35)
        previous = None
        for i, token in enumerate(TOKENS if self.reveal_step >= 2 else TOKENS[-8:]):
            x, y = token["vec"]
            if with_position: y += (i % 6 - 2.5) * 0.035
            sx, sy, _ = self.to_vector_screen(area, x, y); radius = 15 if token["txt"] in ["roi", "reine", "elle"] else 11
            bg, border = self.token_color(token["kind"]); pygame.draw.circle(self.screen, bg, (sx, sy), radius); pygame.draw.circle(self.screen, border, (sx, sy), radius, 2); draw_centered_text(self.screen, token["txt"], f["small"], TEXT, (sx, sy-25))
            if with_position and previous: draw_arrow(self.screen, previous, (sx, sy), (200,190,170), 1)
            previous = (sx, sy)
        draw_text(self.screen, "Dans un vrai LLM, ces vecteurs ont des centaines ou milliers de dimensions.", f["small"], MUTED, area.x+35, area.bottom-45)
    def draw_attention(self, area):
        f = self.fonts; positions = self.token_positions(area); draw_text(self.screen, "Attention calculée pour le token « elle »", f["h3"], ORANGE, area.x+35, area.y+35)
        for token in TOKENS: self.draw_token(token, positions[token["txt"]], big=(token["txt"] == "elle"))
        if self.show_attention:
            source = positions[ATTENTION_FROM].center
            for label, weight in ATTENTION_WEIGHTS.items():
                if label in positions:
                    target = positions[label].center; color = PURPLE if label == "reine" else ORANGE
                    draw_arrow(self.screen, source, target, color, 1 + int(weight * 8)); draw_text(self.screen, f"{weight:.2f}", f["small"], color, int((source[0]+target[0])/2), int((source[1]+target[1])/2)-12)
        draw_multiline(self.screen, "Les poids d'attention ne sont pas fixes : ils sont recalculés pour chaque token, dans chaque couche.", f["normal"], MUTED, area.x+35, area.bottom-70, area.w-70)
    def draw_transformer(self, area):
        f = self.fonts; draw_text(self.screen, "Transformer simplifié : attention + réseau neuronal", f["h3"], ORANGE, area.x+35, area.y+35)
        x0, y0, layer_w, gap = area.x+80, area.y+130, 175, 45; layers = [("Tokens", BLUE_LIGHT, BLUE), ("Embeddings", YELLOW_LIGHT, YELLOW), ("Attention", PURPLE_LIGHT, PURPLE), ("Réseau", GREEN_LIGHT, GREEN), ("Vecteur final", ORANGE_LIGHT, ORANGE)]
        centers = []
        for i, (name, bg, border) in enumerate(layers):
            rect = pygame.Rect(x0+i*(layer_w+gap), y0, layer_w, 330); pygame.draw.rect(self.screen, bg, rect, border_radius=16); pygame.draw.rect(self.screen, border, rect, 2, border_radius=16); draw_centered_text(self.screen, name, f["h2"], TEXT, (rect.centerx, rect.y+30)); centers.append(rect.center)
            for n in range(5): pygame.draw.circle(self.screen, border, (rect.centerx, rect.y+85+n*45), 13); pygame.draw.circle(self.screen, PANEL, (rect.centerx, rect.y+85+n*45), 10)
            if i > 0: draw_arrow(self.screen, (centers[i-1][0]+85, centers[i-1][1]), (rect.x-8, rect.centery), DARK, 3)
        draw_text(self.screen, "Ce n'est pas une recherche dans une base : c'est une transformation progressive des vecteurs.", f["normal"], MUTED, area.x+80, area.bottom-105)
    def draw_probabilities(self, area):
        f = self.fonts; draw_text(self.screen, "Distribution de probabilités du prochain token", f["h3"], ORANGE, area.x+35, area.y+35)
        prompt_rect = pygame.Rect(area.x+55, area.y+90, area.w-110, 76); pygame.draw.rect(self.screen, BLUE_LIGHT, prompt_rect, border_radius=12); pygame.draw.rect(self.screen, BLUE, prompt_rect, 2, border_radius=12)
        draw_text(self.screen, "Texte courant :", f["h3"], BLUE, prompt_rect.x+18, prompt_rect.y+12); draw_text(self.screen, USER_PROMPT, f["mono_big"], TEXT, prompt_rect.x+18, prompt_rect.y+42)
        y = area.y+210
        for token, prob in NEXT_TOKEN_PROBS:
            label_rect = pygame.Rect(area.x+105, y-5, 115, 30); pygame.draw.rect(self.screen, PANEL, label_rect, border_radius=8); pygame.draw.rect(self.screen, BORDER, label_rect, 1, border_radius=8); draw_centered_text(self.screen, token, f["mono_big"], TEXT, label_rect.center)
            draw_progress(self.screen, pygame.Rect(area.x+245, y, 500, 18), prob, ORANGE if token == "est" else BLUE); draw_text(self.screen, f"{prob:.0%}", f["h3"], TEXT, area.x+770, y-2); y += 55
        decision = pygame.Rect(area.x+255, area.bottom-130, 475, 70); pygame.draw.rect(self.screen, GREEN_LIGHT, decision, border_radius=12); pygame.draw.rect(self.screen, GREEN, decision, 2, border_radius=12); draw_centered_text(self.screen, "Token choisi dans cette démo : « est »", f["big"], TEXT, decision.center)
    def draw_generation(self, area):
        f = self.fonts; draw_text(self.screen, "Génération auto-régressive : un token après l'autre", f["h3"], ORANGE, area.x+35, area.y+35)
        generated = " ".join(tok for tok, _ in GENERATION_STEPS[:self.generated_count]); final_text = USER_PROMPT + (" " + generated if generated else "")
        rect = pygame.Rect(area.x+55, area.y+95, area.w-110, 110); pygame.draw.rect(self.screen, BLUE_LIGHT, rect, border_radius=12); pygame.draw.rect(self.screen, BLUE, rect, 2, border_radius=12); draw_text(self.screen, "Réponse en cours :", f["h3"], BLUE, rect.x+18, rect.y+16); draw_text(self.screen, final_text + (" ▌" if self.generated_count < len(GENERATION_STEPS) else ""), f["mono_big"], TEXT, rect.x+18, rect.y+55)
        y = area.y+250
        for i, (token, msg) in enumerate(GENERATION_STEPS):
            active = i < self.generated_count; bg = GREEN_LIGHT if active else (248,246,240); border = GREEN if active else BORDER
            step = pygame.Rect(area.x+80, y, area.w-160, 70); pygame.draw.rect(self.screen, bg, step, border_radius=12); pygame.draw.rect(self.screen, border, step, 2, border_radius=12); draw_text(self.screen, f"Token {i+1}", f["h3"], border, step.x+18, step.y+14); draw_text(self.screen, f"« {token} »", f["mono_big"], TEXT, step.x+125, step.y+13); draw_text(self.screen, msg, f["normal"], MUTED, step.x+18, step.y+42); y += 90
        if self.generated_count >= len(GENERATION_STEPS):
            done = pygame.Rect(area.x+270, area.bottom-100, 450, 55); pygame.draw.rect(self.screen, YELLOW_LIGHT, done, border_radius=12); pygame.draw.rect(self.screen, YELLOW, done, 2, border_radius=12); draw_centered_text(self.screen, "Phrase finale : Le roi parle à la reine car elle est inquiète.", f["h3"], TEXT, done.center)
    def draw_canvas(self):
        f = self.fonts; rect = self.canvas_rect(); draw_panel(self.screen, rect, "Workflow visible du LLM", f); area = pygame.Rect(rect.x+35, rect.y+65, rect.w-70, rect.h-105); pygame.draw.rect(self.screen, (255,253,247), area, border_radius=10); pygame.draw.rect(self.screen, BORDER, area, 1, border_radius=10)
        [self.draw_prompt_context, self.draw_tokenization, lambda a: self.draw_embeddings(a, False), lambda a: self.draw_embeddings(a, True), self.draw_attention, self.draw_transformer, self.draw_probabilities, self.draw_generation][self.level](area)
        draw_text(self.screen, "Touches : Espace/→ étape suivante · ← étape précédente · S suite · G générer · A attention · R réinitialiser", f["small"], MUTED, rect.x+40, rect.bottom-30)
    def draw_footer_pipeline(self):
        f = self.fonts; labels = ["Prompt", "Tokens", "Vecteurs", "Position", "Attention", "Transformer", "Probabilités", "Texte"]; x, y, w = 48, 880, 155
        for i, label in enumerate(labels):
            r = pygame.Rect(x, y, w, 32); pygame.draw.rect(self.screen, ORANGE_LIGHT if i == self.level else PANEL, r, border_radius=999); pygame.draw.rect(self.screen, ORANGE if i == self.level else BORDER, r, 2, border_radius=999); draw_centered_text(self.screen, label, f["small"], TEXT, r.center)
            if i < len(labels)-1: draw_arrow(self.screen, (x+w+4, y+16), (x+w+30, y+16), BORDER, 2)
            x += w + 36
    def draw(self):
        self.screen.fill(BG); self.draw_header(); self.draw_buttons(); self.draw_left_panel(); self.draw_canvas(); self.draw_footer_pipeline(); pygame.display.flip()
    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: self.handle_click(event.pos)
                if event.type == pygame.KEYDOWN: self.handle_key(event)
            self.draw()

if __name__ == "__main__":
    LLMSynthesisDemo().run()
