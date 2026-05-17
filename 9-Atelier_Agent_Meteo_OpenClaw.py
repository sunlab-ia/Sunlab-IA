import math
import sys
from dataclasses import dataclass

import pygame

# ============================================================
# Atelier_Agent_Meteo_OpenClaw.py
# Démonstrateur pédagogique : Agent IA météo autonome
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

USER_OBJECTIVE = (
    "Surveille la météo là où je suis, toutes les heures, "
    "et envoie-moi une alerte WhatsApp en cas d'orage, "
    "d'inondation ou de canicule."
)

WHATSAPP_ALERT = (
    "ALERTE MÉTÉO : risque d'orage violent détecté dans votre zone. "
    "Restez à l'abri et surveillez les consignes locales."
)

STAGES = [
    {"short": "Objectif", "title": "1. Objectif utilisateur", "subtitle": "L'utilisateur décrit ce qu'il veut obtenir.", "concept": "L'utilisateur ne demande pas une simple réponse. Il donne une mission : surveiller la météo locale et envoyer une alerte si un risque apparaît.", "remember": "Un agent part d'un objectif à atteindre, pas seulement d'une question."},
    {"short": "Analyse", "title": "2. Analyse par le LLM", "subtitle": "Le LLM décompose la mission.", "concept": "Le LLM identifie les sous-tâches : obtenir la localisation, interroger la météo, analyser les risques, envoyer une alerte, répéter toutes les heures.", "remember": "Le LLM sert ici à comprendre, planifier et choisir les actions."},
    {"short": "Outils", "title": "3. Passerelles nécessaires", "subtitle": "L'agent doit disposer de connecteurs.", "concept": "Pour agir, l'agent a besoin d'outils : GPS/localisation, API météo, WhatsApp, planificateur horaire et journal d'activité.", "remember": "Sans outil, le LLM reste un chatbot ; avec des outils contrôlés, il devient agent."},
    {"short": "Permissions", "title": "4. Permissions utilisateur", "subtitle": "L'agent demande les droits nécessaires.", "concept": "Avant d'accéder à la localisation ou d'envoyer des messages, l'agent doit obtenir une autorisation claire de l'utilisateur.", "remember": "Les permissions sont indispensables dès que l'agent agit dans le monde réel."},
    {"short": "Config", "title": "5. Configuration agentique", "subtitle": "Le runtime prépare les passerelles.", "concept": "Le runtime agentique configure les connecteurs, vérifie les clés d'API, prépare le destinataire WhatsApp et fixe les règles météo.", "remember": "La configuration est la phase où l'agent se donne les moyens d'agir."},
    {"short": "Workflow", "title": "6. Workflow automatique", "subtitle": "L'agent construit une tâche récurrente.", "concept": "Le workflow est une boucle planifiée : toutes les heures, récupérer la localisation, lire la météo, évaluer les risques, alerter si nécessaire.", "remember": "Un agent peut exécuter une routine automatiquement selon un calendrier."},
    {"short": "Cycle", "title": "7. Cycle horaire", "subtitle": "Penser → agir → observer → décider.", "concept": "Chaque heure, l'agent exécute la boucle. Les observations météo deviennent le nouveau contexte de décision.", "remember": "Le résultat d'un outil devient l'entrée du raisonnement suivant."},
    {"short": "Risque", "title": "8. Détection du risque", "subtitle": "Le LLM interprète les données météo.", "concept": "L'agent compare la météo aux règles : orage, pluie intense, vent violent, canicule, risque d'inondation. Il décide s'il faut alerter.", "remember": "Le LLM peut aider à interpréter, mais les règles critiques doivent être explicites."},
    {"short": "Alerte", "title": "9. Alerte WhatsApp", "subtitle": "Action réelle vers l'utilisateur.", "concept": "Si un danger est détecté et que l'autorisation existe, le runtime envoie l'alerte via WhatsApp. Sinon, il bloque ou demande confirmation.", "remember": "Une action externe doit être contrôlée et journalisée."},
    {"short": "Sécurité", "title": "10. Garde-fous", "subtitle": "Limiter les risques d'un agent autonome.", "concept": "L'agent doit éviter le spam, protéger la localisation, gérer les erreurs API, refuser les demandes dangereuses et permettre l'arrêt du workflow.", "remember": "Un bon agent est utile parce qu'il est encadré."},
    {"short": "Bilan", "title": "11. Agent météo minimal", "subtitle": "Synthèse de l'atelier final.", "concept": "Le public voit comment on passe d'un LLM qui répond à un agent qui configure ses outils, surveille une situation et agit sous contrôle.", "remember": "LLM + outils + permissions + scheduler + logs = agent autonome contrôlé."},
]

TOOLS = [
    {"name": "GPS", "desc": "localisation approximative", "status": "à autoriser", "bg": BLUE_LIGHT, "color": BLUE},
    {"name": "Météo", "desc": "API prévisions horaires", "status": "à connecter", "bg": GREEN_LIGHT, "color": GREEN},
    {"name": "WhatsApp", "desc": "envoi d'alerte", "status": "à autoriser", "bg": ORANGE_LIGHT, "color": ORANGE},
    {"name": "Scheduler", "desc": "toutes les heures", "status": "à créer", "bg": PURPLE_LIGHT, "color": PURPLE},
    {"name": "Logs", "desc": "journal d'activité", "status": "actif", "bg": YELLOW_LIGHT, "color": YELLOW},
]

PERMISSIONS = [
    ("Localisation approximative", "utile pour météo locale", "Autoriser"),
    ("Accès API météo", "récupérer prévisions horaires", "Autoriser"),
    ("Message WhatsApp", "envoyer seulement en cas de risque", "Autoriser avec limite"),
    ("Exécution horaire", "surveillance automatique", "Autoriser"),
    ("Historique local", "journaliser sans données sensibles", "Autoriser"),
]

WORKFLOW_STEPS = [
    ("1", "Déclenchement horaire", "scheduler.every(1 hour)"),
    ("2", "Récupération localisation", "get_location(approximate=True)"),
    ("3", "Lecture météo", "get_weather(location, hourly=True)"),
    ("4", "Analyse des risques", "classify_weather_risk(data)"),
    ("5", "Décision", "if risk >= threshold"),
    ("6", "Alerte / log", "send_whatsapp() or write_log()"),
]

CYCLE_STEPS = [
    ("PENSER", "Je dois vérifier si c'est l'heure du contrôle météo.", BLUE_LIGHT, BLUE),
    ("AGIR", "tool_call: get_location(approximate=True)", ORANGE_LIGHT, ORANGE),
    ("OBSERVER", "Localisation : zone approximative de l'utilisateur.", GREEN_LIGHT, GREEN),
    ("AGIR", "tool_call: get_weather(hourly=True)", ORANGE_LIGHT, ORANGE),
    ("OBSERVER", "Prévision : orage à 16h, pluie intense, rafales.", GREEN_LIGHT, GREEN),
    ("PENSER", "Le risque dépasse le seuil d'alerte.", BLUE_LIGHT, BLUE),
    ("AGIR", "tool_call: send_whatsapp(alert_message)", ORANGE_LIGHT, ORANGE),
    ("OBSERVER", "Alerte envoyée et action journalisée.", GREEN_LIGHT, GREEN),
]

WEATHER_ROWS = [
    ("13h", "Nuageux", "faible", GREEN),
    ("14h", "Pluie modérée", "surveillance", YELLOW),
    ("15h", "Pluie intense", "risque", ORANGE),
    ("16h", "Orage violent", "alerte", RED),
    ("17h", "Rafales", "risque", ORANGE),
]

GUARDRAILS = [
    ("Pas de GPS précis", "utiliser une zone approximative", GREEN),
    ("Anti-spam", "ne pas répéter la même alerte", GREEN),
    ("Confirmation", "requise pour changer le destinataire", ORANGE),
    ("Arrêt facile", "bouton désactiver le workflow", GREEN),
    ("Logs sobres", "ne pas stocker de position précise", GREEN),
    ("Erreur API", "prévenir sans paniquer", ORANGE),
]

@dataclass
class Button:
    rect: pygame.Rect
    text: str
    kind: str = "light"

    def draw(self, screen, fonts):
        mouse = pygame.mouse.get_pos()
        hover = self.rect.collidepoint(mouse)
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

class WeatherAgentWorkshop:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Atelier Agent IA — Surveillance météo autonome")
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
            "mono": pygame.font.SysFont("consolas", 15),
            "mono_big": pygame.font.SysFont("consolas", 20, bold=True),
        }
        self.stage = 0
        self.reveal = 0
        self.cycle = 0
        self.permissions_ok = False
        self.workflow_on = False
        self.alert_sent = False
        self.buttons = {}

    @property
    def info(self): return STAGES[self.stage]
    def canvas_rect(self): return pygame.Rect(470, 170, 995, 695)

    def next_stage(self):
        self.stage = (self.stage + 1) % len(STAGES)
        self.reveal = 0
        if self.stage != 6: self.cycle = 0
    def prev_stage(self): self.stage = (self.stage - 1) % len(STAGES); self.reveal = 0
    def next_reveal(self): self.reveal = (self.reveal + 1) % 7
    def next_cycle(self):
        self.stage = 6; self.workflow_on = True
        self.cycle = min(len(CYCLE_STEPS), self.cycle + 1)
        if self.cycle >= 7: self.alert_sent = True
    def reset(self):
        self.stage = self.reveal = self.cycle = 0
        self.permissions_ok = self.workflow_on = self.alert_sent = False

    def handle_click(self, pos):
        for key, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                if key == "prev": self.prev_stage()
                elif key == "next": self.next_stage()
                elif key == "suite": self.next_reveal()
                elif key == "cycle": self.next_cycle()
                elif key == "perm": self.permissions_ok = not self.permissions_ok
                elif key == "workflow": self.workflow_on = not self.workflow_on
                elif key == "reset": self.reset()
                return

    def handle_key(self, event):
        if event.key in [pygame.K_RIGHT, pygame.K_SPACE]: self.next_stage()
        elif event.key == pygame.K_LEFT: self.prev_stage()
        elif event.key == pygame.K_s: self.next_reveal()
        elif event.key == pygame.K_c: self.next_cycle()
        elif event.key == pygame.K_p: self.permissions_ok = not self.permissions_ok
        elif event.key == pygame.K_w: self.workflow_on = not self.workflow_on
        elif event.key == pygame.K_r: self.reset()
        elif event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()

    def draw_header(self):
        f = self.fonts
        draw_text(self.screen, "ATELIER AGENT IA — SURVEILLANCE MÉTÉO AUTONOME", f["title"], TEXT, 34, 22)
        draw_text(self.screen, "Un agent type OpenClaw configure ses outils, surveille la météo et alerte l'utilisateur sous contrôle.", f["normal"], MUTED, 34, 56)
        stats = [("Permissions", "OK" if self.permissions_ok else "à valider"), ("Workflow", "actif" if self.workflow_on else "inactif"), ("Alerte", "envoyée" if self.alert_sent else "aucune"), ("Étape", f"{self.stage + 1} / {len(STAGES)}")]
        x = 790
        for label, value in stats:
            w = 130
            draw_centered_text(self.screen, label, f["small"], MUTED, (x + w // 2, 25))
            rect = pygame.Rect(x, 42, w, 30)
            pygame.draw.rect(self.screen, (255, 241, 198), rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)
            draw_centered_text(self.screen, value, f["button"], TEXT, rect.center)
            x += w + 14

    def draw_buttons(self):
        f = self.fonts; y = 92
        self.buttons["prev"] = Button(pygame.Rect(34, y, 120, 42), "← Étape")
        self.buttons["next"] = Button(pygame.Rect(164, y, 210, 42), "▶ Étape suivante", "orange")
        self.buttons["suite"] = Button(pygame.Rect(470, y, 115, 42), "SUITE", "green")
        self.buttons["cycle"] = Button(pygame.Rect(598, y, 170, 42), "Cycle horaire", "purple")
        self.buttons["perm"] = Button(pygame.Rect(785, y, 160, 42), "Permissions OK" if self.permissions_ok else "Valider droits", "green" if self.permissions_ok else "light")
        self.buttons["workflow"] = Button(pygame.Rect(960, y, 160, 42), "Workflow ON" if self.workflow_on else "Workflow OFF", "blue" if self.workflow_on else "light")
        self.buttons["reset"] = Button(pygame.Rect(1320, y, 145, 42), "Réinitialiser", "dark")
        for key in ["prev", "next", "suite", "cycle", "perm", "workflow", "reset"]: self.buttons[key].draw(self.screen, f)

    def draw_left_panel(self):
        f = self.fonts; rect = pygame.Rect(34, 170, 410, 695)
        draw_panel(self.screen, rect, "Concept couvert", f)
        draw_text(self.screen, self.info["title"], f["big"], TEXT, 54, 214)
        draw_text(self.screen, self.info["subtitle"], f["h3"], BLUE, 54, 252)
        draw_multiline(self.screen, self.info["concept"], f["normal"], MUTED, 54, 292, 350, 5, max_lines=11)
        box = pygame.Rect(54, 510, 350, 180)
        pygame.draw.rect(self.screen, LIGHT_PANEL, box, border_radius=10); pygame.draw.rect(self.screen, BORDER, box, 1, border_radius=10)
        draw_text(self.screen, "À retenir", f["h3"], TEXT, 72, 530)
        draw_multiline(self.screen, self.info["remember"], f["normal"], MUTED, 72, 562, 310, 5, max_lines=6)
        draw_text(self.screen, "Message clé", f["h3"], ORANGE, 54, 725)
        draw_multiline(self.screen, "L'agent ne doit pas agir sans cadre : permissions, limites, logs et arrêt manuel sont essentiels.", f["small"], MUTED, 54, 752, 340, 4, max_lines=5)

    def draw_component(self, rect, title, subtitle, bg, border):
        f = self.fonts
        pygame.draw.rect(self.screen, bg, rect, border_radius=14); pygame.draw.rect(self.screen, border, rect, 2, border_radius=14)
        draw_centered_text(self.screen, title, f["h2"], TEXT, (rect.centerx, rect.y + 30))
        if subtitle: draw_centered_text(self.screen, subtitle, f["small"], MUTED, (rect.centerx, rect.y + 58))

    def draw_canvas(self):
        f = self.fonts; rect = self.canvas_rect()
        draw_panel(self.screen, rect, "Vue principale", f)
        area = pygame.Rect(rect.x + 35, rect.y + 65, rect.w - 70, rect.h - 105)
        pygame.draw.rect(self.screen, LIGHT_PANEL, area, border_radius=10); pygame.draw.rect(self.screen, BORDER, area, 1, border_radius=10)
        [self.draw_objective, self.draw_analysis, self.draw_tools, self.draw_permissions, self.draw_configuration, self.draw_workflow, self.draw_cycle, self.draw_risk, self.draw_alert, self.draw_safety, self.draw_summary][self.stage](area)
        draw_text(self.screen, "Touches : Espace/→ étape suivante · ← étape précédente · S suite · C cycle · P permissions · W workflow · R réinitialiser", f["small"], MUTED, rect.x + 40, rect.bottom - 30)

    def draw_objective(self, area):
        f = self.fonts; draw_text(self.screen, "L'utilisateur exprime une mission complète", f["h3"], ORANGE, area.x + 35, area.y + 32)
        bubble = pygame.Rect(area.x + 95, area.y + 95, area.w - 190, 160)
        pygame.draw.rect(self.screen, BLUE_LIGHT, bubble, border_radius=16); pygame.draw.rect(self.screen, BLUE, bubble, 3, border_radius=16)
        draw_text(self.screen, "Utilisateur", f["h3"], BLUE, bubble.x + 22, bubble.y + 18)
        draw_multiline(self.screen, USER_OBJECTIVE, f["mono_big"], TEXT, bubble.x + 22, bubble.y + 58, bubble.w - 44, 7)
        draw_arrow(self.screen, (area.centerx, bubble.bottom + 10), (area.centerx, bubble.bottom + 70), ORANGE, 4)
        agent = pygame.Rect(area.x + 310, area.y + 365, area.w - 620, 105)
        pygame.draw.rect(self.screen, PURPLE_LIGHT, agent, border_radius=16); pygame.draw.rect(self.screen, PURPLE, agent, 3, border_radius=16)
        draw_centered_text(self.screen, "Agent météo autonome", f["big"], TEXT, (agent.centerx, agent.centery - 15))
        draw_centered_text(self.screen, "comprendre → configurer → surveiller → alerter", f["h3"], MUTED, (agent.centerx, agent.centery + 25))

    def draw_analysis(self, area):
        f = self.fonts; draw_text(self.screen, "Le LLM transforme l'objectif en sous-tâches", f["h3"], ORANGE, area.x + 35, area.y + 32)
        tasks = [("Comprendre", "surveillance météo personnelle", BLUE_LIGHT, BLUE), ("Localiser", "où est l'utilisateur ?", GREEN_LIGHT, GREEN), ("Mesurer", "quelles prévisions météo ?", PURPLE_LIGHT, PURPLE), ("Décider", "danger ou non ?", ORANGE_LIGHT, ORANGE), ("Notifier", "WhatsApp si risque", RED_LIGHT, RED), ("Automatiser", "toutes les heures", YELLOW_LIGHT, YELLOW)]
        for i, (title, desc, bg, border) in enumerate(tasks):
            self.draw_component(pygame.Rect(area.x + 85 + (i % 3) * 295, area.y + 95 + (i // 3) * 160, 250, 105), title, desc, bg, border)
        note = pygame.Rect(area.x + 190, area.bottom - 115, area.w - 380, 60)
        pygame.draw.rect(self.screen, YELLOW_LIGHT, note, border_radius=12); pygame.draw.rect(self.screen, YELLOW, note, 2, border_radius=12)
        draw_centered_text(self.screen, "Le LLM ne fait pas encore l'action : il prépare le plan.", f["h2"], TEXT, note.center)

    def draw_tools(self, area):
        f = self.fonts; draw_text(self.screen, "Passerelles nécessaires pour agir", f["h3"], ORANGE, area.x + 35, area.y + 32)
        for i, tool in enumerate(TOOLS):
            rect = pygame.Rect(area.x + 75 + (i % 3) * 300, area.y + 105 + (i // 3) * 165, 255, 108)
            configured = self.reveal >= i or self.workflow_on
            self.draw_component(rect, tool["name"], tool["desc"], tool["bg"] if configured else PANEL, tool["color"] if configured else BORDER)
            draw_centered_text(self.screen, "configuré" if configured else tool["status"], f["small"], tool["color"] if configured else MUTED, (rect.centerx, rect.bottom - 18))
        draw_text(self.screen, "Clique sur SUITE pour simuler la configuration progressive des passerelles.", f["normal"], MUTED, area.x + 190, area.bottom - 90)

    def draw_permissions(self, area):
        f = self.fonts; draw_text(self.screen, "Demande d'autorisations", f["h3"], ORANGE, area.x + 35, area.y + 32)
        y = area.y + 85
        for title, why, action in PERMISSIONS:
            rect = pygame.Rect(area.x + 90, y, area.w - 180, 68)
            pygame.draw.rect(self.screen, GREEN_LIGHT if self.permissions_ok else PANEL, rect, border_radius=12); pygame.draw.rect(self.screen, GREEN if self.permissions_ok else BORDER, rect, 2, border_radius=12)
            draw_text(self.screen, title, f["h3"], TEXT, rect.x + 20, rect.y + 11); draw_text(self.screen, why, f["normal"], MUTED, rect.x + 20, rect.y + 37)
            btn = pygame.Rect(rect.right - 170, rect.y + 18, 140, 32); pygame.draw.rect(self.screen, GREEN if self.permissions_ok else ORANGE, btn, border_radius=999)
            draw_centered_text(self.screen, "OK" if self.permissions_ok else action, f["small"], (255,255,255), btn.center)
            y += 82
        draw_text(self.screen, "Permissions accordées : l'agent peut configurer le workflow." if self.permissions_ok else "Les permissions doivent être validées avant l'automatisation.", f["h3"], GREEN if self.permissions_ok else ORANGE, area.x + 235, area.bottom - 75)

    def draw_configuration(self, area):
        f = self.fonts; draw_text(self.screen, "Phase 1 : l'agent se configure", f["h3"], ORANGE, area.x + 35, area.y + 32)
        runtime = pygame.Rect(area.x + 350, area.y + 210, 260, 130); self.draw_component(runtime, "Runtime agentique", "configuration", YELLOW_LIGHT, YELLOW)
        nodes = [("GPS", "zone approximative", area.x+75, area.y+115, BLUE_LIGHT, BLUE), ("API météo", "prévisions horaires", area.x+665, area.y+115, GREEN_LIGHT, GREEN), ("WhatsApp", "destinataire validé", area.x+75, area.y+405, ORANGE_LIGHT, ORANGE), ("Scheduler", "toutes les heures", area.x+665, area.y+405, PURPLE_LIGHT, PURPLE)]
        for title, sub, x, y, bg, border in nodes:
            rect = pygame.Rect(x, y, 220, 92); active = self.permissions_ok or self.reveal > 0
            self.draw_component(rect, title, sub, bg if active else PANEL, border if active else BORDER); draw_arrow(self.screen, rect.center, runtime.center, border if active else BORDER, 3)

    def draw_workflow(self, area):
        f = self.fonts; draw_text(self.screen, "Phase 2 : création du workflow automatique", f["h3"], ORANGE, area.x + 35, area.y + 32)
        y = area.y + 85
        for num, title, code in WORKFLOW_STEPS:
            active = self.workflow_on or self.reveal >= int(num)-1; rect = pygame.Rect(area.x + 100, y, area.w - 200, 66)
            pygame.draw.rect(self.screen, BLUE_LIGHT if active else PANEL, rect, border_radius=12); pygame.draw.rect(self.screen, BLUE if active else BORDER, rect, 2, border_radius=12)
            pygame.draw.circle(self.screen, ORANGE if active else BORDER, (rect.x + 34, rect.centery), 20)
            draw_centered_text(self.screen, num, f["h2"], (255,255,255) if active else MUTED, (rect.x + 34, rect.centery))
            draw_text(self.screen, title, f["h3"], TEXT, rect.x + 75, rect.y + 10); draw_text(self.screen, code, f["mono"], MUTED, rect.x + 75, rect.y + 36)
            y += 78
        status = pygame.Rect(area.x + 270, area.bottom - 92, area.w - 540, 48)
        pygame.draw.rect(self.screen, GREEN_LIGHT if self.workflow_on else PANEL, status, border_radius=12); pygame.draw.rect(self.screen, GREEN if self.workflow_on else BORDER, status, 2, border_radius=12)
        draw_centered_text(self.screen, "Workflow horaire actif" if self.workflow_on else "Workflow prêt mais non activé", f["h2"], TEXT, status.center)

    def draw_cycle(self, area):
        f = self.fonts; draw_text(self.screen, "Exécution horaire : penser → agir → observer", f["h3"], ORANGE, area.x + 35, area.y + 32)
        if self.cycle == 0:
            hint = pygame.Rect(area.x + 230, area.y + 250, area.w - 460, 90); pygame.draw.rect(self.screen, YELLOW_LIGHT, hint, border_radius=12); pygame.draw.rect(self.screen, YELLOW, hint, 2, border_radius=12)
            draw_centered_text(self.screen, "Clique « Cycle horaire » ou appuie sur C", f["big"], TEXT, hint.center); return
        y = area.y + 90
        for kind, msg, bg, border in CYCLE_STEPS[max(0, self.cycle-6):self.cycle]:
            rect = pygame.Rect(area.x + 80, y, area.w - 160, 62); pygame.draw.rect(self.screen, bg, rect, border_radius=12); pygame.draw.rect(self.screen, border, rect, 2, border_radius=12)
            draw_text(self.screen, kind, f["h3"], border, rect.x + 18, rect.y + 11); draw_text(self.screen, msg, f["mono"] if kind == "AGIR" else f["normal"], TEXT, rect.x + 130, rect.y + 18)
            y += 74

    def draw_risk(self, area):
        f = self.fonts; draw_text(self.screen, "Analyse des prévisions horaires", f["h3"], ORANGE, area.x + 35, area.y + 32)
        table = pygame.Rect(area.x + 170, area.y + 85, area.w - 340, 350); pygame.draw.rect(self.screen, PANEL, table, border_radius=14); pygame.draw.rect(self.screen, BORDER, table, 2, border_radius=14)
        draw_text(self.screen, "Heure", f["h3"], BLUE, table.x + 40, table.y + 28); draw_text(self.screen, "Prévision", f["h3"], BLUE, table.x + 180, table.y + 28); draw_text(self.screen, "Niveau", f["h3"], BLUE, table.x + 470, table.y + 28)
        y = table.y + 80
        for hour, weather, level, color in WEATHER_ROWS:
            row = pygame.Rect(table.x + 24, y - 10, table.w - 48, 44); pygame.draw.rect(self.screen, LIGHT_PANEL, row, border_radius=8); pygame.draw.rect(self.screen, BORDER, row, 1, border_radius=8)
            draw_text(self.screen, hour, f["normal"], TEXT, row.x + 18, row.y + 12); draw_text(self.screen, weather, f["normal"], TEXT, row.x + 150, row.y + 12)
            badge = pygame.Rect(row.x + 450, row.y + 8, 130, 28); pygame.draw.rect(self.screen, color, badge, border_radius=999); draw_centered_text(self.screen, level, f["small"], (255,255,255), badge.center)
            y += 52
        decision = pygame.Rect(area.x + 230, area.bottom - 125, area.w - 460, 70); pygame.draw.rect(self.screen, RED_LIGHT, decision, border_radius=12); pygame.draw.rect(self.screen, RED, decision, 3, border_radius=12)
        draw_centered_text(self.screen, "Décision : seuil d'alerte atteint à 16h", f["big"], TEXT, decision.center)

    def draw_alert(self, area):
        f = self.fonts; draw_text(self.screen, "Envoi d'une alerte WhatsApp contrôlée", f["h3"], ORANGE, area.x + 35, area.y + 32)
        phone = pygame.Rect(area.x + 330, area.y + 80, 330, 470); pygame.draw.rect(self.screen, DARK, phone, border_radius=28)
        screen = pygame.Rect(phone.x + 20, phone.y + 45, phone.w - 40, phone.h - 80); pygame.draw.rect(self.screen, (245,255,245), screen, border_radius=18); pygame.draw.rect(self.screen, GREEN, screen, 2, border_radius=18)
        draw_centered_text(self.screen, "WhatsApp", f["h2"], GREEN, (screen.centerx, screen.y + 28))
        sent = self.alert_sent or self.cycle >= 7; msg = pygame.Rect(screen.x + 22, screen.y + 80, screen.w - 44, 150)
        pygame.draw.rect(self.screen, GREEN_LIGHT if sent else PANEL, msg, border_radius=14); pygame.draw.rect(self.screen, GREEN if sent else BORDER, msg, 2, border_radius=14)
        draw_multiline(self.screen, WHATSAPP_ALERT if sent else "Alerte prête, en attente de condition météo ou autorisation.", f["normal"], TEXT, msg.x + 16, msg.y + 18, msg.w - 32)
        status = pygame.Rect(area.x + 230, area.bottom - 100, area.w - 460, 55); pygame.draw.rect(self.screen, GREEN_LIGHT if sent else ORANGE_LIGHT, status, border_radius=12); pygame.draw.rect(self.screen, GREEN if sent else ORANGE, status, 2, border_radius=12)
        draw_centered_text(self.screen, "Alerte envoyée + log enregistré" if sent else "Alerte non envoyée pour l'instant", f["h2"], TEXT, status.center)

    def draw_safety(self, area):
        f = self.fonts; draw_text(self.screen, "Garde-fous nécessaires", f["h3"], ORANGE, area.x + 35, area.y + 32)
        for i, (title, desc, color) in enumerate(GUARDRAILS):
            rect = pygame.Rect(area.x + 95 + (i % 2) * 410, area.y + 90 + (i // 2) * 120, 355, 78)
            pygame.draw.rect(self.screen, GREEN_LIGHT if color == GREEN else ORANGE_LIGHT, rect, border_radius=12); pygame.draw.rect(self.screen, color, rect, 2, border_radius=12)
            draw_text(self.screen, title, f["h3"], TEXT, rect.x + 16, rect.y + 12); draw_text(self.screen, desc, f["normal"], MUTED, rect.x + 16, rect.y + 40)
        note = pygame.Rect(area.x + 195, area.bottom - 105, area.w - 390, 56); pygame.draw.rect(self.screen, RED_LIGHT, note, border_radius=12); pygame.draw.rect(self.screen, RED, note, 2, border_radius=12)
        draw_centered_text(self.screen, "Un agent autonome doit toujours pouvoir être arrêté et audité.", f["h2"], TEXT, note.center)

    def draw_summary(self, area):
        f = self.fonts; draw_text(self.screen, "Synthèse : agent météo minimal", f["h3"], ORANGE, area.x + 35, area.y + 32)
        center = pygame.Rect(area.x + 355, area.y + 230, 230, 110); self.draw_component(center, "Agent météo", "autonome contrôlé", YELLOW_LIGHT, YELLOW)
        nodes = [("LLM", "raisonne", area.x+90, area.y+100, PURPLE_LIGHT, PURPLE), ("Runtime", "contrôle", area.x+680, area.y+100, BLUE_LIGHT, BLUE), ("GPS", "localise", area.x+90, area.y+410, GREEN_LIGHT, GREEN), ("Météo", "prévoit", area.x+380, area.y+410, ORANGE_LIGHT, ORANGE), ("WhatsApp", "alerte", area.x+680, area.y+410, RED_LIGHT, RED)]
        for title, sub, x, y, bg, border in nodes:
            rect = pygame.Rect(x, y, 190, 86); self.draw_component(rect, title, sub, bg, border); draw_arrow(self.screen, rect.center, center.center, border, 3)
        final = pygame.Rect(area.x + 180, area.bottom - 110, area.w - 360, 60); pygame.draw.rect(self.screen, GREEN_LIGHT, final, border_radius=12); pygame.draw.rect(self.screen, GREEN, final, 2, border_radius=12)
        draw_centered_text(self.screen, "LLM + outils + permissions + workflow horaire = agent utile", f["h2"], TEXT, final.center)

    def draw_bottom_pipeline(self):
        f = self.fonts; labels = [s["short"] for s in STAGES]
        x, y, gap = 34, 880, 8; w = int((WIDTH - 68 - gap * (len(labels) - 1)) / len(labels))
        for i, label in enumerate(labels):
            rect = pygame.Rect(x, y, w, 32); bg = ORANGE_LIGHT if i == self.stage else PANEL; border = ORANGE if i == self.stage else BORDER
            pygame.draw.rect(self.screen, bg, rect, border_radius=999); pygame.draw.rect(self.screen, border, rect, 2, border_radius=999)
            draw_centered_text(self.screen, label, f["small"], TEXT, rect.center); x += w + gap

    def draw(self):
        self.buttons = {}; self.screen.fill(BG)
        self.draw_header(); self.draw_buttons(); self.draw_left_panel(); self.draw_canvas(); self.draw_bottom_pipeline()
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
    WeatherAgentWorkshop().run()
