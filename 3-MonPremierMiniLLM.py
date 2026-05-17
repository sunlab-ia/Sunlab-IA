import random
import re
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass

import pygame

# ============================================================
# Mini-LLM pédagogique probabiliste
# Trois modes progressifs :
# 1) Lettre précédente
# 2) T9 simple par préfixe de mot
# 3) T9 contextualisé avec réduction progressive du contexte
# ============================================================

WIDTH, HEIGHT = 1500, 930
FPS = 60

BG = (255, 250, 240)
PANEL = (255, 255, 255)
TEXT = (10, 10, 10)
MUTED = (82, 72, 58)
BORDER = (240, 217, 159)
YELLOW = (255, 191, 0)
ORANGE = (255, 136, 23)
BLUE = (30, 111, 159)
GREEN = (90, 157, 47)
RED = (216, 59, 45)
PURPLE = (113, 69, 168)
DARK = (35, 35, 35)

CORPUS_FILE = "corpus.txt"

DEFAULT_CORPUS = """
Il était une fois un petit robot qui voulait apprendre à écrire.
Au début, il ne connaissait rien. Il observait seulement les lettres,
les espaces, les mots, les phrases et les répétitions.

Quand il voyait souvent la lettre q, il remarquait que la lettre suivante
était presque toujours u. Quand il voyait la lettre l, il observait souvent
la lettre e ou la lettre a. Petit à petit, il construisait une carte des
probabilités.

Le petit robot aimait les histoires. Il était une fois un roi, une reine,
un cheval, un chemin, une maison et une forêt. Le cheval suivait le chemin.
Je cherche un chemin. Je cherche une idée. Je cherche un exemple. Le chien
cherche la balle. La maison est près du chemin.

Ce petit modèle ne comprenait pas vraiment le sens des phrases. Il ne savait
pas ce qu'était un chat, une maison ou une étoile. Mais il pouvait imiter
certaines habitudes du texte. Il pouvait deviner quelle lettre venait souvent
après une autre, ou quel mot apparaissait souvent après certains mots.

Plus le texte était long, plus ses statistiques devenaient riches.
Plus le contexte était grand, plus ses prédictions semblaient précises.
Mais si le contexte était inconnu, le modèle devait improviser.
"""


def normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text):
    return re.findall(r"[a-zàâäéèêëîïôöùûüçœæ'-]+", text.lower())


def visible_char(ch):
    return "␠ espace" if ch == " " else ch


def split_prompt(prompt):
    original = prompt
    prompt = prompt.lower().replace("’", "'")
    ends_with_space = bool(original) and original[-1].isspace()
    words = tokenize_words(prompt)
    if ends_with_space:
        return words, ""
    if words:
        return words[:-1], words[-1]
    return [], ""


class ProbabilisticLanguageModel:
    def __init__(self, max_word_context=4):
        self.max_word_context = max_word_context
        self.char_after_char = defaultdict(Counter)
        self.word_counts = Counter()
        self.words_by_prefix = defaultdict(Counter)
        self.next_word_tables = {n: defaultdict(Counter) for n in range(1, max_word_context + 1)}
        self.corpus_text = ""
        self.words = []
        self.trained = False

    def train(self, text):
        self.corpus_text = normalize_text(text)
        self.words = tokenize_words(self.corpus_text)
        self.char_after_char.clear()
        self.word_counts.clear()
        self.words_by_prefix.clear()
        self.next_word_tables = {n: defaultdict(Counter) for n in range(1, self.max_word_context + 1)}

        for i in range(len(self.corpus_text) - 1):
            self.char_after_char[self.corpus_text[i]][self.corpus_text[i + 1]] += 1

        self.word_counts.update(self.words)
        for word, count in self.word_counts.items():
            for k in range(0, len(word) + 1):
                self.words_by_prefix[word[:k]][word] += count

        for i in range(len(self.words)):
            next_word = self.words[i]
            for n in range(1, self.max_word_context + 1):
                if i - n < 0:
                    continue
                ctx = tuple(self.words[i - n:i])
                self.next_word_tables[n][ctx][next_word] += 1
        self.trained = True

    def next_letters(self, prompt, limit=10, normalize_space=True):
        if not self.trained:
            return [], ""
        text = normalize_text(prompt)
        if not text:
            counts = Counter()
            for c in self.char_after_char.values():
                counts.update(c)
            context = "(global)"
        else:
            context = text[-1]
            counts = Counter(self.char_after_char.get(context, Counter()))
        if normalize_space and " " in counts:
            counts[" "] = max(1, int(counts[" "] * 0.25))
        total = sum(counts.values())
        if total == 0:
            return [], context
        return [(ch, count, count / total) for ch, count in counts.most_common(limit)], context

    def t9_simple(self, prefix, limit=10):
        counts = self.words_by_prefix.get(prefix.lower(), Counter())
        total = sum(counts.values())
        if total == 0:
            return []
        return [(word, count, count / total) for word, count in counts.most_common(limit)]

    def t9_contextual(self, previous_words, prefix, limit=10):
        prefix = prefix.lower()
        previous_words = [w.lower() for w in previous_words]
        max_n = min(self.max_word_context, len(previous_words))
        attempted = []
        for n in range(max_n, 0, -1):
            ctx = tuple(previous_words[-n:])
            attempted.append(" ".join(ctx))
            next_counts = self.next_word_tables[n].get(ctx, Counter())
            if not next_counts:
                continue
            filtered = Counter({w: c for w, c in next_counts.items() if w.startswith(prefix)})
            total = sum(filtered.values())
            if total > 0:
                return ([(w, c, c / total) for w, c in filtered.most_common(limit)],
                        f"contexte {n} mot(s) : « {' '.join(ctx)} »", attempted, False)
        return self.t9_simple(prefix, limit), "retour au T9 simple", attempted, True

    def weighted_choice(self, suggestions):
        if not suggestions:
            return ""
        total = sum(p for _, _, p in suggestions)
        r = random.random() * total
        acc = 0.0
        for word, _, p in suggestions:
            acc += p
            if acc >= r:
                return word
        return suggestions[0][0]


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



def wrap_text_lines(text, font, max_width):
    words = text.split(" ")
    lines = []
    line = ""

    for word in words:
        test = line + word + " "
        if font.size(test)[0] > max_width and line:
            lines.append(line.rstrip())
            line = word + " "
        else:
            line = test

    if line:
        lines.append(line.rstrip())

    return lines


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


class MiniLLMT9Demo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Mini-LLM probabiliste — lettres, T9 et contexte")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": pygame.font.SysFont("arial", 27, bold=True),
            "h2": pygame.font.SysFont("arial", 18, bold=True),
            "h3": pygame.font.SysFont("arial", 15, bold=True),
            "normal": pygame.font.SysFont("arial", 14),
            "small": pygame.font.SysFont("arial", 12),
            "mono": pygame.font.SysFont("consolas", 15),
            "button": pygame.font.SysFont("arial", 13, bold=True),
        }
        self.mode = 1
        self.max_word_context = 4
        self.normalize_space = True
        self.available_corpora = self.find_corpora()
        self.selected_corpus_index = 0
        self.corpus_dropdown_open = False

        # corpus.txt chargé par défaut si présent.
        lower_names = [name.lower() for name in self.available_corpora]
        if "corpus.txt" in lower_names:
            self.selected_corpus_index = lower_names.index("corpus.txt")

        self.current_corpus_name = (
            self.available_corpora[self.selected_corpus_index]
            if self.available_corpora
            else CORPUS_FILE
        )

        self.corpus = self.load_corpus()
        self.model = ProbabilisticLanguageModel(max_word_context=self.max_word_context)
        self.model.train(self.corpus)
        self.prompt = "q"
        self.message = "Mode 1 : tape une lettre. Exemple idéal : q, l, e."
        self.buttons = {}


    def find_corpora(self):
        """
        Recherche automatiquement tous les fichiers .txt présents
        dans le dossier où se trouve ce fichier Python.
        """
        self.program_dir = Path(__file__).resolve().parent
        txt_files = []

        for path in self.program_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".txt":
                txt_files.append(path.name)

        txt_files.sort(key=lambda name: name.lower())

        if not txt_files:
            txt_files.append("corpus.txt")

        return txt_files


    def load_corpus(self):
        try:
            with open(self.program_dir / self.current_corpus_name, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    return normalize_text(text)
        except FileNotFoundError:
            pass
        except UnicodeDecodeError:
            try:
                with open(self.program_dir / self.current_corpus_name, "r", encoding="latin-1") as f:
                    return normalize_text(f.read())
            except Exception:
                pass
        return normalize_text(DEFAULT_CORPUS)

    def reload_corpus(self):
        """
        Recharge simplement le corpus actuellement sélectionné
        dans la liste déroulante.
        """
        self.corpus = self.load_corpus()

        self.model = ProbabilisticLanguageModel(
            max_word_context=self.max_word_context
        )

        self.model.train(self.corpus)

        self.message = (
            f"Corpus chargé : {self.current_corpus_name} — "
            f"{len(self.corpus)} caractères, "
            f"{len(self.model.words)} mots."
        )


    def next_corpus(self):
        """
        Passe au corpus suivant dans la liste.
        """
        if not self.available_corpora:
            return

        self.selected_corpus_index += 1

        if self.selected_corpus_index >= len(self.available_corpora):
            self.selected_corpus_index = 0

        self.current_corpus_name = self.available_corpora[self.selected_corpus_index]
        self.reload_corpus()


    def previous_corpus(self):
        """
        Passe au corpus précédent dans la liste.
        """
        if not self.available_corpora:
            return

        self.selected_corpus_index -= 1

        if self.selected_corpus_index < 0:
            self.selected_corpus_index = len(self.available_corpora) - 1

        self.current_corpus_name = self.available_corpora[self.selected_corpus_index]
        self.reload_corpus()


    def toggle_corpus_dropdown(self):
        """
        Ouvre ou ferme le menu déroulant des corpus.
        La liste est recalculée à chaque ouverture pour tenir compte
        des fichiers .txt ajoutés ou supprimés dans le dossier.
        """
        if not self.corpus_dropdown_open:
            self.available_corpora = self.find_corpora()
            if self.current_corpus_name in self.available_corpora:
                self.selected_corpus_index = self.available_corpora.index(self.current_corpus_name)
            else:
                self.selected_corpus_index = 0
                self.current_corpus_name = self.available_corpora[0]

        self.corpus_dropdown_open = not self.corpus_dropdown_open


    def select_corpus(self, index):
        """
        Sélectionne un corpus depuis le menu déroulant.
        """
        if index < 0 or index >= len(self.available_corpora):
            return

        self.selected_corpus_index = index
        self.current_corpus_name = self.available_corpora[index]
        self.corpus_dropdown_open = False
        self.reload_corpus()


    def set_mode(self, mode):
        self.mode = mode
        if mode == 1:
            self.prompt = "q"
            self.message = "Mode 1 : on regarde seulement la dernière lettre."
        elif mode == 2:
            self.prompt = "che"
            self.message = "Mode 2 : T9 simple, les mots sont classés par fréquence."
        else:
            self.prompt = "je che"
            self.message = "Mode 3 : T9 contextualisé avec réduction progressive du contexte."

    def current_word_suggestions(self):
        previous_words, prefix = split_prompt(self.prompt)
        if self.mode == 2:
            return self.model.t9_simple(prefix, limit=10)
        if self.mode == 3:
            suggestions, _, _, _ = self.model.t9_contextual(previous_words, prefix, limit=10)
            return suggestions
        return []

    def accept_suggestion(self, index):
        suggestions = self.current_word_suggestions()
        if index >= len(suggestions):
            return
        self.replace_prefix_with_word(suggestions[index][0])
        self.message = f"Suggestion acceptée : « {suggestions[index][0]} »."

    def replace_prefix_with_word(self, word):
        _, prefix = split_prompt(self.prompt)
        if prefix:
            pos = self.prompt.lower().rfind(prefix)
            self.prompt = self.prompt[:pos] + word + " "
        else:
            if self.prompt and not self.prompt.endswith(" "):
                self.prompt += " "
            self.prompt += word + " "

    def add_next_word(self):
        previous_words, prefix = split_prompt(self.prompt)
        if self.mode == 2:
            suggestions = self.model.t9_simple(prefix, limit=20)
        elif self.mode == 3:
            suggestions, _, _, _ = self.model.t9_contextual(previous_words, prefix, limit=20)
        else:
            return
        word = self.model.weighted_choice(suggestions)
        if not word:
            self.message = "Aucun mot proposé."
            return
        self.replace_prefix_with_word(word)
        self.message = f"Mot ajouté : « {word} »."

    def add_next_letter(self):
        suggestions, _ = self.model.next_letters(self.prompt, limit=10, normalize_space=self.normalize_space)
        if not suggestions:
            return
        ch = self.model.weighted_choice(suggestions)
        self.prompt += ch
        self.message = f"Lettre ajoutée : {visible_char(ch)}."

    def handle_click(self, pos):
        # Si le menu corpus est ouvert, on regarde d'abord
        # si l'utilisateur clique sur un élément du menu déroulant.
        if self.corpus_dropdown_open:
            for i in range(min(len(self.available_corpora), 8)):
                key = f"corpus_item_{i}"
                if key in self.buttons and self.buttons[key].rect.collidepoint(pos):
                    self.select_corpus(i)
                    return

        for key, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                if key == "mode1": self.set_mode(1)
                elif key == "mode2": self.set_mode(2)
                elif key == "mode3": self.set_mode(3)
                elif key == "reload":
                    self.toggle_corpus_dropdown()
                elif key == "next_corpus":
                    self.next_corpus()
                elif key == "prev_corpus":
                    self.previous_corpus()
                elif key == "reset":
                    self.prompt = ""
                    self.message = "Phrase effacée."
                elif key == "example":
                    if self.mode == 1:
                        self.prompt = random.choice(["q", "l", "e", "qu", "il "])
                    elif self.mode == 2:
                        self.prompt = random.choice(["che", "ma", "ro", "pe", "app"])
                    else:
                        self.prompt = random.choice(["je che", "le che", "il était une f", "la maison est", "le petit"])
                    self.message = "Exemple chargé."
                elif key == "add_letter": self.add_next_letter()
                elif key == "add_word": self.add_next_word()
                elif key == "space_norm":
                    self.normalize_space = not self.normalize_space
                    self.message = "Normalisation des espaces activée." if self.normalize_space else "Espaces naturels."
                elif key.startswith("suggest_"):
                    self.accept_suggestion(int(key.split("_")[1]))
                return

    def handle_key(self, event):
        if event.key == pygame.K_BACKSPACE:
            self.prompt = self.prompt[:-1]
        elif event.key == pygame.K_RETURN:
            self.add_next_letter() if self.mode == 1 else self.add_next_word()
        elif event.key == pygame.K_ESCAPE:
            pygame.quit(); sys.exit()
        elif event.key in [pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4]:
            self.accept_suggestion(event.key - pygame.K_F1)
        elif event.key == pygame.K_TAB:
            self.prompt += " "
        else:
            ch = event.unicode
            if ch:
                self.prompt += ch.lower()
                if len(self.prompt) > 180:
                    self.prompt = self.prompt[-180:]

    def draw_header(self):
        f = self.fonts
        draw_text(self.screen, "MINI-LLM PROBABILISTE : LETTRES, T9 ET CONTEXTE", f["title"], TEXT, 34, 22)
        draw_text(self.screen, "Une progression pédagogique vers l'idée centrale des modèles de langage : prédire la suite probable.", f["normal"], MUTED, 34, 56)
        stats = [
            ("Mode", str(self.mode)),
            ("Corpus", self.current_corpus_name[:18]),
            ("Car.", f"{len(self.corpus):,}".replace(",", " ")),
            ("Mots", f"{len(self.model.words):,}".replace(",", " ")),
            ("Contexte max", f"{self.max_word_context} mots")
        ]
        x = 675
        for label, value in stats:
            draw_centered_text(self.screen, label, f["small"], MUTED, (x + 70, 25))
            rect = pygame.Rect(x, 42, 140, 30)
            pygame.draw.rect(self.screen, (255, 241, 198), rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=8)
            draw_centered_text(self.screen, value, f["button"], TEXT, rect.center)
            x += 152

    def draw_mode_buttons(self):
        f = self.fonts
        y = 92

        self.buttons["mode1"] = Button(
            pygame.Rect(34, y, 280, 42),
            "Mode 1 — lettre précédente",
            "orange" if self.mode == 1 else "light"
        )

        self.buttons["mode2"] = Button(
            pygame.Rect(330, y, 280, 42),
            "Mode 2 — T9 simple",
            "orange" if self.mode == 2 else "light"
        )

        self.buttons["mode3"] = Button(
            pygame.Rect(626, y, 320, 42),
            "Mode 3 — T9 contextualisé",
            "orange" if self.mode == 3 else "light"
        )

        self.buttons["prev_corpus"] = Button(
            pygame.Rect(972, y, 42, 42),
            "◀",
            "light"
        )

        label = self.current_corpus_name
        if len(label) > 24:
            label = label[:21] + "..."

        self.buttons["reload"] = Button(
            pygame.Rect(1022, y, 280, 42),
            ("▼ " if self.corpus_dropdown_open else "▶ ") + label,
            "blue" if self.corpus_dropdown_open else "light"
        )

        self.buttons["next_corpus"] = Button(
            pygame.Rect(1310, y, 42, 42),
            "▶",
            "light"
        )

        self.buttons["reset"] = Button(
            pygame.Rect(1360, y, 105, 42),
            "Effacer",
            "dark"
        )

        for key in ["mode1", "mode2", "mode3", "prev_corpus", "reload", "next_corpus", "reset"]:
            self.buttons[key].draw(self.screen, f)

        if self.corpus_dropdown_open:
            self.draw_corpus_dropdown()


    def draw_corpus_dropdown(self):
        """
        Dessine la liste déroulante des fichiers .txt détectés.
        """
        f = self.fonts

        x = 1022
        y = 138
        w = 330
        item_h = 34
        max_items = min(len(self.available_corpora), 8)

        dropdown_rect = pygame.Rect(x, y, w, item_h * max_items + 10)
        pygame.draw.rect(self.screen, PANEL, dropdown_rect, border_radius=8)
        pygame.draw.rect(self.screen, BLUE, dropdown_rect, 2, border_radius=8)

        for i in range(max_items):
            item_y = y + 5 + i * item_h
            name = self.available_corpora[i]
            label = name if len(name) <= 36 else name[:33] + "..."

            kind = "blue" if i == self.selected_corpus_index else "light"
            self.buttons[f"corpus_item_{i}"] = Button(
                pygame.Rect(x + 5, item_y, w - 10, item_h - 4),
                label,
                kind
            )
            self.buttons[f"corpus_item_{i}"].draw(self.screen, f)

        if len(self.available_corpora) > max_items:
            draw_text(
                self.screen,
                f"+ {len(self.available_corpora) - max_items} autre(s) fichier(s)",
                f["small"],
                MUTED,
                x + 12,
                y + 5 + max_items * item_h
            )


    def draw_left_panel(self):
        f = self.fonts
        rect = pygame.Rect(34, 158, 420, 710)
        draw_panel(self.screen, rect, "Corpus et principe", f)
        if self.mode == 1:
            title = "Mode 1 : lettre précédente"; msg = "Le modèle regarde uniquement la dernière lettre. Il compte dans le corpus quelles lettres apparaissent juste après. C'est idéal pour montrer q → u."
        elif self.mode == 2:
            title = "Mode 2 : T9 simple"; msg = "Le modèle repère le début du mot en cours. Il propose les mots du corpus qui commencent pareil, classés par fréquence globale."
        else:
            title = "Mode 3 : T9 contextualisé"; msg = "Le modèle utilise les mots précédents pour classer les suggestions. Si le contexte est introuvable, il le réduit progressivement."
        draw_text(self.screen, title, f["h3"], BLUE, 54, 198)
        draw_multiline(self.screen, msg, f["normal"], MUTED, 54, 228, 360, 5, max_lines=7)
        box = pygame.Rect(54, 350, 360, 360)
        pygame.draw.rect(self.screen, (255, 253, 247), box, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, box, 1, border_radius=10)
        draw_text(self.screen, "Extrait du corpus", f["h3"], TEXT, 70, 366)
        draw_multiline(self.screen, self.corpus[:1200], f["small"], TEXT, 70, 398, 320, 4, max_lines=18)
        self.buttons["example"] = Button(pygame.Rect(54, 735, 170, 40), "Charger un exemple", "light")
        self.buttons["space_norm"] = Button(pygame.Rect(238, 735, 176, 40), "Espaces normalisés" if self.normalize_space else "Espaces naturels", "blue" if self.normalize_space else "light")
        self.buttons["example"].draw(self.screen, f); self.buttons["space_norm"].draw(self.screen, f)
        draw_multiline(self.screen, "Astuce : F1 à F4 acceptent directement les premières suggestions. Entrée ajoute la prochaine lettre ou le prochain mot.", f["small"], MUTED, 54, 795, 350, 4, max_lines=4)

    def draw_center_panel(self):
        f = self.fonts
        rect = pygame.Rect(478, 158, 520, 710)
        draw_panel(self.screen, rect, "Écriture interactive", f)
        draw_text(self.screen, "Phrase en cours :", f["h3"], TEXT, 500, 202)
        input_rect = pygame.Rect(500, 230, 476, 130)
        pygame.draw.rect(self.screen, (255, 253, 247), input_rect, border_radius=10)
        pygame.draw.rect(self.screen, ORANGE, input_rect, 2, border_radius=10)
        # Zone de texte avec défilement automatique vers le bas.
        # Le texte complet est conservé dans self.prompt, mais seules les
        # dernières lignes visibles sont affichées dans le cadre.
        display_text = self.prompt if self.prompt else "Tape ici..."
        display_color = TEXT if self.prompt else (150, 140, 120)

        lines = wrap_text_lines(display_text, f["mono"], 420)
        line_height = f["mono"].get_height() + 6
        max_visible_lines = 5

        # Scroll automatique : on affiche toujours les dernières lignes.
        start_line = max(0, len(lines) - max_visible_lines)
        visible_lines = lines[start_line:]

        yy = 250
        for line in visible_lines:
            draw_text(self.screen, line, f["mono"], display_color, 518, yy)
            yy += line_height

        # Ascenseur visuel : il indique que le texte est plus long que la zone.
        if len(lines) > max_visible_lines:
            bar_x = input_rect.right - 14
            bar_y = input_rect.y + 12
            bar_h = input_rect.height - 24

            pygame.draw.rect(
                self.screen,
                (230, 230, 230),
                (bar_x, bar_y, 7, bar_h),
                border_radius=999
            )

            ratio = max_visible_lines / len(lines)
            thumb_h = max(20, int(bar_h * ratio))
            scroll_ratio = start_line / max(1, len(lines) - max_visible_lines)
            thumb_y = bar_y + int((bar_h - thumb_h) * scroll_ratio)

            pygame.draw.rect(
                self.screen,
                ORANGE,
                (bar_x, thumb_y, 7, thumb_h),
                border_radius=999
            )
        if self.mode == 1:
            self.buttons["add_letter"] = Button(pygame.Rect(500, 382, 230, 42), "Générer le prochain caractère", "orange")
            self.buttons["add_letter"].draw(self.screen, f)
        else:
            self.buttons["add_word"] = Button(pygame.Rect(500, 382, 230, 42), "Compléter le mot", "orange")
            self.buttons["add_word"].draw(self.screen, f)
        previous_words, prefix = split_prompt(self.prompt)
        draw_text(self.screen, "Analyse de la phrase", f["h3"], BLUE, 500, 456)
        ana = pygame.Rect(500, 486, 476, 154)
        pygame.draw.rect(self.screen, (255, 253, 247), ana, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, ana, 1, border_radius=10)
        if self.mode == 1:
            last = normalize_text(self.prompt)[-1:] if normalize_text(self.prompt) else "(aucun)"
            lines = [f"Dernier caractère : {repr(last)}", "Le modèle consulte : lettre précédente → lettre suivante", "Exemple pédagogique : q donne souvent u."]
        else:
            lines = [f"Mots précédents : {' '.join(previous_words[-6:]) if previous_words else '(aucun)'}", f"Préfixe du mot en cours : {repr(prefix)}"]
            if self.mode == 3:
                _, used, attempted, _ = self.model.t9_contextual(previous_words, prefix, limit=10)
                lines.append(f"Contexte utilisé : {used}")
                if attempted:
                    lines.append("Contextes testés : " + " | ".join(attempted[-3:]))
        yy = 504
        for line in lines:
            draw_multiline(self.screen, line, f["small"], TEXT, 518, yy, 430, 3, max_lines=2)
            yy += 34
        draw_text(self.screen, "Message pédagogique", f["h3"], BLUE, 500, 680)
        draw_multiline(self.screen, self.message, f["normal"], MUTED, 500, 710, 450, 5, max_lines=5)

    def draw_right_panel(self):
        f = self.fonts
        rect = pygame.Rect(1022, 158, 444, 710)
        draw_panel(self.screen, rect, "Suggestions", f)
        if self.mode == 1:
            self.draw_letter_suggestions()
        else:
            self.draw_word_suggestions()

    def draw_letter_suggestions(self):
        f = self.fonts
        suggestions, context = self.model.next_letters(self.prompt, limit=12, normalize_space=self.normalize_space)
        draw_text(self.screen, f"Contexte : {repr(context)}", f["h3"], TEXT, 1042, 202)
        draw_text(self.screen, "Lettres suivantes probables", f["normal"], MUTED, 1042, 232)
        max_prob = max([p for _, _, p in suggestions], default=1)
        y = 274; colors = [ORANGE, BLUE, GREEN, PURPLE, RED]
        for idx, (ch, count, prob) in enumerate(suggestions):
            draw_text(self.screen, visible_char(ch), f["mono"], TEXT, 1042, y - 3)
            draw_progress(self.screen, pygame.Rect(1160, y, 180, 17), prob / max_prob, colors[idx % len(colors)])
            draw_right_text(self.screen, f"{prob * 100:5.1f}%", f["normal"], TEXT, 1410, y - 4)
            draw_text(self.screen, f"({count})", f["small"], MUTED, 1418, y - 2)
            y += 36
        draw_text(self.screen, "Ce mode montre la statistique la plus simple :", f["h3"], BLUE, 1042, 760)
        draw_text(self.screen, "une lettre conditionne la suivante.", f["normal"], MUTED, 1042, 790)

    def draw_word_suggestions(self):
        f = self.fonts
        previous_words, prefix = split_prompt(self.prompt)
        if self.mode == 2:
            suggestions = self.model.t9_simple(prefix, limit=10)
            used = "fréquence globale des mots"; fallback = False
        else:
            suggestions, used, attempted, fallback = self.model.t9_contextual(previous_words, prefix, limit=10)
        draw_text(self.screen, f"Préfixe : {repr(prefix)}", f["h3"], TEXT, 1042, 202)
        draw_multiline(self.screen, f"Classement : {used}", f["small"], MUTED, 1042, 232, 360, 3, max_lines=3)
        y = 326 if fallback and self.mode == 3 else 306
        if fallback and self.mode == 3:
            draw_text(self.screen, "Aucun contexte utile : retour au T9 simple.", f["small"], RED, 1042, 292)
        if not suggestions:
            draw_multiline(self.screen, "Aucun mot trouvé. Essaie un préfixe plus court ou recharge un corpus plus grand.", f["normal"], MUTED, 1042, y, 360, 5, max_lines=4)
            return
        max_prob = max([p for _, _, p in suggestions], default=1); colors = [ORANGE, BLUE, GREEN, PURPLE, RED]
        for idx, (word, count, prob) in enumerate(suggestions[:10]):
            key = f"suggest_{idx}"
            self.buttons[key] = Button(pygame.Rect(1042, y - 5, 42, 28), f"F{idx + 1}" if idx < 4 else str(idx + 1), "light")
            self.buttons[key].draw(self.screen, f)
            draw_text(self.screen, word, f["mono"], TEXT, 1094, y - 3)
            draw_progress(self.screen, pygame.Rect(1228, y, 120, 16), prob / max_prob, colors[idx % len(colors)])
            draw_right_text(self.screen, f"{prob * 100:4.1f}%", f["small"], TEXT, 1408, y - 2)
            draw_text(self.screen, f"({count})", f["small"], MUTED, 1418, y - 2)
            y += 42
        draw_text(self.screen, "Idée clé", f["h3"], BLUE, 1042, 760)
        msg = "Le modèle complète un mot grâce aux fréquences du corpus." if self.mode == 2 else "Le contexte change l'ordre des mots proposés. S'il manque, le modèle revient en arrière."
        draw_multiline(self.screen, msg, f["normal"], MUTED, 1042, 790, 360, 5, max_lines=3)

    def draw_footer(self):
        draw_text(self.screen, "Clavier : tape une phrase · Entrée : compléter · Backspace : effacer · F1-F4 : accepter suggestion · Échap : quitter", self.fonts["small"], MUTED, 34, 900)

    def draw(self):
        self.buttons = {}
        self.screen.fill(BG)
        self.draw_header(); self.draw_mode_buttons(); self.draw_left_panel(); self.draw_center_panel(); self.draw_right_panel(); self.draw_footer()
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
    MiniLLMT9Demo().run()
