# SPDX-FileCopyrightText: 2026 Pedro Gomes
# SPDX-License-Identifier: LicenseRef-DLV-Personal-Use

"""Multiple-choice quizzes and exercise feedback for the course notebooks.

Questions follow the style of professional certification exams: one
correct answer out of four, often with a short scenario.
"""

import html
import random

import ipywidgets as widgets
from IPython.display import HTML, display

GREEN = "#0B5F61"
RED = "#D64545"
GREY = "#595959"
LETTERS = "ABCD"


def _feedback(correct, text):
    color = GREEN if correct else RED
    verdict = "Correct." if correct else "Not quite."
    return f"<div style='color:{color}; margin: 4px 0 12px 0'><b>{verdict}</b> {text}</div>"


def _shuffled(questions, seed):
    """Shuffle each question's options in a fixed order, so every student sees the same quiz."""
    rng = random.Random(seed)
    for item in questions:
        order = list(range(len(item["options"])))
        rng.shuffle(order)
        yield dict(item, options=[item["options"][k] for k in order], answer=order.index(item["answer"]))


def run_quiz(questions, shuffle=True, seed=0):
    """Show the questions with a Check button each and a running score.

    Each question is a dict with the keys ``question``, ``options`` (four
    strings), ``answer`` (index of the correct option), ``explanation`` and,
    optionally, ``topic``. With ``shuffle`` the options are reordered with a
    fixed ``seed``, so the correct answers are spread over A-D.
    """
    if shuffle:
        questions = list(_shuffled(questions, seed))
    results = {}
    score = widgets.HTML()

    def update_score():
        right = sum(results.values())
        score.value = (f"<h4 style='margin: 8px 0'>Score: {right} / {len(questions)} correct"
                       f" ({len(results)} answered)</h4>")

    rows = [score]
    for number, item in enumerate(questions, 1):
        topic = f" <span style='color:{GREY}'>· {html.escape(item['topic'])}</span>" if item.get("topic") else ""
        header = widgets.HTML(f"<b>Question {number}</b>{topic}<br>{html.escape(item['question'])}")
        choices = widgets.RadioButtons(
            options=[(f"{LETTERS[k]}. {text}", k) for k, text in enumerate(item["options"])],
            value=None, layout=widgets.Layout(width="auto"),
        )
        check = widgets.Button(description="Check", button_style="success",
                               layout=widgets.Layout(width="90px"))
        feedback = widgets.HTML()

        def on_check(_, item=item, number=number, choices=choices, feedback=feedback):
            if choices.value is None:
                feedback.value = f"<div style='color:{GREY}'>Choose an answer first.</div>"
                return
            correct = choices.value == item["answer"]
            results[number] = correct
            answer = f"The answer is {LETTERS[item['answer']]}. " if not correct else ""
            feedback.value = _feedback(correct, answer + html.escape(item["explanation"]))
            update_score()

        check.on_click(on_check)
        rows += [header, choices, check, feedback]
    rows.append(widgets.HTML(f"<i style='color:{GREY}'>Scroll back up to see your score.</i>"))
    update_score()
    display(widgets.VBox(rows))


def passed(message):
    display(HTML(_feedback(True, html.escape(message))))


def failed(message):
    display(HTML(_feedback(False, html.escape(message))))
