import os

# Unit tests start from an empty answer cache: the deployed api/starter_answers.json is loaded only when asked.
os.environ.setdefault("STARTER_ANSWERS", os.path.join(os.path.dirname(__file__), "no-starter-answers.json"))
