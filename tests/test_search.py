import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import search  # noqa: E402
from scraper import Page  # noqa: E402
from search import Index, Passage, SearchResult, detect_lang, normalize_text, trim_passage  # noqa: E402

SHOP = [
    Page("https://shop.example/shipping", "Shipping and delivery",
         "We deliver everywhere in Morocco. Orders to Casablanca and Rabat arrive in 1 to 2 working days.\n"
         "Shipping costs 35 MAD per order and is free for orders over 500 MAD."),
    Page("https://shop.example/returns", "Returns and refunds",
         "You can return any item within 30 days for a full refund. Items must be unused and in their box.\n"
         "Contact support to start a return. We pay the return shipping for damaged items."),
    Page("https://shop.example/payment", "Payment",
         "We accept Visa and Mastercard bank cards, PayPal, and cash on delivery in Morocco."),
    Page("https://shop.example/about", "About us",
         "We sell handmade rugs, ceramics and brass lamps made by artisans in Morocco."),
]
SHOP_AR = [
    Page("https://souk.example/ar/shipping", "الشحن والتوصيل",
         "نوصل الطلبات إلى جميع مدن المغرب خلال 2 إلى 4 أيام عمل. التوصيل مجاني للطلبات فوق 400 درهم."),
    Page("https://souk.example/ar/returns", "الإرجاع والاسترداد",
         "يمكنك إرجاع المنتج خلال 14 يوماً واسترداد أموالك كاملة. تواصل معنا عبر واتساب لبدء الإرجاع."),
]


def urls(found: SearchResult) -> list[str]:
    return [p.url for p in found.passages]


class StubTranslate:
    """Fake translator: no network; records every call."""

    def __init__(self, table: dict[str, str]):
        self.table, self.calls = table, []

    def __call__(self, question: str, target_lang: str) -> str:
        self.calls.append((question, target_lang))
        return self.table.get(question, question)


# ---------- normalization and language ----------

def test_normalize_text_arabic_forms():
    assert normalize_text("مُدَّة") == normalize_text("مدة") == "مده"            # tashkeel, ة -> ه
    assert normalize_text("إرجاع أموال آخر") == "ارجاع اموال اخر"                # أإآ -> ا
    assert normalize_text("إلى مستشفى") == "الي مستشفي"                          # ى -> ي
    assert normalize_text("التـــوصيل") == "التوصيل"                              # tatweel
    assert normalize_text("ＳＨＩＰＰＩＮＧ ٣٠") == "shipping 30"                    # NFKC + lower + Arabic-Indic digits


def test_detect_lang():
    assert detect_lang("كيف أرجع المنتج؟") == "ar"
    assert detect_lang("How do I return an item?") == "latin"
    assert detect_lang("Quels moyens de paiement acceptez-vous ?") == "latin"
    assert detect_lang("") == "latin"
    assert detect_lang("سعر iPhone") == "latin"  # Latin letters dominate


def test_index_language_majority_and_translation_target():
    assert (Index(SHOP).lang, Index(SHOP).site_lang) == ("latin", "en")
    assert (Index(SHOP_AR).lang, Index(SHOP_AR).site_lang) == ("ar", "ar")
    fr = Index([Page("https://fr.example/", "Livraison", "Nous livrons dans toute la France. Les frais de port sont offerts dès 50 euros pour vous.")])
    assert (fr.lang, fr.site_lang) == ("latin", "fr")
    assert Index([]).lang == "latin"


def test_arabic_prefixes_are_stripped_for_matching():
    toks = search._tokens("الشحن بالشحن والشحن للشحن فالشحن شحن")
    assert set(toks) == {"شحن"}
    assert search._tokens("من") == ["من"]  # short words stay as they are


# ---------- search ----------

def test_search_finds_the_right_page_and_reports_scores():
    found = Index(SHOP).search("How many days do I have to return an item?")
    assert found.matched and found.passages[0].url.endswith("/returns")
    assert found.lang == "latin" and not found.translated
    assert found.search_query == "How many days do I have to return an item?"
    hit = found.hits[0]
    assert hit.score > 0 and 0 < hit.coverage <= 1


@pytest.mark.parametrize("q", ["zzzz unrelated", "Who won the 2022 World Cup?", "what is it?", ""])
def test_off_topic_or_empty_question_finds_nothing(q):
    found = Index(SHOP).search(q)
    assert found.matched is False and found.passages == []


def test_arabic_question_on_arabic_site_without_translation():
    tr = StubTranslate({})
    found = Index(SHOP_AR).search("هل يمكنني إرْجاع المُنتج واسترداد أموالي؟", translate=tr)
    assert found.matched and found.passages[0].url.endswith("/returns")
    assert found.lang == "ar" and tr.calls == []  # same script as the site: no model call


def test_arabic_question_on_english_site_uses_translation():
    q = "كم يوما لدي لإرجاع المنتج واسترداد المال؟"
    assert Index(SHOP).search(q).matched is False  # no translator: nothing shares a word
    tr = StubTranslate({q: "How many days do I have to return the product for a refund?"})
    found = Index(SHOP).search(q, translate=tr)
    assert tr.calls == [(q, "en")]  # exactly one call, into the site's language
    assert found.translated and found.lang == "ar"
    assert found.passages[0].url.endswith("/returns")
    assert q in found.search_query and "refund" in found.search_query


def test_translation_is_skipped_when_the_question_already_matches():
    bilingual = SHOP + [Page("https://shop.example/ar/returns", "الإرجاع", "يمكنك إرجاع المنتج خلال 30 يوما واسترداد أموالك.")]
    tr = StubTranslate({})
    found = Index(bilingual).search("هل يمكنني إرجاع المنتج؟", translate=tr)
    assert found.passages[0].url.endswith("/ar/returns") and tr.calls == [] and not found.translated


def test_failing_or_useless_translator_falls_back_quietly():
    q = "كم يوما لدي لإرجاع المنتج؟"

    def boom(question, lang):
        raise RuntimeError("network down")

    for tr in (boom, StubTranslate({}), lambda question, lang: ""):
        found = Index(SHOP).search(q, translate=tr)
        assert found.translated is False and found.matched is False


def test_follow_up_uses_previous_question():
    idx = Index(SHOP)
    assert idx.search("How much is it?").matched is False  # only stopwords: nothing to search for
    found = idx.search("How much is it?", prev_question="How long does delivery to Casablanca take?")
    assert found.passages[0].url.endswith("/shipping")
    assert "Casablanca" in found.search_query


def test_follow_up_coverage_uses_the_current_question_when_it_has_words():
    found = Index(SHOP).search("And can I pay with PayPal?", prev_question="How long does delivery to Casablanca take?")
    assert urls(found)[0].endswith("/payment")
    assert "https://shop.example/shipping" not in urls(found)  # matches only the previous question's words


def test_relevance_cutoff_and_top_k():
    pages = [Page(f"https://docs.example/p{i}", f"Page {i}", f"The parser reads config files. Section {i} covers parser options.") for i in range(8)]
    pages.append(Page("https://docs.example/weak", "Misc", "A long unrelated note about the weather, cooking, travel and music, with one config word."))
    found = Index(pages).search("parser config options")
    assert len(found.hits) == search.TOP_K == 4
    assert "https://docs.example/weak" not in urls(found)
    assert len(Index(pages).search("parser config options", k=2).hits) == 2
    best = found.hits[0].score
    assert all(h.score >= search.REL_CUTOFF * best for h in found.hits)
    assert [h.score for h in found.hits] == sorted((h.score for h in found.hits), reverse=True)


def test_identical_text_on_two_urls_is_returned_once():
    text = "Delivery takes 3 to 5 working days to Morocco and Spain."
    found = Index([Page("https://a.example/ship", "Ship", text), Page("https://a.example/ship?ref=x", "Ship", text)]).search("delivery days")
    assert len(found.hits) == 1


def test_top_keeps_old_behaviour_with_default_k_4():
    idx = Index(SHOP * 3)
    assert len(idx.top("Morocco")) == 4
    assert idx.top("zzzz unrelated") == []
    assert Index([]).top("anything") == [] and Index([]).search("anything").matched is False


# ---------- trimming ----------

def test_trim_passage_keeps_matching_sentences_with_context():
    text = ("Intro one. Intro two. Filler three. Filler four. Our returns policy gives you 30 days. "
            "Refunds go back to your card. Filler seven. Filler eight. Gift cards never expire. Filler ten.")
    p = Passage("https://shop.example/help", "Help", text)
    out = trim_passage(p, "how do refunds and returns work")
    assert (out.url, out.title) == (p.url, p.title)
    assert out.text == "Filler four. Our returns policy gives you 30 days. Refunds go back to your card. Filler seven."
    out = trim_passage(p, "return policy and gift", context=0)
    assert out.text == "Our returns policy gives you 30 days. … Gift cards never expire."


def test_trim_passage_cuts_long_text_on_a_word_boundary():
    p = Passage("u", "t", "We ship parcels with a tracked carrier every weekday and every weekend too. " * 20)
    out = trim_passage(p, "tracked parcels", max_chars=200)
    assert len(out.text) <= 200 and out.text.endswith("…")
    assert out.text[:-1].rstrip().split(" ")[-1] in p.text.split(" ")  # no half word before the ellipsis


def test_trim_passage_unchanged_when_nothing_matches_and_works_in_arabic():
    p = Passage("u", "t", "Gift cards never expire. They can be used online.")
    assert trim_passage(p, "zzzz") is p
    ar = Passage("u", "t", "نوصل إلى جميع المدن. يمكنك إرجاع المنتج خلال 14 يوما. نحن متجر مغربي.")
    assert trim_passage(ar, "الإرجاع", context=0).text == "يمكنك إرجاع المنتج خلال 14 يوما."


# ---------- cross-language within the Latin script (wave 2) ----------

def test_question_lang_marks_latin_languages():
    assert search.question_lang("Quels moyens de paiement acceptez-vous ?") == "fr"
    assert search.question_lang("What payment methods do you accept?") == "en"
    assert search.question_lang("¿Cuánto cuesta el envío a Casablanca?") == "es"
    assert search.question_lang("كيف أرجع المنتج؟") == "ar"
    assert search.question_lang("PayPal?") == ""  # nothing marks it: treated as the site's language


def test_french_question_on_english_site_uses_translation():
    q = "Quels moyens de paiement acceptez-vous ?"
    assert Index(SHOP).search(q).matched is False  # no translator: no shared content word
    tr = StubTranslate({q: "Which payment methods do you accept?"})
    found = Index(SHOP).search(q, translate=tr)
    assert tr.calls == [(q, "en")] and found.translated and found.lang == "latin"
    assert found.passages[0].url.endswith("/payment")


def test_english_question_on_french_site_translates_into_french():
    fr = Index([Page("https://fr.example/livraison", "Livraison", "Nous livrons dans toute la France en 48 heures. Les frais de port sont offerts dès 50 euros."),
                Page("https://fr.example/paiement", "Paiement", "Nous acceptons la carte bancaire et PayPal pour vous simplifier la vie.")])
    q = "How long does my parcel take to arrive?"
    tr = StubTranslate({q: "Livraison en France : combien d'heures ?"})
    found = fr.search(q, translate=tr)
    assert tr.calls == [(q, "fr")] and found.translated
    assert found.passages[0].url.endswith("/livraison")
    # "shipping" has the synonym "livraison": found without a translation call
    tr2 = StubTranslate({})
    assert fr.search("How long does shipping take?", translate=tr2).passages[0].url.endswith("/livraison") and tr2.calls == []


def test_synonyms_title_and_address_words_find_the_pricing_page():
    pages = [Page("https://l.example/now", "Now", "We shipped a new inbox. Teams now cost less time in triage and review. " * 3),
             Page("https://l.example/pricing", "Pricing", "Basic plan $10 per user/month. Business plan $16 per user/month. Free for small teams."),
             Page("https://l.example/b", "Bakery", "Opening times: 8am to 5pm every day at Valencia Street.")]
    idx = Index(pages)
    assert idx.search("How much does Linear cost?").passages[0].url.endswith("/pricing")
    assert idx.search("What are your opening hours?").passages[0].url.endswith("/b")


def test_at_most_two_passages_per_page():
    long = Page("https://s.example/korea", "Korea", "\n".join(f"Our Korea bakery {i} opens at 9am in Seoul district {i}. " * 12 for i in range(6)))
    other = Page("https://s.example/cities", "Cities", "Bakery hours: San Francisco opens at 8am, closes 5pm.")
    found = Index([long, other]).search("bakery opens am")
    urls = [p.url for p in found.passages]
    assert urls.count("https://s.example/korea") <= 2 and "https://s.example/cities" in urls


def test_closest_page_for_an_unanswered_question():
    idx = Index([Page("https://b.example/etre-contacte", "Être contacté", "Laissez vos coordonnées et un conseiller vous rappelle."),
                 Page("https://b.example/cartes", "Cartes", "Nos cartes bancaires pour tous les besoins du quotidien.")])
    assert idx.closest("Comment contacter le service client ?").url.endswith("/etre-contacte")


def test_latin_translation_skipped_when_covered_or_same_language_or_unmarked():
    tr = StubTranslate({})
    assert Index(SHOP).search("PayPal et Visa pour vous ?", translate=tr).matched  # French marks, but 100% covered
    Index(SHOP).search("Can I pay with PayPal?", translate=tr)                    # same language
    Index(SHOP).search("Who won the 2022 World Cup?", translate=tr)               # same language, off-topic: still 0 calls
    Index(SHOP).search("Mastercard?", translate=tr)                               # unmarked
    assert tr.calls == []


def test_trim_joins_hard_wrapped_lines():
    p = Passage("https://docs.example/interp", "Interpreter",
                "See the install manager for other ways to launch it.\nTyping an end-of-file character (Control-D on Unix, Control-Z on\n"
                "Windows) at the primary prompt causes the interpreter to exit with a zero exit\nstatus.  If that doesn't work, you can exit "
                "the interpreter by typing the\nfollowing command: quit().\nThe interpreter's line-editing features include history.\n"
                + "Unrelated filler sentence about colours and shapes. " * 12)
    t = trim_passage(p, "How do I exit the interpreter?").text
    assert "Control-D" in t and "quit()" in t and t.count("filler") <= 1  # one context sentence at most


# ---------------------------------------------------------------- wave 3: false "not on this site"

def _fixture(name):
    import json
    path = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / f"{name}.json"
    return Index([Page(p["url"], p["title"], p["text"]) for p in json.loads(path.read_text(encoding="utf-8"))])


def test_how_long_question_finds_the_refund_sentence():
    r = _fixture("shop").search("How long do refunds take?")
    assert r.matched and r.hits[0].passage.url.endswith("/returns")


def test_long_multi_part_question_finds_every_part():
    r = _fixture("shop").search("Explain every payment method, delivery rule and return condition in detail, as a long list.")
    urls = {h.passage.url.rsplit("/", 1)[-1] for h in r.hits}
    assert {"payment", "returns", "shipping"} <= urls


@pytest.mark.parametrize("q", ["كم سعر التوصيل وكم يوما يستغرق؟", "كم تكلفة التوصيل وكم يوما يستغرق؟"])
def test_arabic_two_part_question_finds_the_shipping_page(q):
    r = _fixture("shop_ar").search(q)
    assert r.matched and r.hits[0].passage.url.endswith("/shipping")


def test_waw_plus_stopword_is_a_stopword_and_unit_plurals_meet():
    assert search._content(search._tokens("وكم وهل")) == []
    assert search._tokens("يوما") == search._tokens("أيام") == search._tokens("الأيام")
    assert search._tokens("الواتساب") == search._tokens("واتساب")
    assert search._tokens("tracking") == search._tokens("track") and search._tokens("shipping") == ["ship"]
    assert search._tokens("string thing calling missing") == ["string", "thing", "call", "miss"]


def test_off_site_multi_part_question_still_finds_nothing():
    r = _fixture("shop").search("Who won the 2022 World Cup and who scored the final goal?")
    assert not r.matched


def test_parts_only_add_after_the_whole_question_hits():
    r = _fixture("shop").search("How much is the return fee?")
    assert [h.passage.url.rsplit("/", 1)[-1] for h in r.hits] == ["returns"]
    assert search.split_parts("How much is the return fee?") == []
    assert len(search.split_parts("كم سعر التوصيل وكم يوما يستغرق؟")) == 2


def test_price_question_keeps_the_amount_lines_when_trimming():
    p = Passage("https://l.example/pricing", "Pricing", "Linear Agent included.\nBasic\n$10 per user/month\nUnlimited issues.\nLinear Insights for teams.")
    out = trim_passage(p, "How much does Linear cost?")
    assert "$10 per user/month" in out.text
