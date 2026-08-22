import re


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def harness_query(activity_label, ctx_a, ctx_b):
    ctx = ctx_a + " " + ctx_b.capitalize()
    return preprocess(activity_label + ": " + ctx)


def queries_from_dataset(ds):
    return [harness_query(a, ca, cb)
            for a, ca, cb in zip(ds["activity_label"], ds["ctx_a"],
                                 ds["ctx_b"])]
