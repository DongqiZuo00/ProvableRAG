from copy import deepcopy


def proposal_view(public):
    view = deepcopy(public)
    originals = {}
    for index, passage in enumerate(view["passages"], 1):
        pid = "p" + str(index)
        originals[pid] = passage["id"]
        passage["id"] = pid
        for sentence_index, sentence in enumerate(passage["sentences"], 1):
            sid = pid + "s" + str(sentence_index)
            originals[sid] = sentence["sentence_id"]
            sentence["sentence_id"] = sid
    view["allowed_premise_ids"] = list(public["required_facts"])
    return view, originals
