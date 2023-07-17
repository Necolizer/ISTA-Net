import json
from tqdm import tqdm

# args needed to specify
action_json = r'path/to/action/preds.json'
verb_json = r'path/to/verb/preds.json'
noun_json = r'path/to/noun/preds.json'
result_file = r'path/to/full/preds.json'

print("Loading...")

action = json.load(open(action_json))
verb = json.load(open(verb_json))
noun = json.load(open(noun_json))

with tqdm(total=len(action["results"].keys()), desc="Processing", ncols=100) as pbar:
    for i in action["results"].keys():
        action["results"][i]["verb"] = verb["results"][i]["action"]
        action["results"][i]["object"] = noun["results"][i]["action"]
        pbar.update(1)

print("Saving...")

out = open(result_file, 'w')
json.dump(action, out)