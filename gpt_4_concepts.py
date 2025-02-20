import os
import json
import openai
import time
import sys

def get_concepts(dataset_name, path_class, type_name):
    dataset = dataset_name
    prompt_type = type_name

    openai.api_key = open(os.path.join(os.path.expanduser("~/workspace"), "openai.txt"), "r").read()[:]

    prompts = {
        "important" : "List the most important features for recognizing something as a \"goldfish\":\n\n-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\nList the most important features for recognizing something as a \"beerglass\":\n\n-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\nList the most important features for recognizing something as a \"{}\":",
        "superclass" : "Give superclasses for the word \"tench\":\n\n-fish\n-vertebrate\n-animal\n\nGive superclasses for the word \"beer glass\":\n\n-glass\n-container\n-object\n\nGive superclasses for the word \"{}\":",
        "around" : "List the things most commonly seen around a \"tench\":\n\n- a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\nList the things most commonly seen around a \"beer glass\":\n\n- beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\nList the things most commonly seen around a \"{}\":"
    }

    base_prompt = prompts[prompt_type]

    text_file_path = path_class
    class_names = []
    with open(text_file_path, "r") as file:
        class_names = file.read().splitlines()
    classes=[]
    
    for c in class_names:
        classes.append(str(c.split('\t')[0]))
    classes = classes[1:]

    MAX_TOKEN_PER_MIN = 40
    INTERVAL = 60.0 / MAX_TOKEN_PER_MIN
    feature_dict = {}

    for i, label in enumerate(classes):
        feature_dict[label] = set()
        print("\n", i, label)
        for _ in range(2):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant that provides visual descriptions of objects or actions. Use only adjectives and nouns in your description. Ensure each description is unique, short, and direct. Do not use qualifiers like 'typically', 'generally', or similar words."},
                    {"role": "user", "content": base_prompt.format(label)}],
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            #clean up responses
            features = response["choices"][0]["message"]["content"]
            features = features.split("\n-")
            features = [feat.replace("\n", "") for feat in features]
            features = [feat.strip() for feat in features]
            features = [feat for feat in features if len(feat)>0]
            features = [feat[1:] if feat[0] == '-' else feat for feat in features]
            features = set(features)
            feature_dict[label].update(features)
        time.sleep(INTERVAL)
        feature_dict[label] = sorted(list(feature_dict[label]) + [label])

    json_object = json.dumps(feature_dict, indent=4)
    with open("gpt4_{}_{}.json".format(dataset, prompt_type), "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
   dataset_name = sys.argv[1]
   path_class = sys.argv[2]
   type_name = sys.argv[3] 
   get_concepts(dataset_name, path_class, type_name)