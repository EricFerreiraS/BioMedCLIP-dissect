import json
import conceptset_utils_v2 as conceptset_utils
import sys

def concept_processor(dataset_name, path_class):
    """
    CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
    OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
    MAX_LEN: max number of characters in a concept

    PRINT_PROB: what percentage of filtered concepts will be printed
    """

    CLASS_SIM_CUTOFF = 0.85
    OTHER_SIM_CUTOFF = 0.9
    MAX_LEN = 30
    PRINT_PROB = 1

    dataset = dataset_name
    device = "cuda:1"
    gpt_model = "gpt4"

    save_name = "{}_filtered_remove_target_gpt4.txt".format(dataset)

    with open("{}_{}_important.json".format(gpt_model, dataset), "r") as f:
        important_dict = json.load(f)
    with open("{}_{}_superclass.json".format(gpt_model, dataset), "r") as f:
        superclass_dict = json.load(f)
    with open("{}_{}_around.json".format(gpt_model, dataset), "r") as f:
        around_dict = json.load(f)
        
    text_file_path = path_class
    class_names = []
    with open(text_file_path, "r") as file:
        class_names = file.read().splitlines()
    classes=[]
    for c in class_names:
        classes.append(str(c.split('\t')[0]))
    classes = classes[1:]

    concepts = set()

    for values in important_dict.values():
        concepts.update(set(values))

    for values in superclass_dict.values():
        concepts.update(set(values))
        
    for values in around_dict.values():
        concepts.update(set(values))

    concepts = conceptset_utils.remove_too_long(concepts, MAX_LEN, PRINT_PROB)

    concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, CLASS_SIM_CUTOFF, device, PRINT_PROB)

    concepts = conceptset_utils.filter_too_similar(concepts, OTHER_SIM_CUTOFF, device, PRINT_PROB)

    with open(save_name, "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)

if __name__ == "__main__":
   dataset_name = sys.argv[1]
   path_class = sys.argv[2] 
   concept_processor(dataset_name, path_class)