import numpy as np
from glob import glob
import json

counter = 0
json_file = glob("maskdetection-3" + "/*/" + "/_annotations.coco.json")


for file in json_file:
    f = open(file)
    print(file)
    newfile_name = file.replace("annotations.coco.json", "clean_annotations.coco.json")
    print(newfile_name)
    data = json.load(f)

    for i in data["annotations"]:
        if i["bbox"][0] + i["bbox"][2] > 79:
            i["bbox"][2] = i["bbox"][2] - 2
            counter += 1
        if i["bbox"][1] + i["bbox"][3] > 59:
            i["bbox"][3] = i["bbox"][3] - 2
            counter += 1

    with open(newfile_name, "w") as outfile:
        json.dump(data, outfile)


print("BBOX has been modified {} times".format(counter))
