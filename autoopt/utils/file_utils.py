import json


def save_json(obj: object, path: str):
    with open(path, "w") as file:
        json.dump(obj, file)
