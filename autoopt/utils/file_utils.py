import json


def save_json(obj: object, path: str):
    with open(path, "w") as file:
        json.dump(obj, file)


def read_json(path: str):
    with open(path) as file:
        return json.load(file)
