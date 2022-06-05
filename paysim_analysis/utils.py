import pathlib


def get_project_folder():
    return pathlib.Path(__file__).parent.resolve()
