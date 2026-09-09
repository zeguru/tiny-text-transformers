from pathlib import Path


class TrainingLogger:

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(self, message):
        message = str(message)
        print(message)

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(message + "\n")


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
        )



def separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)