import argparse
from pathlib import Path

from src.bdd100k import DATA
from src.detector import CONFIDENCE, MODELS, Setting, inference_device
from src.experiment import FIELDS, RUNS, SEED, SUMMARY, DEFAULT_IMAGES, VALUES, experiment_settings, run_experiment
from src.plots import plot_saved_results


def ask(prompt, default, cast=str):
    value = input(f"{prompt} [{default}]: ").strip()
    return cast(value) if value else default


def choose(prompt, options, default=0, back=False):
    """Return the value of a numbered option. Enter accepts the displayed default."""
    print(prompt)
    for number, (label, _) in enumerate(options, 1):
        print(f"  {number}. {label}")
    if back:
        print("0. Back")
    while True:
        number = ask("Option", str(default + 1))
        if back and number == "0":
            return None
        if number.isdecimal() and 1 <= int(number) <= len(options):
            return options[int(number) - 1][1]
        print(f"Enter a number from 1 to {len(options)}.")


def choose_number(prompt, presets, default, cast):
    options = [(str(value), value) for value in presets]
    selected = choose(prompt, options + [("Enter my own value", None)], presets.index(default))
    return ask(prompt, default, cast) if selected is None else selected


def ask_settings(mode=None):
    model, size, confidence = MODELS[0], 640, CONFIDENCE
    if mode != "model":
        model = choose("Detector", [(name, name) for name in MODELS])
    if mode != "resolution":
        size = choose_number("Input size (px)", VALUES["resolution"], size, int)
    if mode != "confidence":
        confidence = choose_number("Confidence threshold", VALUES["confidence"], confidence, float)
    return Setting(model, size, confidence)


def ask_values(mode):
    defaults = VALUES[mode]
    if mode == "model":
        return defaults  # Both available models take part in this comparison.
    label = ", ".join(map(str, defaults))
    if choose("Values to compare", [(label, True), ("Enter my own list", False)]):
        return defaults
    raw = ask("Values separated by commas", label)
    convert = int if mode == "resolution" else float
    return [convert(value.strip()) for value in raw.split(",")]


def run_and_plot(mode, base, values, count, seed):
    folder = run_experiment(count, mode=mode, base=base, values=values, seed=seed)
    plot_saved_results(folder)


def choose_saved_run():
    folders = [SUMMARY] if (SUMMARY / "run.json").is_file() else []
    folders.extend(sorted(path.parent for path in RUNS.glob("*/run.json")))
    if not folders:
        print("No saved runs.")
        return
    options = [(folder.name, folder) for folder in folders]
    folder = choose("Saved run", options, default=len(options) - 1, back=True)
    if folder is not None:
        plot_saved_results(folder)



def ask_count(prompt, available, default):
    while True:
        value = ask(f"{prompt} (1–{available:,}, 0 = all)", str(min(default, available)))
        if value.isdecimal() and int(value) <= available:
            return int(value)
        print(f"Enter a number between 0 and {available:,}.")


def choose_experiment():
    mode = choose("Experiment", [
        ("Resolution: quality and speed", "resolution"),
        ("Confidence: precision and recall", "confidence"),
        ("Model: quality and speed", "model"),
    ], back=True)
    if mode is None:
        return None
    values = ask_values(mode)
    base = ask_settings(mode)
    experiment_settings(mode, base, values)
    return mode, base, values


def run_menu():
    total = sum(1 for _ in (DATA / "images").glob("*.jpg"))
    print(f"Inference device: {inference_device()}")
    while True:
        try:
            action = choose("Experiments", [("New experiment", "run"), ("Plot a saved run", "plots")], back=True)
            if action is None:
                return
            if action == "plots":
                choose_saved_run()
                continue
            count = ask_count("How many photos per configuration?", total, DEFAULT_IMAGES)
            seed = ask("Sampling seed", SEED, int)
            selected = choose_experiment()
            if selected is None:
                continue
            mode, base, values = selected
            print(f"Compare {mode} on {count or total:,} photos from BDD100K.")
            if choose("Start experiment?", [("Run", True), ("Back to menu", False)]):
                run_and_plot(mode, base, values, count, seed)
        except EOFError:
            return
        except (OSError, ValueError) as error:
            print(f"Error: {error}")


def add_settings(parser):
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)


def main():
    parser = argparse.ArgumentParser(description="Object detection quality and speed on BDD100K.")
    commands = parser.add_subparsers(dest="action")
    experiment = commands.add_parser("experiment", help="Run a controlled comparison.")
    add_settings(experiment)
    experiment.add_argument("--mode", choices=FIELDS, default="resolution")
    experiment.add_argument("--sizes", type=int, nargs="+")
    experiment.add_argument("--confidences", type=float, nargs="+")
    experiment.add_argument("--models", choices=MODELS, nargs="+")
    experiment.add_argument("--seed", type=int, default=SEED)
    experiment.add_argument("--images", type=int, default=DEFAULT_IMAGES, help="Use 0 for all images.")

    plots = commands.add_parser("plots", help="Regenerate charts from a saved run.")
    plots.add_argument("folder", type=Path, nargs="?", default=SUMMARY)
    args = parser.parse_args()
    try:
        if args.action is None:
            run_menu()
        elif args.action == "experiment":
            comparison = {"resolution": args.sizes, "confidence": args.confidences, "model": args.models}
            if any(values is not None for mode, values in comparison.items() if mode != args.mode):
                parser.error("Choose the values flag that matches the experiment mode.")
            run_and_plot(args.mode, Setting(args.model, args.size, args.confidence),
                         comparison[args.mode], args.images, args.seed)
        elif args.action == "plots":
            plot_saved_results(args.folder)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
