"""
Module: log_plot
---------------------------
Processes and visualizes training process logs.

Intended usage:
---------------
Run as a standalone script with command‑line arguments to specify:
- Input log paths (directly or via directory + file suffix search)
- Output directory for generated plots
- Styling and DPI preferences
"""

import argparse
import ast
import os
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for better font rendering and Chinese support
matplotlib.use("Agg")  # Use non-interactive backend for better performance
rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
    "SimHei",
    "PingFang SC",
]
rcParams["axes.unicode_minus"] = False  # Fix minus sign display
rcParams["figure.dpi"] = 100  # Default DPI for better quality
rcParams["savefig.dpi"] = 300  # High DPI for saved figures
rcParams["savefig.bbox"] = "standard"  # Standard bounding box (no auto-cropping)
rcParams["font.size"] = 10  # Default font size

def find_log_file(directory, ending_string):
    """
    Finds a file in the specified directory that ends with a specific string and .out.

    Args:
      directory: The path of the directory to search.
      ending_string: The numeric string that the filename should end with.

    Returns:
      The full path of the file if found, otherwise None.
    """
    for filename in os.listdir(directory):
        if filename.endswith(f"{ending_string}.out"):
            return os.path.join(directory, filename)
    return None


def extract_and_write_logs(input_path, output_path):
    """
    Finds specific patterns in the input log file, extracts matching content,
    and writes it to the output file. This version can handle complex line
    prefixes, including ANSI color codes.

    Args:
        input_path (str): The path of the original input log file.
        output_path (str): The path of the output file to write the cleaned logs to.
    """
    # Define a powerful regular expression using '|' (or) to match any of the three patterns.
    # The entire expression is wrapped in a capture group '(...)' to directly extract this part.
    extraction_pattern = re.compile(
        r"(rollout \d+: .*|step \d+: {'train/loss':.*|perf \d+: {'perf/sleep_time':.*)"
    )

    print(f"--- STAGE 1: Starting Log Cleaning ---")
    print(f"Reading original log: '{input_path}'")

    try:
        # Use a 'with' statement to safely open both input and output files
        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(output_path, "w", encoding="utf-8") as outfile,
        ):
            found_lines = 0
            for line in infile:
                # Search for the pattern we want to extract in each line
                match = extraction_pattern.search(line)
                if match:
                    # 'match.group(1)' returns the content of the first capture group
                    content_to_write = match.group(1)
                    outfile.write(content_to_write.strip() + "\n")
                    found_lines += 1

        if found_lines > 0:
            print(
                f"Processing complete! Successfully extracted {found_lines} lines and wrote to cleaned file: '{output_path}'"
            )
        else:
            print(
                f"Processing complete, but no matching log lines were found in '{input_path}'."
            )
        print("-" * 30 + "\n")

    except FileNotFoundError:
        print(f"Error: Input file not found '{input_path}'")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


# ==============================================================================
# --- STAGE 2: Data Parsing and Plotting ---
# ==============================================================================


def parse_log_file(file_path, label=None):
    """
    Parses a cleaned log file to extract data for all steps.

    Args:
        file_path (str): The path of the cleaned log file.
        label (str, optional): Label for this dataset (used for legends when plotting multiple files).

    Returns:
        tuple: A tuple containing three elements (data, steps, label).
               - data (dict): Metric data organized by category ('rollout', 'train', 'perf').
               - steps (dict): A list of step numbers organized by category.
               - label (str): Label for this dataset.
               Returns (None, None, None) if the file is not found or a parsing error occurs.
    """
    if label is None:
        label = os.path.basename(file_path)

    print(f"--- STAGE 2: Starting Parsing and Plotting ---")
    print(f"Reading cleaned log: '{file_path}' (label: {label})")

    data = {
        "rollout": defaultdict(list),
        "train": defaultdict(list),
        "perf": defaultdict(list),
    }
    steps = {"rollout": [], "train": [], "perf": []}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                match = re.match(r"^(rollout|step|perf)\s+(\d+):\s*(\{.*\})$", line)
                if not match:
                    continue

                category_raw, step_num_str, data_str = match.groups()
                step_num = int(step_num_str)
                # Classify 'step' as 'train'
                category = "train" if category_raw == "step" else category_raw

                try:
                    metrics_dict = ast.literal_eval(data_str)
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse line: {line}")
                    continue

                steps[category].append(step_num)
                for key, value in metrics_dict.items():
                    simple_key = key.split("/")[-1]
                    data[category][simple_key].append(value)

    except FileNotFoundError:
        print(
            f"Error: File '{file_path}' not found. Please ensure the filename is correct."
        )
        return None, None, None
    except Exception as e:
        print(f"An error occurred while reading or parsing the file: {e}")
        return None, None, None

    print("Log parsing successful.")
    return data, steps, label


def plot_all_data(
    category_name,
    datasets,
    output_dir,
    style="seaborn-v0_8-darkgrid",
    color_palette=None,
    title_prefix=None,
):
    """
    Creates individual plots for each metric of a specified category and saves them to separate files.

    Args:
        category_name (str): Name of the metric category (e.g., 'rollout', 'train', 'perf')
        datasets (list): List of tuples, each containing (metrics_data, steps_list, label)
                        - metrics_data (dict): Dictionary mapping metric names to their values
                        - steps_list (list): List of step numbers
                        - label (str): Label for this dataset
        output_dir (str): Directory to save the plots
        style (str): Matplotlib style to use (default: 'seaborn-v0_8-darkgrid')
        color_palette (list): Optional list of colors for the plots
        title_prefix (str): Optional prefix to add before the title
    """
    if not datasets or len(datasets) == 0:
        print(f"Info: No data found for category '{category_name}' to plot.")
        return

    # Collect all unique metric names across all datasets
    all_metric_names = set()
    for metrics_data, _, _ in datasets:
        all_metric_names.update(metrics_data.keys())

    num_metrics = len(all_metric_names)
    if num_metrics == 0:
        print(f"Info: No metrics found for category '{category_name}' to plot.")
        return

    # Default color palette if none provided
    if color_palette is None:
        # Use a nice color palette
        color_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get DPI from rcParams if available, otherwise use default
    save_dpi = rcParams.get("savefig.dpi", 300)

    # Use context manager for style to avoid affecting other plots
    with plt.style.context(style):
        # Create a separate plot for each metric
        for metric_name in sorted(all_metric_names):
            # Create a new figure for this metric
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot data from all datasets that have this metric
            for dataset_idx, (metrics_data, steps_list, label) in enumerate(datasets):
                if metric_name not in metrics_data:
                    continue

                values = metrics_data[metric_name]

                # Use color from palette (cycle through colors for different datasets)
                color = color_palette[dataset_idx % len(color_palette)]

                # Plot with improved styling
                ax.plot(
                    steps_list,
                    values,
                    marker="o",
                    markersize=3,
                    linestyle="-",
                    linewidth=1.5,
                    color=color,
                    alpha=0.8,
                    label=label,
                )

            # Improved title and labels
            if title_prefix:
                title = f"{title_prefix} {category_name} - {metric_name}"
            else:
                title = f"{category_name} - {metric_name}"
            ax.set_title(title, fontsize=14, weight="semibold", pad=10)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_xlabel("Training Step", fontsize=12)

            # Enhanced grid
            ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)  # Put grid behind plot

            # Improve tick labels
            ax.tick_params(axis="both", which="major", labelsize=10)

            # Add legend with better positioning (always show legend)
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

            # Auto-format y-axis for better readability
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))

            # Save the plot for this metric
            safe_metric_name = metric_name.replace("/", "_").replace(" ", "_")
            save_path = os.path.join(
                output_dir, f"{category_name}_{safe_metric_name}.png"
            )

            # Save with optimized settings (no tight bbox to keep consistent size)
            plt.savefig(
                save_path,
                dpi=save_dpi,
                bbox_inches=None,
                facecolor="white",
                edgecolor="none",
            )

            print(f"Plot successfully saved to: {save_path}")

            # Close the figure to free up memory
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean training logs and generate plots."
    )
    parser.add_argument(
        "--input-log",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to the original log file(s). Can provide multiple files to plot on the same graph. If omitted, provide --log-directory and --file-ending to locate it.",
    )
    parser.add_argument(
        "--log-directory",
        type=str,
        default=None,
        help="Directory to search when locating the input log by suffix.",
    )
    parser.add_argument(
        "--file-ending",
        type=str,
        default=None,
        help="Suffix (without .out) used to locate the log file within --log-directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./slime_out/",
        help="Directory where generated plots will be stored.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="classic",
        help="Matplotlib style to use for plots (e.g., 'seaborn-v0_8-darkgrid', 'ggplot', 'classic').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI (dots per inch) for saved figures (default: 300).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Custom labels for the plots. If provided, must match the number of input log files. If omitted, filenames will be used as labels.",
    )
    parser.add_argument(
        "--colors",
        type=str,
        nargs="+",
        default=None,
        help="Custom colors for the plots (e.g., '#1f77b4' 'red' 'blue'). If provided, must match the number of input log files. If omitted, default color palette will be used.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default=None,
        help="Prefix to add before plot titles (e.g., 'Experiment 1:'). If omitted, titles remain unchanged.",
    )

    args = parser.parse_args()

    input_log_files = args.input_log
    if not input_log_files and args.log_directory and args.file_ending:
        input_log_file = find_log_file(args.log_directory, args.file_ending)
        if input_log_file is None:
            print(
                f"Error: Log file ending with '{args.file_ending}.out' not found in '{args.log_directory}'."
            )
            raise SystemExit(1)
        input_log_files = [input_log_file]

    if not input_log_files:
        print(
            "Error: Please provide --input-log or both --log-directory and --file-ending to locate the log file."
        )
        raise SystemExit(1)

    # Validate labels if provided
    if args.labels is not None:
        if len(args.labels) != len(input_log_files):
            print(
                f"Error: Number of labels ({len(args.labels)}) must match number of input log files ({len(input_log_files)})."
            )
            raise SystemExit(1)

    # Validate colors if provided
    if args.colors is not None:
        if len(args.colors) != len(input_log_files):
            print(
                f"Error: Number of colors ({len(args.colors)}) must match number of input log files ({len(input_log_files)})."
            )
            raise SystemExit(1)

    # Update DPI setting if specified
    if args.dpi:
        rcParams["savefig.dpi"] = args.dpi

    # Process all input log files
    all_datasets = {
        "rollout": [],
        "train": [],
        "perf": [],
    }

    for idx, input_log_file in enumerate(input_log_files):
        print(f"\n{'=' * 60}")
        print(f"Processing log file: {input_log_file}")
        print(f"{'=' * 60}")

        # Stage 1: Extract data from the original log and write to a new file
        cleaned_log_path = f"{input_log_file}_cleaned"
        extract_and_write_logs(input_log_file, cleaned_log_path)

        # Stage 2: Parse the cleaned log
        # Use custom label if provided, otherwise use the original filename (without path)
        if args.labels is not None:
            label = args.labels[idx]
        else:
            label = os.path.basename(input_log_file)
        all_data, all_steps, label = parse_log_file(cleaned_log_path, label=label)

        if all_data and all_steps:
            # Add this dataset to our collection
            for category in ["rollout", "train", "perf"]:
                if all_data[category]:
                    all_datasets[category].append(
                        (all_data[category], all_steps[category], label)
                    )
        else:
            print(f"Warning: Could not parse data from '{input_log_file}'")

    # Stage 3: Generate plots with all datasets
    if any(all_datasets.values()):
        print(f"\n{'=' * 60}")
        print("Generating combined plots...")
        print(f"{'=' * 60}\n")

        for category in ["rollout", "train", "perf"]:
            if all_datasets[category]:
                plot_all_data(
                    category,
                    all_datasets[category],
                    args.output_dir,
                    style=args.style,
                    color_palette=args.colors,
                    title_prefix=args.title_prefix,
                )

        print("\n--- All plots have been generated ---")
    else:
        print(
            "\nPlotting step was skipped because no data could be parsed from any file."
        )
