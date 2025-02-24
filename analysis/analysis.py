# analysis/analysis.py

import pandas as pd
import matplotlib.pyplot as plt


def analyze_energy_consumption(csv_path: str):
    """Analyze energy consumption metrics from inference data."""

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Calculate energy consumption (Power × Time) for each measurement
    df["gpu_energy"] = (
        df["gpu_power"] * df["inf_time_client"]
    )  # Joules = Watts × seconds
    df["battery_energy"] = df["battery_power"] * df["inf_time_client"]
    df["cpu_energy"] = df["cpu_power"] * df["inf_time_client"]

    # Calculate average energy consumption per layer
    layer_metrics = (
        df.groupby("layer_id")
        .agg(
            {
                "inf_time_client": ["mean", "std"],  # Layer execution time
                "gpu_energy": ["mean", "std"],
                "battery_energy": ["mean", "std"],
                "cpu_energy": ["mean", "std"],
                "battery_drain": ["mean", "std"],  # Battery percentage change
            }
        )
        .round(4)
    )

    # Find most efficient layers (using numeric columns only)
    numeric_cols = [
        "gpu_energy",
        "battery_energy",
        "gpu_power",
        "battery_power",
        "inf_time_client",
    ]
    avg_metrics = df.groupby("layer_id")[numeric_cols].mean()

    best_gpu_layer = avg_metrics["gpu_energy"].idxmin()
    best_battery_layer = avg_metrics["battery_energy"].idxmin()

    print("\nMost Energy-Efficient Layers:")
    print(f"Best GPU Layer: {best_gpu_layer}")
    print(f"  - Energy: {avg_metrics.loc[best_gpu_layer, 'gpu_energy']:.4f} Joules")
    print(f"  - Power: {avg_metrics.loc[best_gpu_layer, 'gpu_power']:.4f} Watts")
    print(f"  - Time: {avg_metrics.loc[best_gpu_layer, 'inf_time_client']:.4f} seconds")

    print(f"\nBest Battery Layer: {best_battery_layer}")
    print(
        f"  - Energy: {avg_metrics.loc[best_battery_layer, 'battery_energy']:.4f} Joules"
    )
    print(
        f"  - Power: {avg_metrics.loc[best_battery_layer, 'battery_power']:.4f} Watts"
    )
    print(
        f"  - Time: {avg_metrics.loc[best_battery_layer, 'inf_time_client']:.4f} seconds"
    )

    print("\nEnergy Consumption per Layer:")
    print(layer_metrics)

    # Calculate total energy consumption
    print("\nOverall Energy Statistics:")
    print(f"Total GPU Energy: {df['gpu_energy'].sum():.4f} Joules")
    print(f"Total Battery Energy: {df['battery_energy'].sum():.4f} Joules")
    print(f"Total CPU Energy: {df['cpu_energy'].sum():.4f} Joules")
    print(f"Total Battery Drain: {df['battery_drain'].sum():.4f}%")
    print(f"\nTotal Inference Time: {df['inf_time_client'].sum():.4f} seconds")

    # Visualization of energy consumption per layer
    plt.figure(figsize=(15, 10))

    # Energy plot
    plt.subplot(2, 1, 1)
    plt.plot(
        df["layer_id"].unique(),
        df.groupby("layer_id")["gpu_energy"].mean(),
        label="GPU Energy",
        marker="s",
    )
    plt.plot(
        df["layer_id"].unique(),
        df.groupby("layer_id")["battery_energy"].mean(),
        label="Battery Energy",
        marker="^",
    )

    # Highlight best layers
    plt.axvline(
        x=best_gpu_layer, color="g", linestyle="--", alpha=0.5, label="Best GPU Layer"
    )
    plt.axvline(
        x=best_battery_layer,
        color="r",
        linestyle="--",
        alpha=0.5,
        label="Best Battery Layer",
    )

    plt.xlabel("Layer ID")
    plt.ylabel("Energy (Joules)")
    plt.title("Average Energy Consumption per Layer")
    plt.legend()
    plt.grid(True)

    # Execution time plot
    plt.subplot(2, 1, 2)
    plt.plot(
        df["layer_id"].unique(),
        df.groupby("layer_id")["inf_time_client"].mean(),
        label="Execution Time",
        marker="o",
        color="red",
    )

    plt.xlabel("Layer ID")
    plt.ylabel("Time (seconds)")
    plt.title("Average Execution Time per Layer")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("energy_analysis.png")
    plt.close()


if __name__ == "__main__":
    # Replace with your CSV file path
    csv_path = "results/yolov5s_split/results.csv"
    analyze_energy_consumption(csv_path)
