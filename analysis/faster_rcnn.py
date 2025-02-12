import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import COLORS, STYLE_CONFIG, ANNOTATION, PLOT_PADDING
from plots.base import set_plot_style, create_figure, add_grid

def add_best_point_annotation_custom(ax, x_pos, y_pos):
    """Custom annotation function with better vertical spacing"""
    # Add text with more vertical space
    ax.text(
        x_pos,
        y_pos + 200,  # Increased vertical offset for "Best split"
        'Best split',
        ha='center',
        va='bottom',
        fontsize=6  # Reduced font size
    )
    
    # Add value with medium vertical space
    ax.text(
        x_pos,
        y_pos + 140,  # Increased vertical offset for the value
        f'{y_pos:.1f}s',
        ha='center',
        va='bottom',
        fontsize=6  # Reduced font size
    )
    
    # Add star closer to the bar
    ax.plot(
        x_pos,
        y_pos + 40,  # Moved up from +20 to +40
        marker='*',
        markersize=12,  # Reduced from 15 to 12
        color='gold',
        markeredgecolor='black',
        markeredgewidth=0.5,
        zorder=5
    )

def create_faster_rcnn_plot(excel_path):
    # Set plot style
    set_plot_style()
    
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Combine Host and Compression time
    df['Combined Host Time'] = df['Total Host Time'] + df['Total Compression Time']
    
    # Create figure and axis with compact sizing for 2-column layout
    fig, ax = plt.subplots(figsize=(3.5, 2))
    
    # Define colors using constants
    colors = {
        'Host': COLORS['mobile'],
        'Travel': COLORS['communication'],
        'Server': COLORS['server']
    }
    
    # Create stacked bar chart
    bottom = np.zeros(len(df))
    
    # Combined Host Time (Host + Compression)
    host = ax.bar(df['Split Layer'], df['Combined Host Time'], 
                 bottom=bottom, color=colors['Host'], label='Host')
    bottom += df['Combined Host Time']
    
    # Travel Time
    travel = ax.bar(df['Split Layer'], df['Total Travel Time'],
                   bottom=bottom, color=colors['Travel'], label='Transmission')
    bottom += df['Total Travel Time']
    
    # Server Time
    server = ax.bar(df['Split Layer'], df['Total Server Time'],
                   bottom=bottom, color=colors['Server'], label='Server')
    
    # Add best split annotation with custom spacing
    best_split_idx = 0  # Index for split layer 1
    best_split_time = df.iloc[best_split_idx]['Total Time']
    add_best_point_annotation_custom(
        ax=ax,
        x_pos=1,
        y_pos=best_split_time,
    )
    
    # Customize the plot
    ax.set_xlabel('Split Layer')
    ax.set_ylabel('Time (seconds)')
    
    # Add grid with professional styling
    add_grid(ax, axis='y')
    
    # Customize ticks
    ax.set_xticks(df['Split Layer'])
    
    # Add legend with professional styling - moved to top of plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), 
             ncol=3, fontsize=7, columnspacing=1)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout with minimal padding
    plt.tight_layout(pad=0.1)
    
    return fig

def save_plot(excel_path, output_path):
    """
    Create and save the Faster R-CNN plot
    
    Args:
        excel_path (str): Path to Excel file containing the data
        output_path (str): Path where to save the plot
    """
    fig = create_faster_rcnn_plot(excel_path)
    fig.savefig(output_path, 
                bbox_inches='tight',
                pad_inches=0.02,
                dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    # Example usage
    excel_path = "results/fasterrcnn_split/results.xlsx"
    output_path = "results/fasterrcnn_split/plot.png"
    save_plot(excel_path, output_path)
