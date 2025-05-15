from dataclasses import dataclass, field
from typing import List, Optional
import simple_parsing as sp
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
import random

@dataclass
class Args:
    """Arguments for the dataset viewer script."""
    
    dataset_name: str = field(
        default="tcapelle/cuda-optimized-models",
        metadata={"help": "Name of the HuggingFace dataset to load"}
    )
    
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"}
    )
    
    num_rows: int = field(
        default=10,
        metadata={"help": "Number of rows to display"}
    )
    
    columns: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Specific columns to display (displays all if None)"}
    )
    
    full: bool = field(
        default=False,
        metadata={"help": "Display full content of cells without truncation"}
    )
    
    random: bool = field(
        default=False,
        metadata={"help": "Display random rows"}
    )


def format_messages_as_markdown(messages):
    """Format a messages list as a markdown conversation."""
    # Format the conversation in a more readable way
    conversation = ""
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # Add separator between messages
        if conversation:
            conversation += "\n---\n\n"
        
        conversation += f"**{role.upper()}**:\n\n{content}\n"
        
    return conversation


def main():
    # Parse arguments
    args = sp.parse(Args)
    
    # Initialize Rich console
    console = Console()
    
    # Load the dataset
    console.print(Panel.fit(f"Loading dataset: [bold]{args.dataset_name}[/bold]", 
                            title="Dataset Viewer", 
                            border_style="green"))
    
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
        console.print(f"Total examples: [bold]{len(dataset)}[/bold]")
        
        # Get columns to display
        all_columns = list(dataset.features.keys())
        if args.columns:
            columns_to_show = [col for col in args.columns if col in all_columns]
            if not columns_to_show:
                console.print("[yellow]Warning: None of the specified columns exist in the dataset. Showing all columns.[/yellow]")
                columns_to_show = all_columns
        else:
            columns_to_show = all_columns
        
        # Create a table for dataset preview
        table = Table(title=f"Dataset Preview ({args.split} split)")
        
        # Add columns to the table
        for col in columns_to_show:
            table.add_column(col, overflow="fold")
        
        # Add rows to the table
        num_samples = min(args.num_rows, len(dataset))
        if args.random:
            num_samples = min(args.num_rows, len(dataset))
            indices = random.sample(range(len(dataset)), num_samples)
        else:
            indices = range(num_samples)
        
        for i in indices:
            row_values = []
            for col in columns_to_show:
                value = dataset[i][col]
                
                # Special handling for "messages" column
                if col == "messages" and isinstance(value, list):
                    if args.full:
                        # Format messages as markdown
                        md_content = format_messages_as_markdown(value)
                        # If only the messages column is being displayed, render it directly
                        if len(columns_to_show) == 1:
                            console.print(Panel(Markdown(md_content), 
                                                title=f"Example {i} - Conversation", 
                                                border_style="blue"))
                        display_value = md_content
                    else:
                        display_value = f"{len(value)} messages. Use --full to see content."
                # Handle different data types appropriately for other columns
                elif isinstance(value, (list, dict)):
                    display_value = str(value)
                    if not args.full and len(display_value) > 100:
                        display_value = display_value[:100] + "..."
                elif isinstance(value, str):
                    display_value = value
                    if not args.full and len(display_value) > 100:
                        display_value = display_value[:100] + "..."
                else:
                    display_value = str(value)
                row_values.append(display_value)
            
            # Only add to table if we're not directly rendering messages
            if not (len(columns_to_show) == 1 and columns_to_show[0] == "messages" and args.full):
                table.add_row(*row_values)
        
        # Display the table only if we have rows
        if not (len(columns_to_show) == 1 and columns_to_show[0] == "messages" and args.full):
            console.print(table)
        
        # Display column information
        console.print(Panel.fit("Dataset Columns Information", border_style="blue"))
        column_table = Table(show_header=True)
        column_table.add_column("Column")
        column_table.add_column("Type")
        
        for col in all_columns:
            col_type = str(dataset.features[col])
            column_table.add_row(col, col_type)
        
        console.print(column_table)
        
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {str(e)}")


if __name__ == "__main__":
    main() 