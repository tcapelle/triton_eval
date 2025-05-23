from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import simple_parsing as sp
from datasets import load_dataset, IterableDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import random
import sys

# Initialize Rich console
console = Console()

@dataclass
class Args:
    dataset_name: str = sp.field(default="tcapelle/cuda-optimized-models", positional=True, help="Name of the HuggingFace dataset to load")
    split: str = sp.field(default="train", alias="-s", help="Dataset split to use")
    num_rows: int = sp.field(default=10, alias="-n", help="Number of rows to display")
    columns: Optional[List[str]] = sp.field(default=None, alias="-c", help="Specific columns to display (displays all if None)")
    full: bool = sp.field(default=False, alias="-f", help="Display full content of cells without truncation")
    random: bool = False # "Display random rows"
    stream_dataset: bool = True # "Stream the dataset instead of downloading"
    chat: Optional[str] = sp.field(default=None, help="Column name containing chat messages - displays only this column in conversation format")
    truncate_length: int = sp.field(default=100, alias="-t", help="Max number of characters to display for a cell when not in full mode")


def get_dataset(args: Args) -> Tuple[Optional[List[dict]], Optional[dict], Optional[Any]]:
    """Loads the dataset, applies sampling/shuffling for the initial view, 
       and returns initial rows, features, and the base dataset object.
    """
    console.print(Panel.fit(f"Loading dataset: [bold]{args.dataset_name}[/bold]", 
                            title="Dataset Viewer", 
                            border_style="green"))
    original_dataset_obj = None
    initial_rows_list = []
    features = None

    try:
        original_dataset_obj = load_dataset(args.dataset_name, split=args.split, streaming=args.stream_dataset)
        features = original_dataset_obj.features

        if args.stream_dataset:
            console.print("Dataset is being streamed. Total number of examples is unknown.")
            # Fetch initial batch for display
            temp_iterable = original_dataset_obj
            if args.random:
                console.print(f"[yellow]Note: Random sampling for initial view with streaming shuffles a buffer (size=1000) and takes {args.num_rows} samples. Interactive 'head' will show actual head.[/yellow]")
                buffer_size = 1000 
                temp_iterable = temp_iterable.shuffle(buffer_size=buffer_size)
            initial_rows_list = list(temp_iterable.take(args.num_rows))
        else:
            console.print(f"Total examples: [bold]{len(original_dataset_obj)}[/bold]")
            num_samples = min(args.num_rows, len(original_dataset_obj))
            if args.random:
                indices = random.sample(range(len(original_dataset_obj)), num_samples)
                initial_rows_list = [original_dataset_obj[i] for i in indices]
            else:
                initial_rows_list = [original_dataset_obj[i] for i in range(num_samples)]
        
        return initial_rows_list, features, original_dataset_obj

    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {str(e)}")
        return None, None, None


def get_columns_to_display(args: Args, dataset_features: dict) -> List[str]:
    """Determines which columns to display based on user arguments and dataset features."""
    all_columns = list(dataset_features.keys())
    if args.columns:
        columns_to_show = [col for col in args.columns if col in all_columns]
        if not columns_to_show:
            console.print("[yellow]Warning: None of the specified columns exist in the dataset. Showing all columns.[/yellow]")
            columns_to_show = all_columns
    else:
        columns_to_show = all_columns
    return columns_to_show


def prepare_table_rows(dataset_iterable, columns_to_show: List[str], args: Args) -> List[List[str]]:
    """Prepares and formats row data for the dataset preview table."""
    processed_rows = []
    example_count = 0
    for item in dataset_iterable: # dataset_iterable is already limited by .take() or sampling
        if example_count >= args.num_rows: # Double check, though mostly handled by get_dataset
            break

        row_values = []
        for col_name in columns_to_show:
            value = item.get(col_name)
            
            if isinstance(value, (list, dict)):
                display_value = str(value)
                if not args.full and len(display_value) > args.truncate_length:
                    display_value = display_value[:args.truncate_length] + "..."
            elif isinstance(value, str):
                display_value = value
                if not args.full and len(display_value) > args.truncate_length:
                    display_value = display_value[:args.truncate_length] + "..."
            else:
                display_value = str(value) if value is not None else ""
            row_values.append(display_value)
        processed_rows.append(row_values)
        example_count += 1
    return processed_rows


def create_dataset_preview_table(columns_to_show: List[str], processed_rows: List[List[str]], title: str, args: Args) -> Optional[Table]:
    """Creates the rich Table object for dataset preview."""
    if not processed_rows:
        return None

    table = Table(title=title, show_lines=True, expand=True)
    for col_name in columns_to_show:
        table.add_column(col_name, overflow="fold")
    
    for row_data in processed_rows:
        table.add_row(*row_data)
    return table


def _display_data(rows_to_display: List[dict], args: Args, columns_to_show: List[str], title_prefix: str = ""):
    """Helper function to display a list of rows either as a table or simple chat format."""
    if not rows_to_display:
        console.print(f"[yellow]{title_prefix}No data to display.[/yellow]")
        return

    # If --chat is specified, only display that column in chat format
    if args.chat is not None:
        if args.chat not in rows_to_display[0]:
            console.print(f"[red]Error: Column '{args.chat}' not found in dataset.[/red]")
            return
            
        panel_title = f"{title_prefix}Chat conversations from column: [bold]{args.chat}[/bold]"
        console.print(Panel.fit(panel_title, border_style="blue"))
        
        for item_idx, item_data in enumerate(rows_to_display):
            value = item_data.get(args.chat)
            if isinstance(value, list) and len(value) > 0:
                console.print(f"\n[bold cyan]Example {item_idx + 1}:[/bold cyan]")
                console.rule(style="dim")
                
                for msg_idx, msg in enumerate(value):
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        role = msg['role']
                        content = msg['content']
                        console.print(f"[bold]{role.upper()}:[/bold]")
                        console.print(content)
                        if msg_idx < len(value) - 1:  # Add separator between messages
                            console.print("─" * 50)
                    else:
                        # Fallback: show the raw message if it doesn't have expected structure
                        console.print(f"Message {msg_idx + 1}:")
                        console.print(msg)
                        if msg_idx < len(value) - 1:
                            console.print("─" * 50)
            else:
                console.print(f"[yellow]Example {item_idx + 1}: No chat messages or invalid format[/yellow]")
                if value is not None:
                    console.print(f"Raw value: {value}")
    else:
        # Regular table display
        processed_rows = prepare_table_rows(rows_to_display, columns_to_show, args)
        table_title = f"{title_prefix}Dataset Preview"
        if hasattr(args, 'split'):
            table_title += f" ({args.split} split)"
        preview_table = create_dataset_preview_table(columns_to_show, processed_rows, table_title, args)
        if preview_table:
            console.print(preview_table)


def main():
    args = sp.parse(Args)
    
    initial_rows_list, dataset_features, original_dataset_obj = get_dataset(args)
    
    if initial_rows_list is None or dataset_features is None or original_dataset_obj is None:
        sys.exit(1) 
    
    current_offset = len(initial_rows_list) 

    try:
        columns_to_show = get_columns_to_display(args, dataset_features)
        all_columns = list(dataset_features.keys()) 
        
        # If --chat is specified, ignore column selection and show all columns info but only display the chat column
        if args.chat is not None:
            if args.chat not in all_columns:
                console.print(f"[bold red]Error: Chat column '{args.chat}' not found in dataset.[/bold red]")
                console.print(f"Available columns: {', '.join(all_columns)}")
                sys.exit(1)
            # We still show all column info for reference, but will only display the chat column
        
        console.print(Panel.fit("Dataset Columns Information", border_style="blue"))
        column_table = Table(show_header=True)
        column_table.add_column("Column")
        column_table.add_column("Type")
        for col_info_name in all_columns:
            col_type = str(dataset_features[col_info_name])
            # Highlight the chat column if specified
            if args.chat and col_info_name == args.chat:
                column_table.add_row(f"[bold green]{col_info_name}[/bold green] (chat)", col_type)
            else:
                column_table.add_row(col_info_name, col_type)
        console.print(column_table)
        console.print("\n") 

        _display_data(initial_rows_list, args, columns_to_show, title_prefix="Initial Batch: ")

        interactive_loop(args, original_dataset_obj, dataset_features, columns_to_show, current_offset)
        
        sys.exit(0) 
    except Exception as e:
        console.print(f"[bold red]An error occurred during processing:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc() 
        sys.exit(1) 


def interactive_loop(args: Args, original_dataset_obj: Any, dataset_features: dict, columns_to_show: List[str], initial_offset: int):
    console.print("[bold cyan]Entering interactive mode...[/bold cyan]")
    current_offset = initial_offset

    from rich.prompt import Prompt

    is_streaming = isinstance(original_dataset_obj, IterableDataset)

    while True:
        console.rule("[bold yellow]Interactive Mode[/bold yellow]")
        command_str = Prompt.ask(
            "Press Enter for next batch, or 'q' to quit",
            default="", # Default to empty string for Enter key
            show_default=False
        ).strip().lower()

        if command_str == "q":
            console.print("[bold cyan]Exiting interactive mode.[/bold cyan]")
            break
        elif command_str == "" or command_str == "n": # Enter or 'n' for next
            if is_streaming:
                # For IterableDataset, we skip then take. This re-iterates from beginning if not careful.
                # A true streaming approach would be to keep an iterator and pull `next()` N times.
                # However, dataset.skip().take() is the provided API.
                # Ensure the original_dataset_obj is the one that can be repeatedly queried.
                next_batch = list(original_dataset_obj.skip(current_offset).take(args.num_rows))
            else:
                # For non-streaming (Map-style datasets)
                if current_offset >= len(original_dataset_obj):
                    console.print("[yellow]No more data to display.[/yellow]")
                    continue
                next_batch = original_dataset_obj[current_offset : current_offset + args.num_rows]
            
            if not next_batch:
                console.print("[yellow]No more data to display.[/yellow]")
                # Optionally, could break here or offer other commands if we had them.
                continue # Or break, depending on desired behavior when end is reached.

            title = f"Rows {current_offset + 1}-{current_offset + len(next_batch)}: "
            _display_data(next_batch, args, columns_to_show, title_prefix=title)
            current_offset += len(next_batch)
        else:
            console.print(f"[red]Unknown command: '{command_str}'. Press Enter for next, or 'q' to quit.[/red]")


if __name__ == "__main__":
    main() 