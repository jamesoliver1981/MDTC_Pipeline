

def generate_variable_declarations(stats, suffix="single", multiline=False):
    """
    Prints a block of variable declarations (set to None) based on keys in `stats`.

    Args:
        stats (dict): The full stats dictionary returned by stat_func.
        suffix (str): The suffix to strip from each stat key.
        multiline (bool): If True, each variable on a new line.
    """

    base_names = [
        key[:-len(f"_{suffix}")]
        for key in stats
        if key.endswith(f"_{suffix}")
    ]

    if multiline:
        print("\n".join(f"{name} = None" for name in base_names))
    else:
        line = " = ".join(base_names) + " = None"
        print(line)

    # Optional: write to file
    with open("data/outputs/gen_out_variable_stub.txt", "w") as f:
        if multiline:
            f.write("\n".join(f"{name} = None" for name in base_names))
        else:
            f.write(line + "\n")

    print("\n✅ Variable declaration block saved to outputs/gen_out_variable_stub.txt")


