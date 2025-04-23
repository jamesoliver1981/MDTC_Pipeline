import sys
from ExtractAndTransform.pipeline import run_transform_pipeline
from summarise.pipeline import run_summarise_pipeline
from utils.io_helpers import clear_tmp_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/main.py <path_to_zip_file>")
        sys.exit(1)

    zip_path = sys.argv[1]
    clean_flag = "--clean" in sys.argv
    print(f"Starting pipeline for: {zip_path}")
    eval_table, zip_name = run_transform_pipeline(zip_path)

    run_summarise_pipeline(eval_table, zip_name)

    print("Pipeline completed successfully.")

    if clean_flag:
        clear_tmp_data()

if __name__ == "__main__":
    main()