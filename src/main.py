import sys
from transform.pipeline import run_transform_pipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/main.py <path_to_zip_file>")
        sys.exit(1)

    zip_path = sys.argv[1]
    print(f"Starting pipeline for: {zip_path}")
    run_transform_pipeline(zip_path)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()