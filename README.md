# 🎾 My Digital Tennis Coach Pipeline

This is the pipeline for [MyDigitalTennisCoach](https://apps.apple.com/de/app/my-digital-tennis-coach/id1631299737).  
This project demonstrates a full **data science pipeline** built in Python, showcasing clean engineering practices and modular design. The goal is to process raw match data collected from IoT sensors, extract meaningful insights, and generate statistical summaries using both feature engineering and applied machine learning models.

---

## 🧠 Purpose

The primary aim of this repository is to demonstrate my ability to design and build scalable, testable, and modular data pipelines.

This pipeline turns noisy, zipped JSON sensor data into structured summaries ready for app ingestion.

---

## 🏗️ Project Structure

```bash
tennis-pipeline/
├── data/
│   ├── raw/            # Input zip files containing raw IoT JSON data
│   ├── tmp/            # Temporary extracted files (auto-cleared after runs)
│   └── outputs/        # Final processed outputs and model summaries
│
├── models/             # Pickled ML models (e.g., serve classifiers)
│
├── src/
│   ├── main.py         # Entry point to run the full pipeline
│   │
│   ├── transform/      # Extract and process raw sensor data
│   │   ├── Extract.py
│   │   ├── Feature_Gen.py
│   │   ├── Apply_Models.py
│   │   ├── Blend_Touch_wFeatures.py
│   │   ├── Create_Player_SummaryStats.py
│   │   └── pipeline.py
│   │
│   ├── summarise/      # Build and export summary statistics
│   │   ├── Create_KPIs.py
│   │   ├── Create_Output.py
│   │   └── pipeline.py
│   │
│   ├── utils/          # Reusable utility functions (e.g., file I/O)
│   │   └── io_helpers.py
│
└── README.md

```
## 🔁 Pipeline Overview

The process flows as follows:

1. **Extract**
   - Unzips raw IoT match data
   - Loads JSON sensor and metadata files
   - Builds an initial structured DataFrame from raw time-series

2. **Feature Engineering**
   - Adds biomechanical, match-level, and rally-based features
   - Breaks down data into shots, points, and sequences
   - Prepares features for model input

3. **Model Application**
   - Applies multiple ML models:
     - Serve classification
     - Return quality prediction
     - Shot type prediction (FH/BH/Slice, etc.)
   - Appends predictions to the enriched dataset

4. **Generates Match Stats**
   - Outputs CSV summaries of serve, return, and rally effectiveness
   - Prepares stats for evaluation or reporting

5. **Evaluates and Creates Recommendations**
   - Assess performance vs benchmarks
   - Generates recommendations based on performance
   - Outputs a json file to be consumed by the app

---

## 🚀 Running the Pipeline

From the project root, run:

```bash
python3 src/main.py data/raw/match_001.zip
```
To also clear temporary files after processing:

```bash
python3 src/main.py data/raw/match_001.zip --clean

```

### 🧪 Example test data
This project includes a sample zip file for testing:
```bash
data/raw/241202_JJO_MT.zip
```
Try it with:
```bash
python3 src/main.py data/raw/241202_JJO_MT.zip

```
