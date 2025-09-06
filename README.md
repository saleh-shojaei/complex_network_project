## About Our Contributions

This project extends the original paper with several significant contributions:

### 🔧 **Enhanced Community Detection Pipeline**
- **Multiple Algorithm Support**: Implemented and compared three community detection algorithms:
  - **Louvain** (original paper's method)
  - **Leiden** (improved modularity optimization)
  - **BigCLAM-like** (overlapping community detection)
- **Robustness Analysis**: Multiple trials with different random seeds for statistical significance
- **Parameter Tuning**: Systematic exploration of resolution parameters across algorithms

###  **Advanced Geographic Visualization**
- **High-Quality Cartographic Maps**: Professional-grade visualizations using Natural Earth data
- **Multi-Scale Analysis**: County-level and state-level community analysis
- **Overlap Visualization**: Special handling and visualization of overlapping communities
- **Interactive Map Collages**: Automated generation of comparative visualizations

### **Comprehensive Validation Framework**
- **Geographic Validation**: Community-state alignment analysis using Adjusted Rand Index
- **Statistical Metrics**: AUROC, Average Precision, and modularity scores
- **Automated Reporting**: CSV and JSON output for all validation results
- **Cross-Algorithm Comparison**: Unified analysis across all implemented algorithms

### **Extended Analysis Capabilities**
- **Data Enrichment**: Automated geocoding using OpenStreetMap's Nominatim API
- **Overlapping Communities**: Support for multi-membership community detection
- **Performance Benchmarking**: Systematic comparison of algorithm performance
- **Reproducible Research**: Complete pipeline from raw data to publication-ready figures

---

## Project Structure

```
complex_network_project/
├── algorithms/                    # Community detection implementations
│   ├── bigCLAM/                  # BigCLAM-like overlapping community detection
│   │   ├── big_clam_like.py     # Main algorithm implementation
│   │   └── community_detection_outputs/  # Results and visualizations
│   ├── leiden/                   # Leiden algorithm implementation
│   │   ├── leiden.py            # Main algorithm implementation
│   │   └── community_detection_outputs/  # Results and visualizations
│   └── louvain/                  # Louvain algorithm implementation
│       ├── louvain.py           # Main algorithm implementation
│       └── community_detection_outputs/  # Results and visualizations
├── data_enrichment/              # Data preprocessing and augmentation
│   ├── augmentation.ipynb       # Geographic data enrichment
│   └── community_detection.ipynb # Basic community detection evaluation
├── dataset/                      # Raw network data
│   ├── Communication_Network.gml/  # Communication network data
│   └── Mobility_Network.gml/    # Mobility network data
├── show_on_map/                  # Geographic visualization tools
│   ├── show_on_map.py           # Detailed community mapping
│   ├── show_on_map_overlap.py   # County-level overlap analysis
│   └── downloadedMap/           # Cartographic data (Natural Earth)
├── validation/                   # Analysis and validation framework
│   ├── community_validation_report.py  # Validation report generation
│   ├── reports_aggregator.py    # Results aggregation
│   ├── reports_analysis.py      # Comprehensive analysis and visualization
│   └── figures/                 # Generated analysis figures
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Key Components:

- **`algorithms/`**: Contains three community detection implementations with their respective outputs
- **`data_enrichment/`**: Jupyter notebooks for data preprocessing and basic analysis
- **`show_on_map/`**: Geographic visualization tools with high-quality cartographic output
- **`validation/`**: Comprehensive validation framework with automated reporting
- **`dataset/`**: Raw network data in GML format

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd complex_network_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Geographic Data (Optional)
The visualization tools use Natural Earth data for high-quality maps. If you want to create maps, download the required shapefiles to the `show_on_map/downloadedMap/` directory:

- US States: `cb_2018_us_state_20m/`
- US Counties: `cb_2018_us_county_20m/`
- Natural Earth data: `ne_10m_*` directories

### 4. Verify Installation
```bash
python -c "import networkx, community, leidenalg, igraph, geopandas, matplotlib; print('All dependencies installed successfully!')"
```

---

## Usage

### Quick Start

1. **Data Enrichment** (if using raw data):
   ```bash
   jupyter notebook data_enrichment/augmentation.ipynb
   ```

2. **Run Community Detection**:
   ```bash
   # Louvain algorithm
   cd algorithms/louvain
   python louvain.py
   
   # Leiden algorithm
   cd algorithms/leiden
   python leiden.py
   
   # BigCLAM-like algorithm
   cd algorithms/bigCLAM
   python big_clam_like.py
   ```

3. **Generate Visualizations**:
   ```bash
   cd show_on_map
   python show_on_map.py
   python show_on_map_overlap.py
   ```

4. **Run Validation Analysis**:
   ```bash
   cd validation
   python community_validation_report.py
   python reports_aggregator.py
   python reports_analysis.py
   ```

### Detailed Workflow

#### Step 1: Data Preparation
- Place your GML network files in the `dataset/` directory
- Run the augmentation notebook to add geographic attributes
- Ensure data follows the expected format (nodes with lat/lon coordinates)

#### Step 2: Community Detection
- Configure algorithm parameters in each algorithm's Python file
- Run algorithms with different resolution parameters
- Results will be saved in respective `community_detection_outputs/` directories

#### Step 3: Visualization
- Use `show_on_map.py` for detailed community maps
- Use `show_on_map_overlap.py` for county-level analysis
- Maps will be saved as high-resolution PNG files

#### Step 4: Validation and Analysis
- Run validation reports to analyze community-state alignment
- Use the aggregator to combine results from all algorithms
- Generate comprehensive analysis figures and collages

---

## Configuration

### Algorithm Parameters

**Louvain** (`algorithms/louvain/louvain.py`):
- `RESOLUTION`: Resolution parameter (default: 2.0)
- `N_TRIALS`: Number of trials (default: 3)
- `SEED`: Random seed (default: 37)

**Leiden** (`algorithms/leiden/leiden.py`):
- `RESOLUTION`: Resolution parameter (default: 1.5)
- `N_TRIALS`: Number of trials (default: 3)
- `SEED`: Random seed (default: 37)

**BigCLAM** (`algorithms/bigCLAM/big_clam_like.py`):
- `K_LIST`: Number of communities to test (default: [40])
- `N_ITER`: Number of iterations (default: 80)
- `TAU`: Overlap threshold (default: 0.1)
- `HELDOUT_FRAC`: Test set fraction (default: 0.1)

### Visualization Settings

**Map Visualization** (`show_on_map/show_on_map.py`):
- `JSON_PATH`: Path to community detection results
- `GML_PATH`: Path to network data
- Map extent and styling can be customized

---

## Output Files

### Community Detection Results
- `*_map.json`: Node-to-community mappings
- `*_overlap_*.json`: Overlapping community assignments (BigCLAM only)

### Validation Reports
- `validation_report_on_*.csv`: Detailed per-community analysis
- `validation_report_on_*.json`: Summary statistics
- `all_validation_reports.csv`: Aggregated results from all algorithms

### Visualizations
- `community_map_*.png`: High-resolution community maps
- `county_level_summary.png`: County-level analysis
- `*_maps_collage.png`: Comparative map collages
- Various analysis charts in `validation/figures/`

---

## Results Summary

Our implementation successfully reproduces the original paper's findings while providing additional insights:

- **Community Detection Performance**: All three algorithms show strong geographic clustering
- **State Alignment**: Adjusted Rand Index scores of 0.467 (Mobility) and 0.423 (Communication)
- **Algorithm Comparison**: Leiden and Louvain show similar performance, BigCLAM provides overlapping communities
- **Geographic Validation**: Communities often align with state boundaries, with notable cross-boundary regions