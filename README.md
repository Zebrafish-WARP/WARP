# WARP — Whole-Brain Co-Mapping of Gene Expression and Neuronal Activity at Cellular Resolution in Behaving Zebrafish

> 
> Emmanuel Marquez-Legorreta\*¹, Greg M. Fleishman\*¹, Luuk W. Hesselink\*², Mark Eddison\*¹, Kasper Smeets²˒³, 
> Carsen Stringer¹, Philipp J. Keller¹, Sujatha Narayan¹, Alex B. Chen¹, 
> Brett D. Mensh¹, Scott M. Sternson⁴, Bernhard Englitz#², Paul W. Tillberg#¹,
> Misha B. Ahrens#¹˒⁵  
> 
> \* These authors contributed equally · # These authors contributed equally · ⁵ Lead contact
> 
> DOI: []()

[comment]: <> (![WARP overview figure]&#40;./github_main_figure.png&#41;)

This repository contains the **analysis and figure-generation code** used in the WARP manuscript. It is intended to help readers **reproduce manuscript figures** and **inspect intermediate analysis outputs** (e.g., LCD values, gene/cluster lists, and derived statistics).


---

## Related repositories
WARP is supported by several dedicated repositories that implement key parts of the processing pipeline. This repository focuses on **manuscript analysis + figure generation**, and typically consumes outputs produced by the components below.

- **Bigstream (volumetric image registration):** [https://github.com/JaneliaSciComp/bigstream](https://github.com/JaneliaSciComp/bigstream)  
  *Volumetric registration utilities used for aligning imaging volumes across modalities and/or imaging sessions.*


- **Distributed Cellpose (volumetric cell segmentation):** [https://github.com/MouseLand/cellpose ](https://github.com/MouseLand/cellpose )  
  *Used for volumetric cell segmentation found in `cellpose.contrib.distributed_cellpose`.*


- **Fishspot (RNA transcript spot detection):** [https://github.com/GFleishman/fishspot](https://github.com/GFleishman/fishspot)   
  *Detection of RNA transcript spots in volumetric images.*


- **SpotDMix (spot-to-cell assignment):** [https://github.com/Kepser/SpotDMix](https://github.com/Kepser/SpotDMix)  
  *Probabilistic assignment of detected RNA spots to segmented cells.*  
  *Preprint DOI: [10.64898/2025.12.15.693918](https://doi.org/10.64898/2025.12.15.693918)*


- **segmentNMF (cell-resolution activity extraction):** [https://github.com/L-Hess/segmentNMF](https://github.com/L-Hess/segmentNMF)  
  *Seeded demixed non-negative matrix factorization used to extract cell-resolution neuronal activity.*


- **fish (behavior + calcium analysis):** [https://github.com/L-Hess/segmentNMF](https://github.com/L-Hess/segmentNMF)  
  *Tools for analyzing behavioral time series and calcium imaging data.*

This repository does not contain full processing pipelines for each of the preprocessing steps. Individual notebooks running examples for each of these preprocessing steps can be found under `WARP/preprocessing_notebooks`.

---

## Data availability

All data generated in this study have been deposited at figshare:

- Dataset 1 (main 3 fish analyzed in paper):   
  *https://figshare.com/s/d1d19b105c4f74865c32*  


- Dataset 2 (additional 3 fish not analyzed in paper):  
  *https://figshare.com/s/72ceefe9844c1dda414a*  

Datasets include (as deposited):
- Activity traces and xyz locations for segmented cells for each fish  
- Gene expression profiles for each cell for each fish  
- Behavior traces  
- Stimulus time-series and information  
- Metadata (e.g., cell masks)
---

## Installation instructions

### 1. Clone the necessary repositories

```bash
git clone https://github.com/zebrafishWARP
```

The `janelia_core` repository contains supporting functionality for setting up models, visualization, etc.

### 2. Create and activate the conda environment

```bash
conda create -n warp python=3.10
conda activate warp
```

### 3. Install the dependencies

```bash
cd /path/to/WARP
pip install -e .
```

[comment]: <> (---)

[comment]: <> (## Repository layout)

[comment]: <> (We organize the repository into four top-level components:)

[comment]: <> (- **`WARP/WARP/`**  )

[comment]: <> (  Python module containing reusable **analysis + plotting functions** used across notebooks and figure generation.)

[comment]: <> (- **`WARP/figures/`**  )

[comment]: <> (  One **Jupyter notebook per manuscript figure** &#40;and optionally per supplement figure&#41;, producing the final panels.)

[comment]: <> (- **`WARP/notebooks/`**  )

[comment]: <> (  **Preprocessing / intermediate** notebooks that compute and cache core derived quantities &#40;e.g., LCD matrices/values, cluster assignments, summary tables&#41; used by figure notebooks.)

[comment]: <> (- **`WARP/preprocessing_notebooks/`**  )

[comment]: <> (  **Jupyter notebooks running examples for each of the data preprocessing steps.)

