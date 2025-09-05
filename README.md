# U.S. Social Fragmentation at Multiple Scales — Course Project (Complex Networks)

This repository documents our effort to study, reproduce, and contribute to:

**Hedayatifar, L., Rigg, R. A., Bar-Yam, Y., & Morales, A. J. (2019). _US social fragmentation at multiple scales_. Journal of the Royal Society Interface.**  
DOI: 10.1098/rsif.2019.0509

---

## Paper at a Glance

- **What the paper asks:** How fragmented is U.S. society across geographic scales, and do online interactions mirror offline movement?

- **Data (Twitter, geo-located):** ~87M tweets from ~2.8M users across the U.S. (Aug 22–Dec 25, 2013).  
  Two networks are built on a ~10 km geographic grid:
  - **Mobility network:** edges from consecutive tweets by the same user in different grid cells.
  - **Communication network:** edges from user mentions mapped to each user’s most recent location.

- **Method (multi-scale community structure):**
  - Louvain modularity optimization with a **resolution parameter** to reveal communities from **urban-scale** up to **national-scale**.
  - Robustness checks across algorithm runs; overlap analysis between mobility and communication partitions.
  - **Semantic validation:** location–hashtag matrix → TF-IDF → PCA → t-SNE to show communities also differ in content/interests.

- **Main findings:**
  - **Self-organized geographic patches** emerge consistently in both mobility and communication networks.
  - Boundaries **often align with state lines**, yet notable cross-boundary regions appear (e.g., some multi-state metro areas).
  - Online communication **reproduces** many offline mobility divisions, indicating strong coupling between physical and virtual social spaces.

- **Generative model (explanation):**
  - A network-growth model combining **geographical distance gravity**, **preferential attachment**, and **spatial growth**.
  - When gravity dominates, **spatial fragmentation** emerges; calibrated parameters reproduce empirical **degree distributions** and **modularity**.

---

## Our Scope in This Repo

- **Reproduce key analyses and figures** from the paper (networks, multi-scale communities, hashtag validation).
- **Contribute extensions** (e.g., robustness to grid size / resolution parameter, alternative community detection baselines, ablation of model components).

---

## Reference

Hedayatifar, L., Rigg, R. A., Bar-Yam, Y., & Morales, A. J. (2019). **US social fragmentation at multiple scales.** _J. R. Soc. Interface_, 16(159), 20190509. https://doi.org/10.1098/rsif.2019.0509
