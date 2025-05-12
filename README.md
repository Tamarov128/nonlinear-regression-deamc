# Nonlinear Regression using the Differential Evolution with Adaptive Mutation and Crossover strategies

This notebook tries to replicate the algorithm presented in the paper by Wongsa, Watchara & Puphasuk [1], which introduces the DEAMC algorithm.  

## The structure of the repository:
- `nist_models`: Contains data about 15 NIST nonlinear regression models.
  - `data`
    - `raw/`: Contains original .dat files from NIST.
    - `processed/`: Contains extracted observations data.
    - `models/nist_models.json`: Contains extracted model parameters.
  - `scripts`
    - `preprocess_data.py`: Functions to extract data from raw files.
  - `models` 
    - `nist_models.py`: NISTModel class: extracts data and provides model methods.

- `notebooks`:
  - `nonlinear_regression_models.ipynb`: Notebook which describes nonlinear regression and analyzes the extracted NIST model data [3-5].  

  - `differential_evolution_adaptive_mutation_crossover.ipynb`: Notebook which describes the DE and DEAMC algorithms, implements DEAMC and tests it on the regression data.  

- `presentation`:
  - Contains a presentation on the work done (ppt and pdf formats).

## References
1. Wongsa, Watchara & Puphasuk, Pikul & Wetweerapong, Jeerayut. (2024). *Differential evolution with adaptive mutation and crossover strategies for nonlinear regression problems*. Bulletin of Electrical Engineering and Informatics. 13. 3503-3514. 10.11591/eei.v13i5.6417.  

2. R. Storn and K. Price, *Differential Evolution–A simple and efficient heuristic for global optimization over continuous spaces*, Journal of Global Optimization, vol. 11, no. 4, pp. 341-359, 1997, doi: 10.1023/A:1008202821328.  

3. National Institute of Standards and Technology (NIST). (n.d.). *Nonlinear Least Squares Regression*. Retrieved May 10, 2025, from https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd142.htm  

4. National Institute of Standards and Technology (NIST). (n.d.). *Nonlinear Least Squares Regression Background Information*. Retrieved May 10, 2025, from https://www.itl.nist.gov/div898/strd/nls/nls_info.shtml  

5. National Institute of Standards and Technology (NIST). (n.d.). *Statistical Reference Datasets (STRD)*. Retrieved May 10, 2025, from https://www.itl.nist.gov/div898/strd/frames.html  