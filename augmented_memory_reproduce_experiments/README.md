Augmented Memory (reproducing all experiments)

In the paper, 3 experiments were performed and instructions to reproduce them are presented below. The first thingrequired is to install the conda environment. Run the following command in the parent directory.

`source setup.py`

this will create a conda environment called `augmented_memory`.

Augmented Memory, like `REINVENT`, is run using input.py by passing a configuration JSON. All JSONs to reproduce the experiments in the Augmented Memory paper are provided in this folder. The only thing required is to change save paths in the JSONs.

The Prior used to reproduce Experiments 1 and 2 is provided here and is named `random.prior.new` which is from the ReinventCommunity repository (https://github.com/MolecularAI/ReinventCommunity/tree/master/notebooks/models).

---------------------------------------------------------------------------------------
Experiment 1: Aripiprazole Similarity
---------------------------------------------------------------------------------------
* In this folder, there is a folder named `aripiprazole` which has 2 sub-folders, `experience_replay` and `no_experience_replay`. In each sub-folder are configuration JSONs to run all the algorithms for this experiment. Running the experiments with and without experience replay will reproduce the plot in Figure 2a of the main paper.
* Note that all experiments were run for 300 epochs with batch size 64 (19,200 oracle calls). The exception is Best Agent Reminder (BAR) experiments which were run for 150 epochs as each epoch samples *2* batches.
* For this set of experiments, no paths need to be changed in the configuration JSONs provided. Once you have the conda environment (augmented_memory) activated, go to the `aripiprazole` folder and all JSONs should run by calling python ../../../input.py <whichever JSON you want to run>

---------------------------------------------------------------------------------------
Experiment 2: Practical Molecular Optimization (PMO) Benchmark
---------------------------------------------------------------------------------------
* These experiments were run using the great repository found here: https://github.com/wenhao-gao/mol_opt.
* All the code and instructions to reproduce the benchmark results are in the README in this repository.

---------------------------------------------------------------------------------------
Experiment 3: DRD2 Case Study
---------------------------------------------------------------------------------------
* This experiment requires the `DockStream` to perform AutoDock Vina docking. It can be found here: https://github.com/MolecularAI/DockStream
* Clone the repository and install the conda environment. There are 2 environment yml files in the `DockStream` repository. The `environment.yml` file is sufficient to reproduce Experiment 3. The installed conda environment is called DockStream. 
* AutoDock Vina can be downloaded here: https://vina.scripps.edu/downloads/. The experiments were run on a Linux machine so the autodock_vina_1_1_2_linux_x86.tgz file was downloaded
* All the configuration JSONs are in the `drd2` folder which have 2 sub-folders, `without_qed` and `with_qed`. The former is to reproduce the experimental results in Appendix E: Dopamine Type 2 Receptor (DRD2) Case Study: Exploiting AutoDock Vina. The latter reproduces the main paper experiments (Figure 3a). 
* Executing this experiment needs 2 configuration JSONs. 1 for Augmented Memory and 1 for the docking specifications. The paths here need to be changed to specify where you want to save the results. 
* Finally, the AutoDock Vina docking grid has been prepared and is provided in the `drd2` folder and is named `grid_with_waters.pdbqt`. It can be used as is.
* Note: the docking experiments were parallelized over 36 CPU cores as specified in the docking.json file. If you want to use more (although the overhead may make the overall run slower) or less CPUs, change the parameter in the JSON.
