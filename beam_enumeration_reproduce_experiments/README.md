---------------------------------------------------------------------------------------
Beam Enumeration (reproducing all experiments)
---------------------------------------------------------------------------------------

`Beam Enumeration` builds on `Augmented Memory` and thus shares similarities in the codebase.

In the paper, 2 experiments were performed and instructions to reproduce them are presented below. The first thing required is to install the conda environment. Run the following command in the parent directory. We use the same `conda` environment as `Augmented Memory` but change the name to `beam_enumeration` for the purpose of anonymous code release.

`source setup.py`

this will create a conda environment called `beam_enumeration`.

`Beam Enumeration` is a task-agnostic method that can be added directly on language-based molecular generative models. In the paper, we build on top of `Augmented Memory`. Therefore, executing `Beam Enumeration` with `Augmented Memory` follows the same procedure: passing a configuration JSON to `input.py`. All JSONs to reproduce the experiments in the Beam Enumeration paper are provided in this folder. The only thing required is to change save paths in the JSONs. Once the configuation JSON is ready, all experiments can be run with the following command:

`beam_enumeration/input.py <path to configuration JSON>`

All experiments can be visualized through Tensorboard which is installed in the `beam_enumeration` environment. All experiments output a `.log` file and can be visualized via the following command:

`tensorboard --logdir <.log file> --bind_all`

Note: The Prior (pre-trained model) used in the work is `random.prior.new` which is from the ReinventCommunity repository (https://github.com/MolecularAI/ReinventCommunity/tree/master/notebooks/models). The choice to use this Prior was because `Augmented Memory` and `REINVENT` experiments in previous publications used this Prior. Our results therefore isolate the effect of `Beam Enumeration` on both these algorithms and build on previous insights from literature.

---------------------------------------------------------------------------------------
Illustrative Experiment
---------------------------------------------------------------------------------------
* In this folder, there is a sub-folder named `illustrative-experiment` which contains 5 configuration JSONs to reproduce all results in Table 1 in the main text:
1. Baseline (`Augmented Memory`)
2. `Augmented Memory` + `Beam Enumeration` with `Structure` extraction 
3. `Augmented Memory` + `Beam Enumeration` with `Scaffold` extraction 
4. `Augmented Memory` + `Beam Enumeration` with `Structure` extraction and `Structure Minimum Size` = 15
5. `Augmented Memory` + `Beam Enumeration` with `Scaffold` extraction and `Structure Minimum Size` = 15

* For this set of experiments, no paths need to be changed in the configuration JSONs provided. Once you have the conda environment (`augmented_memory`) activated, go to the `illustrative-experiment` folder and all JSONs can be run using the following command:

`python ../../input.py <whichever JSON you want to run>`

---------------------------------------------------------------------------------------
Drug Discovery Case Studies
---------------------------------------------------------------------------------------
**Preliminaries**
* These experiments require `DockStream` to perform AutoDock Vina docking. It can be found here: https://github.com/MolecularAI/DockStream
* Clone the repository and install the conda environment. There are 2 environment yml files in the `DockStream` repository. The `environment.yml` file is sufficient to reproduce Experiment 3. The installed conda environment is called DockStream. 
* AutoDock Vina can be downloaded here: https://vina.scripps.edu/downloads/. The experiments were run on a Linux machine so the autodock_vina_1_1_2_linux_x86.tgz file was downloaded

**Reproducing the Experiments**
* In this folder, there are sub-folders for each drug discovery case study:
1. `drd2` (DRD2)
2. `mk2` (MK2 Kinase)
3. `ache` (AChE)

* In each of the sub-folders, there are 3 configuration JSONs to reproduce all results in Table 2 in the main text:
1. Baseline (`Augmented Memory`)
2. `Augmented Memory` + `Beam Enumeration` with `Structure` extraction and `Structure Minimum Size` = 15
3. `Augmented Memory` + `Beam Enumeration` with `Scaffold` extraction and `Structure Minimum Size` = 15

**Note: to reproduce the `REINVENT` experiments, the same JSON can be used and only the following parameter needs to be changed: `"augmented_memory": false` instead of `"augmented_memory": true`**

* All AutoDock Vina docking grids are provided in each sub-folder
* All drug discovery experiments require 2 JSONs: The configuration JSON and the docking.json found in each sub-folder. In both of these JSONs, some paths need to be changed to match your system (the paths that need to be changed are indicated by `< >`).

**Note: the docking experiments were parallelized over 36 CPU cores as specified in the docking.json file. If you want to use more (although the overhead may make the overall run slower) or less CPUs, change the `"number_cores"` parameter in the JSON.**
