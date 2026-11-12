# Serialization test

This is a small test I did to see if we can save a FabricPC model to a folder and load it back later, without the Python file that built it. Dr. Agrawal asked for this so a researcher can drop saved models in a folder and benchmark all of them with one command.

It is only a test. The real save/load should live inside FabricPC itself (`fabricpc/serialization.py`, like Matthew's plan in `docs/dev_plans_archive/model_checkpointing.md`), so both teams use the same code and nobody has to maintain a separate format.

## How it works

Each model is saved as a folder:

```
mnist-mlp-spc.fpc/
    structure.json   nodes, edges and settings of the graph
    params/          the weights, saved with Orbax
```

Loading reads `structure.json`, calls the normal FabricPC constructors and `graph()` again, then puts the weights back. Only `fabricpc.*` classes are allowed to load, and a newer file version gives an error instead of loading wrong. You can see an example in `example_structure_mnist-mlp-spc.json`.

## How to run

From the repo root, run the steps in order (each one is a separate process):

```
python scripts/serialization_test/step1_save.py
python scripts/serialization_test/step2_load.py
python scripts/serialization_test/step3_contract.py
python scripts/serialization_test/step4_orbax.py
```

Saved models go into `scripts/serialization_test/out/` (not committed).

## What I found

- `pickle` does not work on GraphStructure (`TypeError: cannot pickle 'mappingproxy' object`).
- Orbax alone saves only the weights, not the graph structure.
- With this test, 9 models (MLP spc/epc, muPC ResNet-8, VGG-5, ResNet-18, char transformer, autoencoder, Hopfield, and a small cyclic graph) loaded back in a new process with the same structure and exactly the same weights. Inference results also matched exactly for the 4 models I checked.
- Two classes can't be rebuilt from their own `.config` (`UniformInitializer` and `InferenceSchedule`), so the code builds those two by hand. That would be a small fix in FabricPC.
