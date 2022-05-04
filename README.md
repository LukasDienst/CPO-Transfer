# CPO-Transfer
Implementation of RL algorithm "Constrained Policy Optimization" for OpenAI-Gym environments in the context of a Bachelor Thesis.

### Reproduzierbarkeit der BA-Ergebnisse
-> die beiden Ordner `PyTorch-CPO` und `gym` in einem Ordner

-> in Branch master liegt `PyTorch-CPO` und das Submodule `gym` (Originalcode (?))

-> in `gym/gym/envs/__init__.py` muss im HalfCheetah-v3-Bereich noch die maximale Schrittanzahl von 1000 auf 500 herabgesetzt werden
  -> dazu reichte mein git-Wissen am Mittwoch nicht aus, wie ich das Submodule verändern kann...
