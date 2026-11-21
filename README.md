[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

# Push-optimal room-split A* Sokoban solver

Solves every instance of a Sokoban set push-optimally: the level is split into rooms at the hallway, each room is solved independently with push-optimal A* (admissible consistent heuristic, live-square + frozen 2x2 pruning, transposition table), and the room solutions are stitched together with walks between rooms.

Practice set (kit scorer): **1,000,000 / 1,000,000** — all 5 instances solved at push-optimal P*. House baseline plain-astar scores 200,000 on the same set. Two runs write byte-identical output; a single run takes ~1 s of the 60 s budget.

## Run

```sh
node main.ts
```

Reads `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR/<id>/level.txt`) and writes `/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`). Node built-ins only; no network, no randomness, no clock reads.
