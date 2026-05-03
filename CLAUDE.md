1. Use nix shells for running commands
2. My computer has very low compute power. Use brev for anything with intensive CPU or GPU activity.
3. Wave/swarm PRs target `main` directly unless the wave depends on unmerged work in another branch. If a `swarm/*` branch absorbs another wave's PR (stacked PRs), open the trailing `swarm/* → main` PR in the same step — never leave a `swarm/*` branch ahead of `main` after a wave is "done". Before declaring a wave complete, verify its commits are reachable from `origin/main`, not just from a `swarm/*` branch. 
