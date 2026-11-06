#!/bin/sh
# Success rate versus number of demonstrations (ACT and BC), plus noisy-demo variants.
# Every run uses about 6,000 optimiser steps; evaluation is 100 seeded unseen episodes.
set -e
for n in 10 20 50 120; do
  frames=$((n * 122)); spe=$((frames / 256)); [ $spe -lt 1 ] && spe=1
  ep=$((6000 / spe))
  for pol in act bc; do
    ra-train --policy $pol --n-train-episodes $n --epochs $ep --out models/${pol}_n$n.pt > outputs/train_${pol}_n$n.log 2>&1
    ra-policy-eval --ckpt models/${pol}_n$n.pt --episodes 100 --out reports/eval_${pol}_n$n.json
  done
done
for pol in bc act; do
  ep=$([ $pol = act ] && echo 50 || echo 100)
  ra-train --policy $pol --demos data/demos_noisy.npz --epochs $ep --out models/${pol}_noisy.pt > outputs/train_${pol}_noisy.log 2>&1
  ra-policy-eval --ckpt models/${pol}_noisy.pt --episodes 300 --out reports/eval_${pol}_noisy.json
done
ra-policy-eval --ckpt models/act_clean.pt --episodes 300 --ensemble -1000 --out reports/eval_act_clean_newest.json
